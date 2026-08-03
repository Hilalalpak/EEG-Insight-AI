# api/main_adaptive.py
import re
import requests
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

print("Loading embedding model...")
embed_model = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
print("Embedding model loaded.")

print("Connecting to ChromaDB...")
chroma_client = chromadb.HttpClient(host="eegi-chroma", port=8000)

eeg_collection = chroma_client.get_or_create_collection(name="eeg_insights", metadata={"hnsw:space": "cosine"})
medical_collection = chroma_client.get_or_create_collection(name="medical_definitions_multisize",
                                                            metadata={"hnsw:space": "cosine"})
print("Connected to ChromaDB and both collections.")

app = FastAPI(title="EEG RAG API - Adaptive")


class QueryRequest(BaseModel):
    query: str
    n_results: int = 5
    n_definitions: int = 3
    filters: dict | None = None


def get_query_config(query: str) -> dict:
    """Query tipine göre chunk sayısı ve boyutu döndür"""
    q = query.lower()

    # Comparison: daha fazla chunk gerekli
    if any(w in q for w in ['compare', 'difference', 'versus', 'vs', 'distinguish']):
        return {'med_n': 2, 'med_chars': 1000, 'eeg_n': 3, 'eeg_chars': 800, 'strategy': 'large'}

    # Definition: az chunk yeterli
    if any(w in q for w in ['what is', 'define', 'explain', 'means']):
        return {'med_n': 1, 'med_chars': 800, 'eeg_n': 2, 'eeg_chars': 600, 'strategy': 'small'}

    # Default: orta
    return {'med_n': 2, 'med_chars': 900, 'eeg_n': 3, 'eeg_chars': 800, 'strategy': 'small'}


def extract_patient_id(query: str) -> str | None:
    patterns = [r'patient\s+(?:id\s+)?(\d+)', r'patient_id[:\s]+(\d+)', r'\b(\d{10})\b']
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return match.group(1)
    return None


def extract_eeg_id(query: str) -> str | None:
    patterns = [r'eeg\s+(?:id\s+)?(\d+)', r'eeg[:\s]+(\d+)']
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return match.group(1)
    return None


def call_llm(prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
    payload = {
        "model": "gemma:2b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 40,
            "num_predict": max_tokens,
            "num_ctx": 3072,
            "num_thread": 4}}

    try:
        response = requests.post("http://eegi-ollama:11434/api/generate", json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"Error: {e}"


def hybrid_rag_pipeline(query: str, eeg_docs: list[str], medical_docs: list[str], config: dict) -> tuple[str, dict]:
    """Adaptive context sizing"""

    medical_text = "\n".join(medical_docs[:config['med_n']])[:config['med_chars']] if medical_docs else ""
    eeg_text = "\n".join(eeg_docs[:config['eeg_n']])[:config['eeg_chars']] if eeg_docs else ""

    prompt = f"""Medical info: {medical_text}

Patient data: {eeg_text}

Q: {query}

A (explain term + describe findings):"""

    answer = call_llm(prompt, temperature=0.2, max_tokens=512)

    validation_info = {
        "passed": len(answer) > 50,
        "confidence": "HIGH" if len(answer) > 150 else "MEDIUM",
        "chunks": f"med:{config['med_n']},eeg:{config['eeg_n']}",
        "corrected": False}

    return answer, validation_info


@app.post("/query")
def query_eeg_data(request: QueryRequest):
    try:
        # 1. Query config
        config = get_query_config(request.query)

        # 2. Extract IDs
        patient_id = extract_patient_id(request.query)
        eeg_id = extract_eeg_id(request.query)
        target_eeg_id = eeg_id or patient_id

        where_filter = request.filters if request.filters else None
        if target_eeg_id and not where_filter:
            where_filter = {"eeg_id": {"$eq": target_eeg_id}}

        query_embedding = embed_model.encode(request.query, normalize_embeddings=True).tolist()

        # 3. Query with strategy filter
        medical_filter = {"strategy": config['strategy']}

        eeg_results = eeg_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(config['eeg_n'] + 1, 6),
            where=where_filter,
            include=["documents", "metadatas", "distances"])

        medical_results = medical_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(config['med_n'] + 1, 4),
            where=medical_filter,
            include=["documents", "metadatas"])

        eeg_docs = eeg_results.get("documents", [[]])[0] if eeg_results else []
        medical_docs = medical_results.get("documents", [[]])[0] if medical_results else []

        # 4. Fallback
        if target_eeg_id and not eeg_docs:
            semantic_query = f"patient {target_eeg_id} EEG recording"
            semantic_embedding = embed_model.encode(semantic_query, normalize_embeddings=True).tolist()

            eeg_results = eeg_collection.query(
                query_embeddings=[semantic_embedding],
                n_results=5,
                include=["documents", "metadatas", "distances"])

            eeg_docs = eeg_results.get("documents", [[]])[0] if eeg_results else []

            if not eeg_docs:
                return {
                    "retrieved_eeg_segments": {"documents": [[]], "metadatas": [[]], "distances": [[]]},
                    "retrieved_medical_definitions": medical_results,
                    "llm_response": f"No data found for EEG/Patient {target_eeg_id}.",
                    "validation": {"confidence": "N/A", "corrected": False}}

        if not eeg_docs and not medical_docs:
            return {
                "retrieved_eeg_segments": {"documents": [[]], "metadatas": [[]], "distances": [[]]},
                "retrieved_medical_definitions": {"documents": [[]]},
                "llm_response": "No relevant data found.",
                "validation": {"confidence": "N/A", "corrected": False}}

        # 5. Generate answer
        final_answer, validation_info = hybrid_rag_pipeline(
            request.query,
            eeg_docs=eeg_docs,
            medical_docs=medical_docs,
            config=config)

        return {
            "retrieved_eeg_segments": eeg_results,
            "retrieved_medical_definitions": medical_results,
            "llm_response": final_answer,
            "validation": {
                "confidence": validation_info["confidence"],
                "chunks_used": validation_info["chunks"],
                "corrected": validation_info["corrected"]},
            "queried_eeg_id": target_eeg_id}

    except Exception as e:
        return {"error": f"API error: {str(e)}"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "mode": "adaptive_chunking"}