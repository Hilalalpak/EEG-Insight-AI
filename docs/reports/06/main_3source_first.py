# api/main.py
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

eeg_collection = chroma_client.get_or_create_collection(name="eeg_insights",metadata={"hnsw:space": "cosine"})
medical_collection = chroma_client.get_or_create_collection(name="medical_definitions",metadata={"hnsw:space": "cosine"})
video_collection = chroma_client.get_or_create_collection(name="video_reasoning",metadata={"hnsw:space": "cosine"})

print("Connected to ChromaDB (3 collections).")

app = FastAPI(title="EEG RAG API")


class QueryRequest(BaseModel):
    query: str
    n_results: int = 5
    n_definitions: int = 3
    n_videos: int = 2  # 🆕
    filters: dict | None = None


def extract_patient_id(query: str) -> str | None:
    patterns = [
        r'patient\s+(?:id\s+)?(\d+)',
        r'patient_id[:\s]+(\d+)',
        r'\b(\d{10})\b']
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return match.group(1)
    return None


def extract_eeg_id(query: str) -> str | None:
    patterns = [
        r'eeg\s+(?:id\s+)?(\d+)',
        r'eeg[:\s]+(\d+)']
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return match.group(1)
    return None


# QUERY TYPE DETECTION
def detect_query_type(query: str) -> str:

    query_lower = query.lower()

    reasoning_keywords = ['how', 'why', 'explain','difference', 'compare', 'detect', 'analyze','step', 'process', 'method']
    definition_keywords = ['what is', 'define', 'definition', 'means']
    patient_keywords = ['patient', 'eeg', 'show', 'find', 'segment']

    if any(kw in query_lower for kw in reasoning_keywords):
        return 'reasoning'
    elif any(kw in query_lower for kw in definition_keywords):
        return 'definition'
    elif any(kw in query_lower for kw in patient_keywords):
        return 'patient_data'
    else:
        return 'general'


def call_llm(prompt: str, temperature: float = 0.2, max_tokens: int = 300) -> str:
    payload = {
        "model": "gemma:2b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 40,
            "num_predict": min(max_tokens, 300),
            "num_ctx": 2048,
            "num_thread": 4}}

    try:
        response = requests.post("http://eegi-ollama:11434/api/generate",json=payload,timeout=120)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"Error: {e}"


def hybrid_rag_pipeline(query: str, query_type: str, eeg_docs: list[str],medical_docs: list[str], video_docs: list[str]) -> tuple[str, dict]:

    medical_text = "\n".join(medical_docs[:1])[:800] if medical_docs else ""
    eeg_text = "\n".join(eeg_docs[:2])[:500] if eeg_docs else ""
    video_text = "\n".join(video_docs[:2])[:700] if video_docs else ""

    if query_type == 'reasoning':
        # Prioritize video reasoning
        prompt = f"""Expert reasoning from Dr. Hirsch:
{video_text}

Reference definition:
{medical_text[:300]}

Patient example:
{eeg_text[:300]}

Question: {query}

Answer (use expert's reasoning steps):"""

    elif query_type == 'definition':
        # Prioritize medical PDF
        prompt = f"""Medical definition (ACNS 2021):
{medical_text}

Clinical context from expert:
{video_text[:300]}

Question: {query}

Answer (define term clearly):"""

    elif query_type == 'patient_data':
        # Prioritize EEG data
        prompt = f"""Patient data:
{eeg_text}

Reference definition:
{medical_text[:300]}

Question: {query}

Answer (describe findings):"""

    else:
        # General - balanced
        prompt = f"""Medical reference: {medical_text[:400]}

Expert guidance: {video_text[:400]}

Patient data: {eeg_text[:400]}

Q: {query}

A:"""

    answer = call_llm(prompt, temperature=0.2, max_tokens=300)

    validation = {
        "passed": len(answer) > 50,
        "confidence": "HIGH" if len(answer) > 150 else "MEDIUM",
        "query_type": query_type,
        "sources_used": {
            "video": len(video_text) > 0,
            "medical": len(medical_text) > 0,
            "eeg": len(eeg_text) > 0}}

    return answer, validation


@app.post("/query")
def query_eeg_data(request: QueryRequest):
    try:
        # Extract IDs
        patient_id = extract_patient_id(request.query)
        eeg_id = extract_eeg_id(request.query)
        target_eeg_id = eeg_id or patient_id

        # DETECT QUERY TYPE
        query_type = detect_query_type(request.query)

        # Prepare filters
        where_filter = request.filters
        if target_eeg_id and not where_filter:
            where_filter = {"eeg_id": {"$eq": target_eeg_id}}

        # Generate embedding
        query_embedding = embed_model.encode(request.query, normalize_embeddings=True).tolist()

        # RETRIEVE FROM 3 COLLECTIONS

        # 1. EEG Data
        eeg_results = eeg_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(request.n_results, 5),
            where=where_filter,
            include=["documents", "metadatas", "distances"])

        # 2. Medical Definitions
        medical_results = medical_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(request.n_definitions, 2),
            include=["documents"])

        # 3. VIDEO REASONING
        video_results = video_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(request.n_videos, 2),
            include=["documents", "metadatas"])

        # Extract documents
        eeg_docs = eeg_results.get("documents", [[]])[0] if eeg_results else []
        medical_docs = medical_results.get("documents", [[]])[0] if medical_results else []
        video_docs = video_results.get("documents", [[]])[0] if video_results else []

        # Fallback: semantic search if patient not found
        if target_eeg_id and not eeg_docs:
            semantic_query = f"patient {target_eeg_id} EEG recording"
            semantic_embedding = embed_model.encode(semantic_query, normalize_embeddings=True).tolist()

            eeg_results = eeg_collection.query(
                query_embeddings=[semantic_embedding],
                n_results=5,
                include=["documents", "metadatas", "distances"])
            eeg_docs = eeg_results.get("documents", [[]])[0] if eeg_results else []

        # No data found
        if not eeg_docs and not medical_docs and not video_docs:
            return {
                "retrieved_eeg_segments": {"documents": [[]]},
                "retrieved_medical_definitions": {"documents": [[]]},
                "retrieved_video_reasoning": {"documents": [[]]},
                "llm_response": "No relevant data found.",
                "validation": {"confidence": "N/A"},
                "query_type": query_type}

        final_answer, validation_info = hybrid_rag_pipeline(
            request.query,
            query_type,
            eeg_docs=eeg_docs,
            medical_docs=medical_docs,
            video_docs=video_docs)

        return {
            "retrieved_eeg_segments": eeg_results,
            "retrieved_medical_definitions": medical_results,
            "retrieved_video_reasoning": video_results,
            "llm_response": final_answer,
            "validation": validation_info,
            "queried_eeg_id": target_eeg_id,
            "query_type": query_type }

    except Exception as e:
        return {"error": f"API error: {str(e)}"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "mode": "mac_air_3_source_rag",
        "collections": {
            "eeg_insights": eeg_collection.count(),
            "medical_definitions": medical_collection.count(),
            "video_reasoning": video_collection.count() }}