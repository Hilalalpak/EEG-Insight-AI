# api/main.py
import re
import requests
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder

from rank_bm25 import BM25Okapi
from hybridizer import reciprocal_rank_fusion

print("Loading embedding model...")
embed_model = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
print("Embedding model loaded.")

print("Connecting to ChromaDB...")
chroma_client = chromadb.HttpClient(host="eegi-chroma", port=8000)

eeg_collection = chroma_client.get_or_create_collection(name="eeg_insights",metadata={"hnsw:space": "cosine"})
medical_collection = chroma_client.get_or_create_collection(name="medical_definitions",metadata={"hnsw:space": "cosine"})
video_collection = chroma_client.get_or_create_collection(name="video_reasoning",metadata={"hnsw:space": "cosine"})

print("Connected to ChromaDB (3 collections).")


# ==========================================================
# HYBRID SEARCH (BM25) SETUP
# Fetch all documents and build BM25 indices (for sparse retrieval)
print("Loading all documents and indexing for BM25 (Sparse Index)...")

# 1. EEG Data BM25 setup
eeg_all_docs = eeg_collection.get(include=['documents'])['documents'][0]
eeg_tokenized_corpus = [doc.lower().split(" ") for doc in eeg_all_docs]
eeg_bm25 = BM25Okapi(eeg_tokenized_corpus)

# 2. Medical Definitions BM25 setup
medical_all_docs = medical_collection.get(include=['documents'])['documents'][0]
medical_tokenized_corpus = [doc.lower().split(" ") for doc in medical_all_docs]
medical_bm25 = BM25Okapi(medical_tokenized_corpus)

# 3. Video Reasoning BM25 setup
video_all_docs = video_collection.get(include=['documents'])['documents'][0]
video_tokenized_corpus = [doc.lower().split(" ") for doc in video_all_docs]
video_bm25 = BM25Okapi(video_tokenized_corpus)

print("BM25 Indices Ready for Hybrid Search.")
# ==========================================================

# --- NEW ADDITION ---
print("Loading Reranker model...")
# Load BGE-reranker-base as a CrossEncoder
rerank_model = CrossEncoder('BAAI/bge-reranker-base')
print("Reranker model loaded.")
# --- END OF NEW ADDITION ---

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


def apply_reranking(query: str, documents: list[str], final_count: int) -> list[str]:
    """Re-ranks documents against the query using the BGE reranker."""
    if not documents or final_count == 0:
        return []

    # The reranker takes the full RRF candidate list as-is and picks the best N

    # Build (query, document) pairs
    sentence_pairs = [[query, doc] for doc in documents]

    # Score each pair
    scores = rerank_model.predict(sentence_pairs)

    # Zip scores with documents and sort descending
    doc_scores = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

    # Return the top N docs
    return [doc for doc, score in doc_scores][:final_count]


@app.post("/query")
def query_eeg_data(request: QueryRequest):
    try:
        # Extract IDs, Query Type, and Filters
        patient_id = extract_patient_id(request.query)
        eeg_id = extract_eeg_id(request.query)
        target_eeg_id = eeg_id or patient_id
        query_type = detect_query_type(request.query)

        where_filter = request.filters
        if target_eeg_id and not where_filter:
            where_filter = {"eeg_id": {"$eq": target_eeg_id}}

        # Prepare query embedding and BM25 tokenization
        query_embedding = embed_model.encode(request.query, normalize_embeddings=True).tolist()
        query_tokenized = request.query.lower().split(" ")  # for BM25

        # Cast a wider net — fetch more candidates before fusion
        N_SEARCH = 15  # max results per retriever (dense + sparse)

        # Final chunk counts to send to the LLM
        N_FINAL_EEG = min(request.n_results, 5)
        N_FINAL_MEDICAL = min(request.n_definitions, 2)
        N_FINAL_VIDEO = min(request.n_videos, 2)

        # ==========================================================
        # 1. DENSE RETRIEVAL (SapBERT + Chroma)
        # ==========================================================

        # 1.1. EEG Data (Dense)
        eeg_dense_results = eeg_collection.query(
            query_embeddings=[query_embedding], n_results=N_SEARCH,
            where=where_filter, include=["documents"])
        eeg_dense_list = [(doc, 0.0) for doc in eeg_dense_results.get("documents", [[]])[0]]

        # 1.2. Medical Definitions (Dense)
        medical_dense_results = medical_collection.query(
            query_embeddings=[query_embedding], n_results=N_SEARCH,
            include=["documents"])
        medical_dense_list = [(doc, 0.0) for doc in medical_dense_results.get("documents", [[]])[0]]

        # 1.3. Video Reasoning (Dense)
        video_dense_results = video_collection.query(
            query_embeddings=[query_embedding], n_results=N_SEARCH,
            include=["documents"])
        video_dense_list = [(doc, 0.0) for doc in video_dense_results.get("documents", [[]])[0]]

        # ==========================================================
        # 2. SPARSE RETRIEVAL (BM25)
        # ==========================================================

        # 2.1. EEG Data (Sparse)
        eeg_sparse_scores = eeg_bm25.get_scores(query_tokenized)
        eeg_sparse_indices = (-eeg_sparse_scores).argsort()[:N_SEARCH]
        eeg_sparse_list = [(eeg_all_docs[i], eeg_sparse_scores[i]) for i in eeg_sparse_indices]

        # 2.2. Medical Definitions (Sparse)
        medical_sparse_scores = medical_bm25.get_scores(query_tokenized)
        medical_sparse_indices = (-medical_sparse_scores).argsort()[:N_SEARCH]
        medical_sparse_list = [(medical_all_docs[i], medical_sparse_scores[i]) for i in medical_sparse_indices]

        # 2.3. Video Reasoning (Sparse)
        video_sparse_scores = video_bm25.get_scores(query_tokenized)
        video_sparse_indices = (-video_sparse_scores).argsort()[:N_SEARCH]
        video_sparse_list = [(video_all_docs[i], video_sparse_scores[i]) for i in video_sparse_indices]

        # ==========================================================
        # 3. HYBRID FUSION (RECIPROCAL RANK FUSION - RRF)
        # ==========================================================

        # RRF merges dense and sparse results and re-ranks them.

        eeg_fused_docs = reciprocal_rank_fusion(eeg_dense_list, eeg_sparse_list)[:N_FINAL_EEG]
        medical_fused_docs = reciprocal_rank_fusion(medical_dense_list, medical_sparse_list)[:N_FINAL_MEDICAL]
        video_fused_docs = reciprocal_rank_fusion(video_dense_list, video_sparse_list)[:N_FINAL_VIDEO]

        # ==========================================================
        # 4. CROSS-ENCODER RERANKING
        # ==========================================================

        # Feed the RRF candidates into the reranker and pick the final top-N chunks.

        # NOTE: The reranker evaluates ALL chunks coming from RRF (N_SEARCH=15 in this example)
        # and picks the best N_FINAL. This is the most effective way to filter out noise.

        eeg_docs = apply_reranking(request.query, eeg_fused_docs, N_FINAL_EEG)
        medical_docs = apply_reranking(request.query, medical_fused_docs, N_FINAL_MEDICAL)
        video_docs = apply_reranking(request.query, video_fused_docs, N_FINAL_VIDEO)

        # NOTE: results are no longer full ChromaDB objects.
        # Build simplified dicts for backwards compatibility with the response schema.

        eeg_results_simple = {"documents": [eeg_docs]}
        medical_results_simple = {"documents": [medical_docs]}
        video_results_simple = {"documents": [video_docs]}

        # Fallback: if a patient ID was specified but no data came back after hybrid search, bail early.
        if target_eeg_id and not eeg_docs:
            # Keeping the fallback simple for now
            return {
                "retrieved_eeg_segments": {"documents": [[]]},
                "retrieved_medical_definitions": {"documents": [[]]},
                "retrieved_video_reasoning": {"documents": [[]]},
                "llm_response": f"No data found for EEG/Patient {target_eeg_id} after hybrid search.",
                "validation": {"confidence": "N/A"},
                "query_type": query_type}

        # No data found
        if not eeg_docs and not medical_docs and not video_docs:
            return {
                "retrieved_eeg_segments": {"documents": [[]]},
                "retrieved_medical_definitions": {"documents": [[]]},
                "retrieved_video_reasoning": {"documents": [[]]},
                "llm_response": "No relevant data found after hybrid search.",
                "validation": {"confidence": "N/A"},
                "query_type": query_type}

        final_answer, validation_info = hybrid_rag_pipeline(
            request.query,
            query_type,
            eeg_docs=eeg_docs,
            medical_docs=medical_docs,
            video_docs=video_docs)

        return {
            "retrieved_eeg_segments": eeg_results_simple,
            "retrieved_medical_definitions": medical_results_simple,
            "retrieved_video_reasoning": video_results_simple,
            "llm_response": final_answer,
            "validation": validation_info,
            "queried_eeg_id": target_eeg_id,
            "query_type": query_type}

    except Exception as e:
        return {"error": f"API error: {str(e)}"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "mode": "mac_air_3_source_hybrid_reranked_rag",
        "collections": {
            "eeg_insights": eeg_collection.count(),
            "medical_definitions": medical_collection.count(),
            "video_reasoning": video_collection.count() }}