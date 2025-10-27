# src/api/app.py
import os
import logging
import requests

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder

from infrastructure.conf.config_loader import ConfigLoader
from infrastructure.conf.interfaces import (DBConfigInterface, LLMConfigInterface, RAGCoreConfigInterface, LoggingConfigInterface)

from src.rag.core.logging_config import LoggingConfig
from src.api.search_indices import load_collections, build_bm25_indices
from src.api.query_utils import find_patient_id, find_eeg_id, get_query_type
from src.api.llm_utils import run_rag_chain
from src.api.retrieval import HybridRetriever


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_PATHS = {
    "base": os.path.join(BASE_DIR, "infrastructure/conf/base.yml"),
    "rag_strategy": os.path.join(BASE_DIR, "infrastructure/conf/rag_strategy.yaml"),
    "models": os.path.join(BASE_DIR, "infrastructure/conf/llm/models.yml"),
    "llm_env": os.path.join(BASE_DIR, "infrastructure/conf/llm/env_dev.yaml"),
    "signal": os.path.join(BASE_DIR, "infrastructure/conf/pipeline/signal.yml"),
    "data_sources": os.path.join(BASE_DIR, "infrastructure/conf/pipeline/data_sources.yml")}

try:
    loader = ConfigLoader(config_paths=CONFIG_PATHS)
    db_config: DBConfigInterface = loader.get_db_config()
    llm_config: LLMConfigInterface = loader.get_llm_config()
    rag_config: RAGCoreConfigInterface = loader.get_rag_core_config()
    log_config: LoggingConfigInterface = loader.get_logging_config()

except Exception as e:
    print(f"Config load failed: {e}")
    exit(1)

LoggingConfig.setup_logging(log_config)
logger = logging.getLogger(__name__)

# Load models into memory
logger.info("Loading embedding model (SapBERT)...")
embed_model_name = llm_config.get_embedding_model_name()
embed_model = SentenceTransformer(embed_model_name)
logger.info("Embedding model ready.")

collections = load_collections(host=db_config.get_chroma_host(), port=db_config.get_chroma_port(), db_config=db_config)
bm25_indices, all_docs = build_bm25_indices(collections)

signal_collection = collections.get("signal")
document_collection = collections.get("document")
transcript_collection = collections.get("transcript")

logger.info("Loading reranker model (BGE)...")
reranker_model_name = llm_config.get_reranker_model_name()
rerank_model = CrossEncoder(reranker_model_name)
logger.info("Reranker model ready.")

# Init hybrid retriever
retriever = HybridRetriever(
    embed_model=embed_model,
    rerank_model=rerank_model,
    signal_collection=signal_collection,
    document_collection=document_collection,
    transcript_collection=transcript_collection,
    signal_bm25=bm25_indices.get("signal"),
    document_bm25=bm25_indices.get("document"),
    transcript_bm25=bm25_indices.get("transcript"),
    signal_all_docs=all_docs.get("signal", []),
    document_all_docs=all_docs.get("document", []),
    transcript_all_docs=all_docs.get("transcript", []),
    rrf_k=rag_config.get_rrf_k())

logger.info("API ready for requests.")
app = FastAPI(title="EEG RAG API")

class QueryRequest(BaseModel):
    """Pydantic model for the query payload. Typing is required here."""
    query: str
    n_results: int = 5
    n_definitions: int = 3
    n_videos: int = 2
    filters: dict | None = None

@app.post("/query")
def query_endpoint(request: QueryRequest):
    try:
        logger.info(f"New query: '{request.query[:80]}...'")

        # Parse query
        patient_id = find_patient_id(request.query)
        eeg_id = find_eeg_id(request.query)
        target_id = eeg_id or patient_id
        q_type = get_query_type(request.query)

        where_filter = request.filters
        if target_id and not where_filter:
            # Find an ID, auto-filter signals for it
            where_filter = {"eeg_id": {"$eq": target_id}}

        n_search = rag_config.get_n_search()
        n_final_s = min(request.n_results, rag_config.get_n_final_signal())
        n_final_d = min(request.n_definitions, rag_config.get_n_final_document())
        n_final_v = min(request.n_videos, rag_config.get_n_final_transcript())

        #Retrieve docs
        retrieved_docs = retriever.search(query=request.query, n_search=n_search, n_final_signal=n_final_s,
                                          n_final_document=n_final_d, n_final_transcript=n_final_v,
                                          where_filter=where_filter)

        signal_docs = retrieved_docs["signal"]
        document_docs = retrieved_docs["document"]
        transcript_docs = retrieved_docs["transcript"]

        if target_id and not signal_docs:
            logger.warning(f"No signal data for EEG/Patient {target_id}")
            return {
                "retrieved_signal_segments": {"documents": [[]]},
                "retrieved_document_chunks": {"documents": [[]]},
                "retrieved_transcript_chunks": {"documents": [[]]},
                "llm_response": f"No data found for EEG/Patient {target_id} after hybrid search.",
                "validation": {"confidence": "N/A"},
                "query_type": q_type}

        if not signal_docs and not document_docs and not transcript_docs:
            logger.warning("No results from hybrid search")
            return {
                "retrieved_signal_segments": {"documents": [[]]},
                "retrieved_document_chunks": {"documents": [[]]},
                "retrieved_transcript_chunks": {"documents": [[]]},
                "llm_response": "No relevant data found after hybrid search.",
                "validation": {"confidence": "N/A"},
                "query_type": q_type}

        final_answer, validation_info = run_rag_chain(llm_config=llm_config,
                                                      query=request.query,
                                                      query_type=q_type,
                                                      signal_docs=signal_docs,
                                                      document_docs=document_docs,
                                                      transcript_docs=transcript_docs)

        logger.info(f"Query done: type={q_type}, conf={validation_info['confidence']}")

        # FIXME: `{"documents": [list]}` review
        return {
            "retrieved_signal_segments": {"documents": [signal_docs]},
            "retrieved_document_chunks": {"documents": [document_docs]},
            "retrieved_transcript_chunks": {"documents": [transcript_docs]},
            "llm_response": final_answer,
            "validation": validation_info,
            "queried_eeg_id": target_id,
            "query_type": q_type}

    except Exception as e:
        logger.error(f"Query endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal API error")

@app.get("/health")
def health_check():
    try:
        # ChromaDB check
        counts = {
            "signal": signal_collection.count() if signal_collection else "Not Loaded",
            "document": document_collection.count() if document_collection else "Not Loaded",
            "transcript": transcript_collection.count() if transcript_collection else "Not Loaded"}

        chroma_status = "ok" if all(collections.values()) else "partial"

        # Ollama check
        try:
            ollama_endpoint = llm_config.get_ollama_endpoint()
            health_url = ollama_endpoint.rstrip('/') + '/'
            requests.get(health_url, timeout=5).raise_for_status()
            ollama_status = "ok"
        except:
            ollama_status = "error"

        return {
            "status": "healthy" if chroma_status == "ok" and ollama_status == "ok" else "degraded",
            "mode": "3_source_hybrid_reranked_rag",
            "collections": counts,
            "dependencies": {
                "chromadb": chroma_status,
                "ollama": ollama_status},
            "models": {
                "embedding": embed_model_name,
                "reranker": reranker_model_name}}
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {"status": "unhealthy", "error": str(e)}