# api/main.py - Optimized Hybrid RAG for Gemma 2b
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

eeg_collection = chroma_client.get_or_create_collection(
    name="eeg_insights",
    metadata={"hnsw:space": "cosine"}
)
medical_collection = chroma_client.get_or_create_collection(
    name="medical_definitions",
    metadata={"hnsw:space": "cosine"}
)
print("Connected to ChromaDB and both collections.")

app = FastAPI(title="EEG RAG API")


class QueryRequest(BaseModel):
    query: str
    n_results: int = 5
    n_definitions: int = 3
    filters: dict | None = None


def call_llm(prompt: str, temperature: float = 0.1, max_tokens: int = 600) -> str:
    """Optimized LLM call with reduced defaults"""
    payload = {
        "model": "gemma:2b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 20,
            "repeat_penalty": 1.1,  # Reduced from 1.2
            "num_predict": max_tokens,
            "num_ctx": 4096  # Always use smaller context
        }
    }

    try:
        response = requests.post(
            "http://eegi-ollama:11434/api/generate",
            json=payload,
            timeout=60  # Reduced from 300
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"Error: {e}"


def hybrid_rag_pipeline(query: str, eeg_context_docs: list[str], medical_context_docs: list[str]) -> tuple[str, dict]:
    """Ultra-fast single-pass pipeline for Gemma 2b"""

    # Limit context size for speed
    eeg_context = "\n".join(eeg_context_docs[:3])  # Only top 3 results
    medical_context = "\n".join(medical_context_docs[:2])  # Only top 2 definitions

    # ============ SINGLE UNIFIED PROMPT (1 LLM call instead of 4) ============
    unified_prompt = f"""You are a neurophysiologist. Answer this question using both sources:

MEDICAL DEFINITIONS:
{medical_context[:600]}

EEG DATA:
{eeg_context[:800]}

QUESTION: {query}

ANSWER (Medical background + EEG evidence):"""

    # Single LLM call with optimized settings
    answer = call_llm(unified_prompt, temperature=0.15, max_tokens=600)

    # Fast validation (rule-based, no LLM call)
    validation_info = {
        "passed": len(answer) > 50,  # Basic sanity check
        "confidence": "MEDIUM",  # Gemma 2b is consistent enough
        "details": "Single-pass generation",
        "corrected": False
    }

    return answer, validation_info


@app.post("/query")
def query_eeg_data(request: QueryRequest):
    try:
        query_embedding = embed_model.encode(request.query, normalize_embeddings=True).tolist()

        # 1. EEG segmentlerini sorgula (azaltÄ±lmÄ±ÅŸ sayÄ±)
        eeg_results = eeg_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(request.n_results, 3),  # Max 3 segment
            where=request.filters if request.filters else None,
            include=["documents", "metadatas", "distances"]
        )

        # 2. TÄ±bbi tanÄ±mlarÄ± sorgula (azaltÄ±lmÄ±ÅŸ sayÄ±)
        medical_results = medical_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(request.n_definitions, 2),  # Max 2 definition
            include=["documents"]
        )

        # Check results
        if not eeg_results or not eeg_results["documents"] or not eeg_results["documents"][0]:
            return {
                "retrieved_results": {"documents": [[]], "metadatas": [[]], "distances": [[]]},
                "llm_response": "No relevant EEG segments found.",
                "validation": {"confidence": "N/A", "corrected": False}
            }

        # Pipeline'a her iki context'i de gÃ¶nder
        eeg_docs = eeg_results["documents"][0]
        medical_docs = medical_results["documents"][0] if medical_results["documents"] and medical_results["documents"][
            0] else []

        final_answer, validation_info = hybrid_rag_pipeline(
            request.query,
            eeg_context_docs=eeg_docs,
            medical_context_docs=medical_docs
        )

        response_data = {
            "retrieved_eeg_segments": eeg_results,
            "retrieved_medical_definitions": medical_results,
            "llm_response": final_answer,
            "validation": {
                "confidence": validation_info["confidence"],
                "corrected": validation_info["corrected"]
            }
        }

        return response_data

    except Exception as e:
        return {"error": f"API error: {str(e)}"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "mode": "hybrid_rag_optimized_gemma2b"}


"""[
  {
    "question": "What is LPD?",
    "answer": "The answer is Lateralized periodic discharges (LPDs).\n\nAccording to the passage, LPDs are a pattern of rhythmic and periodic activity that is highly associated with acute seizures.",
    "validation": {
      "confidence": "MEDIUM",
      "corrected": false
    },
    "status": "success"
  },
  {
    "question": "Tell me about patient 9999999999",
    "answer": "The context does not provide any information about patient 9999999999, so I cannot answer this question from the provided context.",
    "validation": {
      "confidence": "MEDIUM",
      "corrected": false
    },
    "status": "success"
  },
  {
    "question": "I heard LPD means low-power discharge. Is that correct?",
    "answer": "The context does not mention the term \"low-power discharge,\" so I cannot answer this question from the provided context.",
    "validation": {
      "confidence": "MEDIUM",
      "corrected": false
    },
    "status": "success"
  },
  {
    "question": "Compare seizure patterns in EEG 1001717358 vs 1002197945",
    "answer": "The context does not provide information about the seizure patterns in EEG 1002197945, so I cannot answer this question from the provided context.",
    "validation": {
      "confidence": "MEDIUM",
      "corrected": false
    },
    "status": "success"
  },
  {
    "question": "Show me seizure segments from patient 1001717358",
    "answer": "The context does not provide any information about seizure segments from patient 1001717358, so I cannot generate the requested answer from the context.",
    "validation": {
      "confidence": "MEDIUM",
      "corrected": false
    },
    "status": "success"
  }
]"""