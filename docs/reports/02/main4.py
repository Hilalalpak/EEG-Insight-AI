# api/main.py - Hybrid RAG (Two-Step + Self-Validation)
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
collection = chroma_client.get_or_create_collection(
    name="eeg_insights",
    metadata={"hnsw:space": "cosine"}
)
print("Connected to ChromaDB.")

app = FastAPI(title="EEG RAG API")


class QueryRequest(BaseModel):
    query: str
    n_results: int = 5
    filters: dict | None = None


def call_llm(prompt: str, temperature: float = 0.1, max_tokens: int = 1024) -> str:
    """Single LLM call function to avoid repetition"""
    payload = {
        "model": "gemma:2b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 20,
            "repeat_penalty": 1.2,
            "num_predict": max_tokens,
            "num_ctx": 8192 if max_tokens > 512 else 4096
        }
    }

    try:
        response = requests.post(
            "http://eegi-ollama:11434/api/generate",
            json=payload,
            timeout=180
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"Error: {e}"


def hybrid_rag_pipeline(query: str, context_docs: list[str]) -> tuple[str, dict]:
    """
    Hybrid RAG: Two-Step Summarization + Self-Validation
    Returns: (final_answer, validation_info)
    """

    full_context = "\n\n---\n\n".join(context_docs)

    # ============ STEP 1: SUMMARIZE (Filter Noise) ============
    summary_prompt = f"""Extract only relevant information for this question.

QUESTION: {query}

EEG SEGMENTS:
{full_context}

Return only key facts (patient IDs, measurements, classifications). Max 300 words.

RELEVANT FACTS:"""

    summary = call_llm(summary_prompt, temperature=0.0, max_tokens=512)

    # ============ STEP 2: ANSWER (Medical Knowledge + Data) ============
    answer_prompt = f"""You are a clinical neurophysiologist. Answer using:
1. Your medical knowledge (definitions, clinical interpretation)
2. The specific EEG data below

Structure:
## Medical Background
[General medical explanation]

## Evidence from EEG Data
[Specific findings from data]

EEG DATA:
{summary}

QUESTION: {query}

ANSWER:"""

    answer = call_llm(answer_prompt, temperature=0.2, max_tokens=1024)

    # ============ STEP 3: VALIDATE (Medical Accuracy Check) ============
    validation_prompt = f"""Review this answer for medical accuracy.

QUESTION: {query}
EEG DATA: {summary}

ANSWER TO VALIDATE:
{answer}

Check:
1. Medical terminology correct?
2. Claims supported by data or standard medical knowledge?
3. Any hallucinations?

Respond ONLY:
STATUS: [PASS/FAIL]
CONFIDENCE: [HIGH/MEDIUM/LOW]
ISSUE: [Describe problem or write "None"]"""

    validation_text = call_llm(validation_prompt, temperature=0.0, max_tokens=256)

    # Parse validation
    passed = "STATUS: PASS" in validation_text.upper()
    confidence = "HIGH" if "HIGH" in validation_text.upper() else \
        "MEDIUM" if "MEDIUM" in validation_text.upper() else "LOW"

    validation_info = {
        "passed": passed,
        "confidence": confidence,
        "details": validation_text
    }

    # ============ STEP 4: CORRECT (If Validation Failed) ============
    if not passed or confidence == "LOW":
        correction_prompt = f"""The previous answer has issues. Rewrite correctly.

QUESTION: {query}
EEG DATA: {summary}

PREVIOUS ANSWER (with errors):
{answer}

VALIDATION FEEDBACK:
{validation_text}

CORRECTED ANSWER:"""

        corrected_answer = call_llm(correction_prompt, temperature=0.1, max_tokens=1024)

        # Add disclaimer
        final_answer = f"""⚠️ *Note: Response was refined for medical accuracy*

{corrected_answer}"""

        validation_info["corrected"] = True
    else:
        final_answer = answer
        validation_info["corrected"] = False

    return final_answer, validation_info


@app.post("/query")
def query_eeg_data(request: QueryRequest):
    try:
        # Retrieve
        query_embedding = embed_model.encode(request.query, normalize_embeddings=True).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=request.n_results,
            where=request.filters if request.filters else None,
            include=["documents", "metadatas", "distances"]
        )

        # Check results
        if not results or not results["documents"] or not results["documents"][0]:
            return {
                "retrieved_results": {"documents": [[]], "metadatas": [[]], "distances": [[]]},
                "llm_response": "No relevant EEG segments found.",
                "validation": {"confidence": "N/A", "corrected": False}
            }

        # Run Hybrid RAG Pipeline
        final_answer, validation_info = hybrid_rag_pipeline(
            request.query,
            results["documents"][0]
        )

        return {
            "retrieved_results": results,
            "llm_response": final_answer,
            "validation": {
                "confidence": validation_info["confidence"],
                "corrected": validation_info["corrected"]
            }
        }

    except Exception as e:
        return {"error": f"API error: {str(e)}"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "mode": "hybrid_rag"}