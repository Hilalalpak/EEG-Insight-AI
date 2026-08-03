# api/main.py - Fixed and Optimized Hybrid RAG for Gemma 2b
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


def call_llm(prompt: str, temperature: float = 0.1, max_tokens: int = 512) -> str:
    """Optimized LLM call"""
    payload = {
        "model": "gemma:2b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "num_predict": max_tokens,
            "num_ctx": 4096
        }
    }

    try:
        response = requests.post(
            "http://eegi-ollama:11434/api/generate",
            json=payload,
            timeout=90
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"Error: {e}"


def hybrid_rag_pipeline(query: str, eeg_docs: list[str], medical_docs: list[str]) -> tuple[str, dict]:
    """Optimized pipeline with better context handling"""

    # Build focused context
    medical_info = "\n\n".join(medical_docs[:2]) if medical_docs else "No medical definitions available."
    eeg_info = "\n\n".join(eeg_docs[:3]) if eeg_docs else "No EEG data available."

    # ============ TWO-STEP: Summarize then Answer ============

    # STEP 1: Extract key facts (fast)
    summary_prompt = f"""Extract key facts from this data:

{eeg_info[:1000]}

Question: {query}

Key facts (patient IDs, labels, values):"""

    summary = call_llm(summary_prompt, temperature=0.0, max_tokens=200)

    # STEP 2: Generate answer with medical context
    answer_prompt = f"""Answer using medical knowledge and EEG data.

Medical Background:
{medical_info[:500]}

EEG Facts:
{summary}

Question: {query}

Answer:"""

    answer = call_llm(answer_prompt, temperature=0.2, max_tokens=400)

    # Simple validation
    validation_info = {
        "passed": len(answer) > 30 and "cannot" not in answer.lower(),
        "confidence": "MEDIUM",
        "details": f"Summary length: {len(summary)}, Answer length: {len(answer)}",
        "corrected": False
    }

    return answer, validation_info


@app.post("/query")
def query_eeg_data(request: QueryRequest):
    try:
        query_embedding = embed_model.encode(request.query, normalize_embeddings=True).tolist()

        # Query both collections
        eeg_results = eeg_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(request.n_results, 5),
            where=request.filters if request.filters else None,
            include=["documents", "metadatas", "distances"]
        )

        medical_results = medical_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(request.n_definitions, 3),
            include=["documents"]
        )

        # Check if we got results
        eeg_docs = eeg_results.get("documents", [[]])[0] if eeg_results else []
        medical_docs = medical_results.get("documents", [[]])[0] if medical_results else []

        if not eeg_docs and not medical_docs:
            return {
                "retrieved_eeg_segments": {"documents": [[]], "metadatas": [[]], "distances": [[]]},
                "retrieved_medical_definitions": {"documents": [[]]},
                "llm_response": "No relevant data found in the database.",
                "validation": {"confidence": "N/A", "corrected": False}
            }

        # Generate response
        final_answer, validation_info = hybrid_rag_pipeline(
            request.query,
            eeg_docs=eeg_docs,
            medical_docs=medical_docs
        )

        return {
            "retrieved_eeg_segments": eeg_results,
            "retrieved_medical_definitions": medical_results,
            "llm_response": final_answer,
            "validation": {
                "confidence": validation_info["confidence"],
                "corrected": validation_info["corrected"],
                "details": validation_info["details"]
            }
        }

    except Exception as e:
        return {"error": f"API error: {str(e)}"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "mode": "hybrid_rag_fixed_gemma2b"}


"""(eegi-ai) hilalalpak@Hilal-MacBook-Air EEGInsightAI % python tes.py

================================================================================
QUESTION: What is LPD?
================================================================================
📤 Sending request...

📊 RETRIEVED DATA:
  - EEG segments: 5
  - Medical definitions: 3

📄 FIRST EEG SEGMENT (preview):
  EEG segment from patient 1002379034 at second 18. Classified as LPD with mixed expert opinions (64% agreement from 28 expert votes), indicating an edge case or transitional pattern. Alternative interp...

🏷️  FIRST METADATA:
  {
  "context_window_ids": "2,3",
  "confidence": 0.6428571428571429,
  "has_fast_activity": true,
  "eeg_id": "1002379034",
  "prob_lpd": 0.6428571428571429,
  "total_gpd_votes": 6,
  "is_edge_case": false,
  "has_high_amplitude": true,
  "total_other_votes": 0,
  "num_overlapping_windows": 2,
  "start_second": 18,
  "prob_grda": 0.0,
  "mean_snr": 1.6949153335123828,
  "is_high_confidence": false,
  "total_grda_votes": 0,
  "expert_consensus": "LPD",
  "prob_lrda": 0.0,
  "prob_seizure": 0.14285714285714285,
  "total_seizure_votes": 4,
  "total_votes": 28,
  "mean_power": 165.6687619203674,
  "total_lpd_votes": 18,
  "has_slow_activity": false,
  "prob_other": 0.0,
  "prob_gpd": 0.21428571428571427,
  "has_clean_signal": false,
  "num_context_windows": 2,
  "mean_sef": 30.57894736842105,
  "is_mixed_pattern": false,
  "total_lrda_votes": 0
}

📚 FIRST MEDICAL DEFINITION (preview):
  complete multicenter investigations.
After the establishment of the standardized terminology and
free access to a database incorporating these terms, there have been
many investigations into the clini...

🤖 LLM RESPONSE:
  Sure, here's the answer to your question:

The label for LPD in this context is lateralized periodic discharges (LPDs). LPDs are a type of epilepsy that is characterized by the presence of sharp, high-frequency waves on an EEG.

✅ VALIDATION:
  {
  "confidence": "MEDIUM",
  "corrected": false,
  "details": "Summary length: 307, Answer length: 227"
}

Press Enter to continue...

================================================================================
QUESTION: Tell me about patient 42516
================================================================================
📤 Sending request...

📊 RETRIEVED DATA:
  - EEG segments: 5
  - Medical definitions: 3

📄 FIRST EEG SEGMENT (preview):
  EEG segment from patient 1002197945 at second 41. No expert annotations available for this timepoint, representing unlabeled baseline or artifact-prone activity. Signal analysis reveals: mean power of...

🏷️  FIRST METADATA:
  {
  "total_lrda_votes": 0,
  "is_edge_case": false,
  "mean_snr": 15.14507145299126,
  "prob_lrda": 0.0,
  "prob_gpd": 0.0,
  "confidence": 0.0,
  "has_fast_activity": false,
  "total_lpd_votes": 0,
  "has_high_amplitude": true,
  "total_votes": 0,
  "has_clean_signal": true,
  "total_gpd_votes": 0,
  "has_slow_activity": false,
  "prob_seizure": 0.0,
  "prob_other": 0.0,
  "context_window_ids": "",
  "prob_grda": 0.0,
  "start_second": 41,
  "total_seizure_votes": 0,
  "num_overlapping_windows": 0,
  "total_other_votes": 0,
  "prob_lpd": 0.0,
  "total_grda_votes": 0,
  "expert_consensus": "unknown",
  "mean_power": 3947.119444496378,
  "num_context_windows": 0,
  "eeg_id": "1002197945",
  "is_high_confidence": false,
  "mean_sef": 13.263157894736842,
  "is_mixed_pattern": true
}

📚 FIRST MEDICAL DEFINITION (preview):
  41a. Brief Potentially Ictal Rhythmic Discharges (BIRDs)
(deﬁnite)
41b. Brief Potentially Ictal Rhythmic Discharges (BIRDs)
(deﬁnite)
41c. Brief Potentially Ictal Rhythmic Discharges (BIRDs)
(deﬁnite)...

🤖 LLM RESPONSE:
  I am unable to provide a response to this question as the context does not provide any information about patient 42516.

✅ VALIDATION:
  {
  "confidence": "MEDIUM",
  "corrected": false,
  "details": "Summary length: 125, Answer length: 119"
}

Press Enter to continue...

================================================================================
QUESTION: Show me seizure segments
================================================================================
📤 Sending request...

📊 RETRIEVED DATA:
  - EEG segments: 5
  - Medical definitions: 3

📄 FIRST EEG SEGMENT (preview):
  EEG segment from patient 1001717358 at second 12. Classified as SEIZURE with high expert consensus (100% agreement from 6 expert votes), indicating a clear, well-defined pattern. This assessment is ba...

🏷️  FIRST METADATA:
  {
  "has_slow_activity": true,
  "total_lrda_votes": 0,
  "has_clean_signal": true,
  "num_overlapping_windows": 2,
  "total_votes": 6,
  "has_high_amplitude": false,
  "start_second": 12,
  "total_lpd_votes": 0,
  "total_other_votes": 0,
  "prob_lrda": 0.0,
  "mean_snr": 142.0048098177618,
  "is_mixed_pattern": false,
  "context_window_ids": "1,2",
  "is_edge_case": false,
  "prob_gpd": 0.0,
  "mean_power": 1.7701079886010247e-09,
  "mean_sef": 1.0,
  "has_fast_activity": false,
  "prob_lpd": 0.0,
  "prob_other": 0.0,
  "is_high_confidence": true,
  "total_seizure_votes": 6,
  "num_context_windows": 2,
  "prob_grda": 0.0,
  "expert_consensus": "Seizure",
  "total_grda_votes": 0,
  "prob_seizure": 1.0,
  "confidence": 1.0,
  "eeg_id": "1001717358",
  "total_gpd_votes": 0
}

📚 FIRST MEDICAL DEFINITION (preview):
  an ECSz).
NOTE: For patients with prior known epileptic encephalop-
athy, to qualify as ECSE, the EEG pattern needs to represent
either:
a. an increase in prominence or frequency of epileptiform
disch...

🤖 LLM RESPONSE:
  I am unable to generate seizure segments from the context, as the context does not provide any EEG data.

✅ VALIDATION:
  {
  "confidence": "MEDIUM",
  "corrected": false,
  "details": "Summary length: 307, Answer length: 104"
}

Press Enter to continue..."""