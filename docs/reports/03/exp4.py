# api/main.py - Final Optimized for Gemma 2b
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
            "repeat_penalty": 1.15,
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
    """Optimized pipeline with strict instructions"""

    # Prepare contexts
    medical_text = "\n\n".join(medical_docs[:2]) if medical_docs else ""
    eeg_text = "\n\n".join(eeg_docs[:4]) if eeg_docs else ""

    # ============ STRICT DIRECTIVE PROMPT ============
    prompt = f"""You are an EEG analyst. Follow these rules strictly:

RULES:
1. Use ONLY the exact information from the data below
2. Quote exact patient IDs, numbers, and labels from the data
3. If medical terms appear in MEDICAL KNOWLEDGE, explain them first
4. Then describe what PATIENT DATA shows with specific values
5. DO NOT invent terms or expand abbreviations unless they appear in the data

MEDICAL KNOWLEDGE:
{medical_text[:600]}

PATIENT DATA:
{eeg_text[:1200]}

Question: {query}

Structure your answer:
- Medical Context: [explain terms from medical knowledge]
- Data Findings: [describe what patient data shows with exact numbers]

Answer:"""

    answer = call_llm(prompt, temperature=0.15, max_tokens=450)

    # Enhanced validation
    has_content = len(answer) > 50
    has_refusal = any(
        phrase in answer.lower() for phrase in ["context does not", "unable to", "cannot answer", "no information"])

    validation_info = {
        "passed": has_content and not has_refusal,
        "confidence": "HIGH" if (has_content and not has_refusal) else "MEDIUM" if has_content else "LOW",
        "details": f"Length: {len(answer)}, Refusal detected: {has_refusal}",
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

        # Extract documents
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
    return {"status": "healthy", "mode": "hybrid_rag_final_optimized"}


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
  "prob_other": 0.0,
  "total_seizure_votes": 4,
  "prob_lrda": 0.0,
  "mean_sef": 30.57894736842105,
  "total_votes": 28,
  "has_slow_activity": false,
  "has_high_amplitude": true,
  "is_edge_case": false,
  "mean_power": 165.6687619203674,
  "mean_snr": 1.6949153335123828,
  "is_high_confidence": false,
  "num_overlapping_windows": 2,
  "has_fast_activity": true,
  "total_grda_votes": 0,
  "confidence": 0.6428571428571429,
  "start_second": 18,
  "prob_lpd": 0.6428571428571429,
  "total_gpd_votes": 6,
  "prob_grda": 0.0,
  "total_lpd_votes": 18,
  "expert_consensus": "LPD",
  "eeg_id": "1002379034",
  "has_clean_signal": false,
  "is_mixed_pattern": false,
  "prob_seizure": 0.14285714285714285,
  "total_lrda_votes": 0,
  "total_other_votes": 0,
  "context_window_ids": "2,3",
  "prob_gpd": 0.21428571428571427,
  "num_context_windows": 2
}

📚 FIRST MEDICAL DEFINITION (preview):
  complete multicenter investigations.
After the establishment of the standardized terminology and
free access to a database incorporating these terms, there have been
many investigations into the clini...

🤖 LLM RESPONSE:
  **Medical Context:** Lateralized periodic discharges (LPDs) are a clinical presentation characterized by rhythmic, high-amplitude electrical activity in specific brain regions. They are often seen in critically ill patients and can be associated with seizures.

**Data Findings:**
- Mean power: 165.67
- Signal-to-noise ratio: 1.69
- Spectral edge frequency: 30.58 Hz

✅ VALIDATION:
  {
  "confidence": "HIGH",
  "corrected": false,
  "details": "Length: 367, Refusal detected: False"
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
  "total_gpd_votes": 0,
  "mean_snr": 15.14507145299126,
  "has_slow_activity": false,
  "total_seizure_votes": 0,
  "total_lrda_votes": 0,
  "has_fast_activity": false,
  "is_mixed_pattern": true,
  "eeg_id": "1002197945",
  "is_high_confidence": false,
  "total_other_votes": 0,
  "prob_lpd": 0.0,
  "total_lpd_votes": 0,
  "prob_grda": 0.0,
  "mean_sef": 13.263157894736842,
  "context_window_ids": "",
  "num_context_windows": 0,
  "is_edge_case": false,
  "start_second": 41,
  "prob_other": 0.0,
  "expert_consensus": "unknown",
  "prob_seizure": 0.0,
  "prob_lrda": 0.0,
  "num_overlapping_windows": 0,
  "has_clean_signal": true,
  "mean_power": 3947.119444496378,
  "prob_gpd": 0.0,
  "total_votes": 0,
  "confidence": 0.0,
  "total_grda_votes": 0,
  "has_high_amplitude": true
}

📚 FIRST MEDICAL DEFINITION (preview):
  41a. Brief Potentially Ictal Rhythmic Discharges (BIRDs)
(deﬁnite)
41b. Brief Potentially Ictal Rhythmic Discharges (BIRDs)
(deﬁnite)
41c. Brief Potentially Ictal Rhythmic Discharges (BIRDs)
(deﬁnite)...

🤖 LLM RESPONSE:
  **Medical Context:**
The context does not provide any information about patient 42516, so I cannot answer this question from the provided context.

✅ VALIDATION:
  {
  "confidence": "MEDIUM",
  "corrected": false,
  "details": "Length: 146, Refusal detected: True"
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
  "num_overlapping_windows": 2,
  "num_context_windows": 2,
  "has_clean_signal": true,
  "mean_power": 1.7701079886010247e-09,
  "prob_seizure": 1.0,
  "prob_grda": 0.0,
  "has_fast_activity": false,
  "is_mixed_pattern": false,
  "prob_lrda": 0.0,
  "mean_snr": 142.0048098177618,
  "start_second": 12,
  "total_gpd_votes": 0,
  "is_high_confidence": true,
  "prob_other": 0.0,
  "prob_lpd": 0.0,
  "context_window_ids": "1,2",
  "total_lpd_votes": 0,
  "prob_gpd": 0.0,
  "total_other_votes": 0,
  "expert_consensus": "Seizure",
  "total_seizure_votes": 6,
  "is_edge_case": false,
  "confidence": 1.0,
  "total_grda_votes": 0,
  "has_slow_activity": true,
  "eeg_id": "1001717358",
  "total_lrda_votes": 0,
  "has_high_amplitude": false,
  "mean_sef": 1.0,
  "total_votes": 6
}

📚 FIRST MEDICAL DEFINITION (preview):
  an ECSz).
NOTE: For patients with prior known epileptic encephalop-
athy, to qualify as ECSE, the EEG pattern needs to represent
either:
a. an increase in prominence or frequency of epileptiform
disch...

🤖 LLM RESPONSE:
  **Medical Context:**

An ECSz is an epileptic seizure with high confidence and clean signal quality.

**Data Findings:**

* Mean power of 0.00 indicating low amplitude
* Signal-to-noise ratio of 142.00 showing excellent signal quality with minimal artifact
* Spectral edge frequency at 1.00 Hz representing delta-dominant activity characteristic of deep sleep, encephalopathy, or ictal patterns

✅ VALIDATION:
  {
  "confidence": "HIGH",
  "corrected": false,
  "details": "Length: 394, Refusal detected: False"
}

Press Enter to continue..."""
