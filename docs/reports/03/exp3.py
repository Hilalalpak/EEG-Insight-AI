# api/main.py - Fixed with Directive Prompts for Gemma 2b
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
    """Optimized pipeline with directive prompts"""

    # Prepare contexts
    medical_text = "\n".join(medical_docs[:2]) if medical_docs else ""
    eeg_text = "\n".join(eeg_docs[:4]) if eeg_docs else ""

    # ============ SINGLE DIRECTIVE PROMPT ============
    # Gemma 2b responds better to direct instructions with clear structure

    prompt = f"""You must answer using ONLY the data below. Do NOT say "context does not provide" - use what's given.

MEDICAL KNOWLEDGE:
{medical_text[:600]}

PATIENT DATA:
{eeg_text[:1200]}

Question: {query}

Instructions:
1. First explain the medical term/concept
2. Then describe what the data shows
3. Use specific numbers from the data

Answer:"""

    answer = call_llm(prompt, temperature=0.2, max_tokens=450)

    # Validation
    has_content = len(answer) > 50 and "context does not" not in answer.lower()

    validation_info = {
        "passed": has_content,
        "confidence": "HIGH" if has_content else "LOW",
        "details": f"Answer length: {len(answer)}",
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
    return {"status": "healthy", "mode": "hybrid_rag_directive_prompts"}


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
  "mean_sef": 30.57894736842105,
  "is_edge_case": false,
  "num_context_windows": 2,
  "prob_seizure": 0.14285714285714285,
  "is_mixed_pattern": false,
  "prob_gpd": 0.21428571428571427,
  "total_other_votes": 0,
  "prob_lrda": 0.0,
  "total_lpd_votes": 18,
  "has_high_amplitude": true,
  "start_second": 18,
  "context_window_ids": "2,3",
  "has_fast_activity": true,
  "prob_other": 0.0,
  "total_votes": 28,
  "total_seizure_votes": 4,
  "confidence": 0.6428571428571429,
  "mean_power": 165.6687619203674,
  "total_lrda_votes": 0,
  "mean_snr": 1.6949153335123828,
  "expert_consensus": "LPD",
  "total_grda_votes": 0,
  "prob_lpd": 0.6428571428571429,
  "eeg_id": "1002379034",
  "num_overlapping_windows": 2,
  "total_gpd_votes": 6,
  "is_high_confidence": false,
  "has_slow_activity": false,
  "has_clean_signal": false,
  "prob_grda": 0.0
}

📚 FIRST MEDICAL DEFINITION (preview):
  complete multicenter investigations.
After the establishment of the standardized terminology and
free access to a database incorporating these terms, there have been
many investigations into the clini...

🤖 LLM RESPONSE:
  **Explanation:**

The term LPD stands for lateralized periodic discharge. It is a pattern of high-amplitude, fast-frequency activity that is typically seen in patients with epilepsy. The data shows that the segment from patient 1002379034 at second 18 is classified as LPD with mixed expert opinions (64% agreement from 28 expert votes), indicating an edge case or transitional pattern.

**Data:**

* Mean power of 165.67
* Signal-to-noise ratio of 1.69
* Spectral edge frequency at 30.58 Hz

✅ VALIDATION:
  {
  "confidence": "HIGH",
  "corrected": false,
  "details": "Answer length: 491"
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
  "total_seizure_votes": 0,
  "prob_lrda": 0.0,
  "is_edge_case": false,
  "start_second": 41,
  "total_votes": 0,
  "has_fast_activity": false,
  "is_mixed_pattern": true,
  "num_context_windows": 0,
  "mean_sef": 13.263157894736842,
  "has_clean_signal": true,
  "context_window_ids": "",
  "prob_grda": 0.0,
  "has_high_amplitude": true,
  "total_lpd_votes": 0,
  "is_high_confidence": false,
  "prob_other": 0.0,
  "total_other_votes": 0,
  "prob_lpd": 0.0,
  "eeg_id": "1002197945",
  "expert_consensus": "unknown",
  "total_gpd_votes": 0,
  "mean_power": 3947.119444496378,
  "has_slow_activity": false,
  "prob_gpd": 0.0,
  "total_grda_votes": 0,
  "confidence": 0.0,
  "mean_snr": 15.14507145299126,
  "num_overlapping_windows": 0,
  "prob_seizure": 0.0,
  "total_lrda_votes": 0
}

📚 FIRST MEDICAL DEFINITION (preview):
  41a. Brief Potentially Ictal Rhythmic Discharges (BIRDs)
(deﬁnite)
41b. Brief Potentially Ictal Rhythmic Discharges (BIRDs)
(deﬁnite)
41c. Brief Potentially Ictal Rhythmic Discharges (BIRDs)
(deﬁnite)...

🤖 LLM RESPONSE:
  **Medical Term/Concept:** The data describes a case of GPD (Generalized Paroxysmal Derebral Disorder) with moderate agreement (75% agreement from 8 expert votes).

**Data shows:**
- Mean power of 3675.65 indicating very high amplitude activity
- Signal-to-noise ratio of 7.10 showing good signal quality
- Spectral edge frequency at 22.05 Hz representing beta-dominant activity suggesting arousal or ictal patterns

✅ VALIDATION:
  {
  "confidence": "HIGH",
  "corrected": false,
  "details": "Answer length: 414"
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
  "total_gpd_votes": 0,
  "total_lpd_votes": 0,
  "prob_seizure": 1.0,
  "start_second": 12,
  "eeg_id": "1001717358",
  "is_edge_case": false,
  "num_context_windows": 2,
  "prob_lpd": 0.0,
  "total_votes": 6,
  "has_slow_activity": true,
  "is_mixed_pattern": false,
  "prob_other": 0.0,
  "confidence": 1.0,
  "expert_consensus": "Seizure",
  "has_high_amplitude": false,
  "context_window_ids": "1,2",
  "total_other_votes": 0,
  "mean_snr": 142.0048098177618,
  "total_lrda_votes": 0,
  "prob_grda": 0.0,
  "total_grda_votes": 0,
  "prob_gpd": 0.0,
  "is_high_confidence": true,
  "num_overlapping_windows": 2,
  "mean_sef": 1.0,
  "prob_lrda": 0.0,
  "mean_power": 1.7701079886010247e-09,
  "has_clean_signal": true,
  "total_seizure_votes": 6,
  "has_fast_activity": false
}

📚 FIRST MEDICAL DEFINITION (preview):
  an ECSz).
NOTE: For patients with prior known epileptic encephalop-
athy, to qualify as ECSE, the EEG pattern needs to represent
either:
a. an increase in prominence or frequency of epileptiform
disch...

🤖 LLM RESPONSE:
  **Medical Term/Concept:** ECSz - Epilepsy with focal onset and sharp wave complexes on EEG

**Description:** The patient's EEG segment at second 12 represents a seizure with high confidence and clean signal quality. The signal analysis revealed low mean power, a high signal-to-noise ratio, and a spectral edge frequency of 1.00 Hz, indicating delta-dominant activity characteristic of deep sleep, encephalopathy, or ictal patterns.

**Specific numbers:**
* Mean power: 0.00
* Signal-to-noise ratio: 142.00
* Spectral edge frequency: 1.00 Hz

✅ VALIDATION:
  {
  "confidence": "HIGH",
  "corrected": false,
  "details": "Answer length: 541"
}

Press Enter to continue..."""