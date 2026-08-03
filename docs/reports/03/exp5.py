# api/main.py - Enhanced with Smart Patient Filtering
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


def extract_patient_id(query: str) -> str | None:
    """Extract patient ID from query if present"""
    # Look for patterns like "patient 12345" or "patient ID 12345"
    patterns = [
        r'patient\s+(?:id\s+)?(\d+)',
        r'patient_id[:\s]+(\d+)',
        r'\b(\d{5,10})\b'  # 5-10 digit number
    ]

    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return match.group(1)
    return None


def extract_eeg_id(query: str) -> str | None:
    """Extract EEG ID from query if present"""
    patterns = [
        r'eeg\s+(?:id\s+)?(\d+)',
        r'eeg[:\s]+(\d+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return match.group(1)
    return None


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
2. Quote exact patient IDs, EEG IDs, numbers, and labels from the data
3. If medical terms appear in MEDICAL KNOWLEDGE, explain them first
4. Then describe what PATIENT DATA shows with specific values
5. DO NOT invent terms or expand abbreviations unless they appear in the data
6. If data is about different patients than asked, say so clearly

MEDICAL KNOWLEDGE:
{medical_text[:600]}

PATIENT DATA:
{eeg_text[:1200]}

Question: {query}

Structure your answer:
- Medical Context: [explain terms from medical knowledge]
- Data Findings: [describe what patient data shows with exact patient/EEG IDs and numbers]

Answer:"""

    answer = call_llm(prompt, temperature=0.15, max_tokens=450)

    # Enhanced validation
    has_content = len(answer) > 50
    has_refusal = any(phrase in answer.lower() for phrase in ["context does not", "unable to", "cannot answer"])

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
        # Smart filtering: detect if query asks for specific patient or EEG
        patient_id = extract_patient_id(request.query)
        eeg_id = extract_eeg_id(request.query)

        # Build filter if specific ID requested
        where_filter = request.filters if request.filters else None

        if patient_id and not where_filter:
            where_filter = {"patient_id": {"$eq": patient_id}}
        elif eeg_id and not where_filter:
            where_filter = {"eeg_id": {"$eq": eeg_id}}

        query_embedding = embed_model.encode(request.query, normalize_embeddings=True).tolist()

        # Query both collections
        eeg_results = eeg_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(request.n_results, 5),
            where=where_filter,
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

        # Special message if specific patient/EEG requested but not found
        if (patient_id or eeg_id) and not eeg_docs:
            target = f"patient {patient_id}" if patient_id else f"EEG {eeg_id}"
            return {
                "retrieved_eeg_segments": {"documents": [[]], "metadatas": [[]], "distances": [[]]},
                "retrieved_medical_definitions": medical_results,
                "llm_response": f"No data found for {target} in the database. This patient/EEG may not be in the current dataset.",
                "validation": {"confidence": "N/A", "corrected": False, "details": f"Searched for {target}"}
            }

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
            },
            "applied_filter": {"patient_id": patient_id, "eeg_id": eeg_id} if (patient_id or eeg_id) else None
        }

    except Exception as e:
        return {"error": f"API error: {str(e)}"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "mode": "hybrid_rag_smart_filtering"}


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
  "total_grda_votes": 0,
  "eeg_id": "1002379034",
  "mean_power": 165.6687619203674,
  "start_second": 18,
  "total_seizure_votes": 4,
  "is_edge_case": false,
  "prob_grda": 0.0,
  "is_high_confidence": false,
  "is_mixed_pattern": false,
  "prob_seizure": 0.14285714285714285,
  "has_high_amplitude": true,
  "confidence": 0.6428571428571429,
  "mean_sef": 30.57894736842105,
  "mean_snr": 1.6949153335123828,
  "expert_consensus": "LPD",
  "total_gpd_votes": 6,
  "num_context_windows": 2,
  "prob_lrda": 0.0,
  "prob_other": 0.0,
  "total_votes": 28,
  "prob_lpd": 0.6428571428571429,
  "has_slow_activity": false,
  "total_lpd_votes": 18,
  "num_overlapping_windows": 2,
  "total_lrda_votes": 0,
  "has_fast_activity": true,
  "total_other_votes": 0,
  "context_window_ids": "2,3",
  "has_clean_signal": false,
  "prob_gpd": 0.21428571428571427
}

📚 FIRST MEDICAL DEFINITION (preview):
  complete multicenter investigations.
After the establishment of the standardized terminology and
free access to a database incorporating these terms, there have been
many investigations into the clini...

🤖 LLM RESPONSE:
  **Medical Context:** Lateralized periodic discharges (LPDs) are a pattern of rhythmic and periodic activity that is highly associated with acute seizures. 

**Data Findings:**

- Patient 1002379034 has an LPD with mixed expert opinions (64% agreement from 28 expert votes), indicating an edge case or transitional pattern.
- Patient 1003011202 has an LPD with moderate agreement (67% agreement from 3 expert votes), suggesting typical presentation with some variability.

✅ VALIDATION:
  {
  "confidence": "HIGH",
  "corrected": false,
  "details": "Length: 470, Refusal detected: False"
}

Press Enter to continue...

================================================================================
QUESTION: Tell me about patient 42516
================================================================================
📤 Sending request...

📊 RETRIEVED DATA:
  - EEG segments: 0
  - Medical definitions: 3

⚠️  NO EEG SEGMENTS RETRIEVED!

📚 FIRST MEDICAL DEFINITION (preview):
  41a. Brief Potentially Ictal Rhythmic Discharges (BIRDs)
(deﬁnite)
41b. Brief Potentially Ictal Rhythmic Discharges (BIRDs)
(deﬁnite)
41c. Brief Potentially Ictal Rhythmic Discharges (BIRDs)
(deﬁnite)...

🤖 LLM RESPONSE:
  No data found for patient 42516 in the database. This patient/EEG may not be in the current dataset.

✅ VALIDATION:
  {
  "confidence": "N/A",
  "corrected": false,
  "details": "Searched for patient 42516"
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
  "total_seizure_votes": 6,
  "num_overlapping_windows": 2,
  "is_high_confidence": true,
  "num_context_windows": 2,
  "is_mixed_pattern": false,
  "total_gpd_votes": 0,
  "mean_power": 1.7701079886010247e-09,
  "prob_lpd": 0.0,
  "confidence": 1.0,
  "mean_snr": 142.0048098177618,
  "total_grda_votes": 0,
  "expert_consensus": "Seizure",
  "has_clean_signal": true,
  "has_high_amplitude": false,
  "prob_gpd": 0.0,
  "eeg_id": "1001717358",
  "total_votes": 6,
  "start_second": 12,
  "prob_seizure": 1.0,
  "total_lrda_votes": 0,
  "has_slow_activity": true,
  "context_window_ids": "1,2",
  "is_edge_case": false,
  "total_lpd_votes": 0,
  "prob_other": 0.0,
  "mean_sef": 1.0,
  "prob_lrda": 0.0,
  "has_fast_activity": false,
  "prob_grda": 0.0,
  "total_other_votes": 0
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

An ECSz (Electroencephalography) segment from patient 1001717358 at second 12 classified as SEIZURE with high expert consensus (100% agreement from 6 expert votes). This assessment is based on 2 overlapping 50-second expert review windows, providing multiple temporal perspectives on the same brain activity.

**Data Findings:**

* Mean power of 0.00 indicating low amplitude
* Signal-to-noise ratio of 142.00 showing excellent signal quality with minimal artifact
* Spectral edge frequency at 1.00 Hz representing delta-dominant activity characteristic of deep sleep, encephalopathy, or ictal patterns

✅ VALIDATION:
  {
  "confidence": "HIGH",
  "corrected": false,
  "details": "Length: 624, Refusal detected: False"
}

Press Enter to continue...
"""