# api/main.py - Mac Air 8GB Final Optimized
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
    patterns = [
        r'patient\s+(?:id\s+)?(\d+)',
        r'patient_id[:\s]+(\d+)',
        r'\b(\d{5,10})\b'
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


def call_llm(prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
    """Mac Air optimized - ensure full responses"""
    payload = {
        "model": "gemma:2b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 40,
            "num_predict": max_tokens,  # Cevap iÃ§in ayrÄ±lan
            "num_ctx": 3072,  # TOPLAM context (input + output)
            "num_thread": 4
        }
    }

    try:
        response = requests.post(
            "http://eegi-ollama:11434/api/generate",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"Error: {e}"


def hybrid_rag_pipeline(query: str, eeg_docs: list[str], medical_docs: list[str]) -> tuple[str, dict]:
    """Context carefully sized to fit within num_ctx"""

    # CRITICAL: Keep total prompt under ~600 tokens (leaving 500+ for response)
    medical_text = "\n".join(medical_docs[:2])[:600] if medical_docs else ""  # ~150 tokens
    eeg_text = "\n".join(eeg_docs[:3])[:800] if eeg_docs else ""  # ~200 tokens
    # Prompt structure: ~100 tokens
    # Total input: ~450 tokens, leaving 2600 tokens for context window

    prompt = f"""Medical info: {medical_text}

Patient data: {eeg_text}

Q: {query}

A (explain term + describe findings):"""

    answer = call_llm(prompt, temperature=0.2, max_tokens=512)

    validation_info = {
        "passed": len(answer) > 50,
        "confidence": "HIGH" if len(answer) > 150 else "MEDIUM",
        "details": f"Length: {len(answer)}",
        "corrected": False
    }

    return answer, validation_info


@app.post("/query")
def query_eeg_data(request: QueryRequest):
    try:
        patient_id = extract_patient_id(request.query)
        eeg_id = extract_eeg_id(request.query)

        where_filter = request.filters if request.filters else None

        if patient_id and not where_filter:
            where_filter = {"patient_id": {"$eq": patient_id}}
        elif eeg_id and not where_filter:
            where_filter = {"eeg_id": {"$eq": eeg_id}}

        query_embedding = embed_model.encode(request.query, normalize_embeddings=True).tolist()

        # Reduced retrieval counts
        eeg_results = eeg_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(request.n_results, 3),  # Max 3 instead of 5
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        medical_results = medical_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(request.n_definitions, 2),  # Max 2 instead of 3
            include=["documents"]
        )

        eeg_docs = eeg_results.get("documents", [[]])[0] if eeg_results else []
        medical_docs = medical_results.get("documents", [[]])[0] if medical_results else []

        if (patient_id or eeg_id) and not eeg_docs:
            target = f"patient {patient_id}" if patient_id else f"EEG {eeg_id}"
            return {
                "retrieved_eeg_segments": {"documents": [[]], "metadatas": [[]], "distances": [[]]},
                "retrieved_medical_definitions": medical_results,
                "llm_response": f"No data found for {target}.",
                "validation": {"confidence": "N/A", "corrected": False}
            }

        if not eeg_docs and not medical_docs:
            return {
                "retrieved_eeg_segments": {"documents": [[]], "metadatas": [[]], "distances": [[]]},
                "retrieved_medical_definitions": {"documents": [[]]},
                "llm_response": "No relevant data found.",
                "validation": {"confidence": "N/A", "corrected": False}
            }

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
                "corrected": validation_info["corrected"]
            }
        }

    except Exception as e:
        return {"error": f"API error: {str(e)}"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "mode": "mac_air_optimized"}







"""[
  {
    "success": true,
    "time": 32.11974501609802,
    "answer": "Sure, here's the answer to your question:\n\n**Lateralized periodic discharges (LPDs)** are a type of rhythmic and periodic pattern (RPP) that is highly associated with acute seizures in critically ill patients.",
    "length": 209,
    "eeg_count": 3,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Medical Terms",
    "question": "What is LPD?"
  },
  {
    "success": true,
    "time": 37.0915949344635,
    "answer": "Sure, here's the explanation of GPD and its clinical significance:\n\n**GPD (Generalized Periodic Discharges):**\n\nGeneralized periodic discharges (GPDs) are a type of epilepsy characterized by the appearance of regular, periodic discharges on an EEG. They are typically seen in patients with epilepsy and can also be seen in patients with other conditions, such as dementia and metabolic disorders.\n\n**Clinical Significance of GPD:**\n\nGPDs are typically seen in patients with epilepsy and can be a sign of underlying brain abnormalities. They can also be a sign of a metabolic disorder, such as lactic acidosis. The clinical significance of GPDs is that they can provide valuable information about the underlying cause of the epilepsy.\n\nIn this case, the EEG segment shows characteristics of GPD, including regular, periodic discharges that are similar to those seen in GPDs. This suggests that the patient may have epilepsy and that the GPDs are a sign of an underlying brain abnormality.",
    "length": 987,
    "eeg_count": 3,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Medical Terms",
    "question": "Explain GPD and its clinical significance"
  },
  {
    "success": true,
    "time": 35.27853298187256,
    "answer": "The passage does not provide any information about what BIRDs are in EEG terminology, so I cannot answer this question from the provided context.",
    "length": 145,
    "eeg_count": 3,
    "medical_count": 2,
    "confidence": "MEDIUM",
    "category": "Medical Terms",
    "question": "What are BIRDs in EEG terminology?"
  },
  {
    "success": true,
    "time": 24.201051712036133,
    "answer": "The passage does not provide information about the differences between LRDA and GRDA, so I cannot answer this question from the provided context.",
    "length": 145,
    "eeg_count": 3,
    "medical_count": 2,
    "confidence": "MEDIUM",
    "category": "Medical Terms",
    "question": "Differentiate between LRDA and GRDA"
  },
  {
    "success": true,
    "time": 38.72442102432251,
    "answer": "The passage describes an EEG segment classified as GPD with mixed expert opinions, indicating an edge case or transitional pattern. Alternative interpretation as OTHER (20%) suggests this segment exhibits characteristics bridging both patterns, which is clinically significant for understanding seizure evolution or interictal-ictal transitions. This assessment is based on 4 overlapping 50-second expert review windows, providing multiple temporal perspectives on the same brain activity.\n\nThe Ictal-Interictal Continuum refers to the gradual transition between ictal and interictal patterns in epilepsy. The passage suggests that this continuum is present in the patient's EEG segment, with characteristics suggestive of both ictal and interictal patterns. This finding could provide valuable insights into the underlying mechanisms of epilepsy and could be used to develop more effective treatment strategies.",
    "length": 912,
    "eeg_count": 3,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Medical Terms",
    "question": "What is the Ictal-Interictal Continuum?"
  },
  {
    "success": true,
    "time": 0.18802404403686523,
    "answer": "No data found for patient 1002379034.",
    "length": 37,
    "eeg_count": 0,
    "medical_count": 2,
    "confidence": "N/A",
    "category": "Patient Queries",
    "question": "Tell me about patient 1002379034"
  },
  {
    "success": true,
    "time": 0.0472257137298584,
    "answer": "No data found for patient 1001717358.",
    "length": 37,
    "eeg_count": 0,
    "medical_count": 2,
    "confidence": "N/A",
    "category": "Patient Queries",
    "question": "What patterns are in patient 1001717358?"
  },
  {
    "success": true,
    "time": 0.045606136322021484,
    "answer": "No data found for patient 1001717358.",
    "length": 37,
    "eeg_count": 0,
    "medical_count": 2,
    "confidence": "N/A",
    "category": "Patient Queries",
    "question": "Show seizure events for patient 1001717358"
  },
  {
    "success": true,
    "time": 0.04404807090759277,
    "answer": "No data found for patient 42516.",
    "length": 32,
    "eeg_count": 0,
    "medical_count": 2,
    "confidence": "N/A",
    "category": "Patient Queries",
    "question": "Find patient 42516 data"
  },
  {
    "success": true,
    "time": 0.047885894775390625,
    "answer": "No data found for patient 1002379034.",
    "length": 37,
    "eeg_count": 0,
    "medical_count": 2,
    "confidence": "N/A",
    "category": "Patient Queries",
    "question": "What is the confidence for patient 1002379034's LPD?"
  },
  {
    "success": true,
    "time": 50.271422147750854,
    "answer": "Sure, here are the high-confidence seizure segments from patient 1002197945 at second 20 and 5:\n\n**Patient 1002197945 - Second 20:**\n- Mean power of 2559.95 indicating very high amplitude activity\n- Signal-to-noise ratio of 6.47 showing good signal quality\n- Spectral edge frequency at 16.37 Hz representing beta-dominant activity suggesting arousal or ictal patterns\n\n**Patient 1002197945 - Second 5:**\n- Mean power of 2559.95 indicating very high amplitude activity\n- Signal-to-noise ratio of 6.47 showing good signal quality\n- Spectral edge frequency at 16.37 Hz representing beta-dominant activity suggesting arousal or ictal patterns",
    "length": 638,
    "eeg_count": 3,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Pattern Analysis",
    "question": "Show high-confidence seizure segments"
  },
  {
    "success": true,
    "time": 37.59553098678589,
    "answer": "Typical seizure signal characteristics include:\n\n- Mean power of 4681.43 indicating very high amplitude activity\n\n- Signal-to-noise ratio of 26.28 showing excellent signal quality with minimal artifact\n\n- Spectral edge frequency at 5.21 Hz representing theta-dominant activity seen in drowsiness or temporal lobe pathology",
    "length": 322,
    "eeg_count": 3,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Pattern Analysis",
    "question": "What are typical seizure signal characteristics?"
  },
  {
    "success": true,
    "time": 41.32443690299988,
    "answer": "Sure, here's the answer to your question:\n\nThe EEG segment from patient 1002142157 at second 46 is classified as GPD with mixed expert opinions (53% agreement from 15 expert votes). This indicates an edge case or transitional pattern.\n\nThe segment exhibits characteristics that are consistent with both GPD and OTHER interpretations. These characteristics include:\n\n* Mean power of 367.17 indicating very high amplitude activity\n* Signal-to-noise ratio of 7.90 showing good signal quality\n* Spectral edge frequency at 12.16 Hz representing alpha-range activity consistent with relaxed wakefulness or posterior rhythms\n\nThese characteristics suggest that the segment may represent an edge case or transitional pattern.",
    "length": 717,
    "eeg_count": 3,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Pattern Analysis",
    "question": "Find segments with mixed expert opinions"
  },
  {
    "success": true,
    "time": 37.08929109573364,
    "answer": "Sure, the patterns that have high amplitude and fast activity are the two patterns classified as \"OTHER\" in the patient's EEG segment at second 4 and 5.\n\n**Pattern 1 (Second 4):**\n* Mean power: 1164.24\n* Signal-to-noise ratio: 4.10\n* Spectral edge frequency: 21.63 Hz\n\n**Pattern 2 (Second 5):**\n* Mean power: 1048.72\n* Signal-to-noise ratio: 4.10\n* Spectral edge frequency: 21.63 Hz",
    "length": 382,
    "eeg_count": 3,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Pattern Analysis",
    "question": "What patterns have high amplitude and fast activity?"
  },
  {
    "success": true,
    "time": 29.80394983291626,
    "answer": "Sure, here are the segments with clean signal quality:\n\n- Patient 1001487592 at second 35\n- Patient 1002197945 at second 20",
    "length": 123,
    "eeg_count": 3,
    "medical_count": 2,
    "confidence": "MEDIUM",
    "category": "Pattern Analysis",
    "question": "Show segments with clean signal quality"
  },
  {
    "success": true,
    "time": 39.88339304924011,
    "answer": "Sure, here's the comparison between LPD and GPD patterns:\n\n**Lateralized Periodic Discharges (LPDs):**\n\n* Characterized by bilateral, symmetric, and regular patterns of high-amplitude, high-frequency activity.\n* Typically seen in patients with focal seizures, especially focal cortical lesions.\n* Can also be seen in patients with other conditions, such as intracranial hemorrhage or metabolic disorders.\n\n**Generalized Periodic Discharges (GPDs):**\n\n* Characterized by bilateral, symmetric, and regular patterns of high-amplitude, high-frequency activity.\n* Typically seen in patients with generalized epilepsy, especially focal lesions in the temporal lobe.\n* Can also be seen in patients with intracranial hemorrhage or metabolic disorders.\n\nIn this case, the EEG segment exhibits characteristics of both LPDs and GPDs, suggesting a clinical significance for understanding seizure evolution or interictal-ictal transitions.",
    "length": 926,
    "eeg_count": 3,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Comparisons",
    "question": "Compare LPD vs GPD patterns"
  },
  {
    "success": true,
    "time": 0.19860410690307617,
    "answer": "No data found for patient 1001717358.",
    "length": 37,
    "eeg_count": 0,
    "medical_count": 2,
    "confidence": "N/A",
    "category": "Comparisons",
    "question": "Compare seizure in EEG 1001717358 vs 1002197945"
  },
  {
    "success": true,
    "time": 36.81901407241821,
    "answer": "Sure, here's the difference between high and low confidence classifications:\n\n**High-confidence classification:**\n\n* Expert consensus of 92% agreement from 12 expert votes.\n* Clear, well-defined pattern.\n* Mean power of 1602.44 indicating very high amplitude activity.\n* Signal-to-noise ratio of 7.02 showing good signal quality.\n* Spectral edge frequency at 18.42 Hz representing beta-dominant activity suggesting arousal or ictal patterns.\n\n**Low-confidence classification:**\n\n* Expert consensus of 92% agreement from 24 expert votes.\n* Clear, well-defined pattern.\n* Mean power of 1602.44 indicating very high amplitude activity.\n* Signal-to-noise ratio of 7.02 showing good signal quality.\n* No spectral edge frequency at 18.42 Hz.",
    "length": 735,
    "eeg_count": 3,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Comparisons",
    "question": "Difference between high and low confidence classifications?"
  },
  {
    "success": true,
    "time": 0.18201279640197754,
    "answer": "No data found for patient 999999999.",
    "length": 36,
    "eeg_count": 0,
    "medical_count": 2,
    "confidence": "N/A",
    "category": "Edge Cases",
    "question": "Tell me about patient 999999999"
  },
  {
    "success": true,
    "time": 25.639209985733032,
    "answer": "The context does not provide any information about a XYZ pattern, so I cannot answer this question from the provided context.",
    "length": 125,
    "eeg_count": 3,
    "medical_count": 2,
    "confidence": "MEDIUM",
    "category": "Edge Cases",
    "question": "What is XYZ pattern?"
  },
  {
    "success": true,
    "time": 23.545461654663086,
    "answer": "The passage does not specify whether LPD and low-power discharge are the same, so I cannot answer this question from the provided context.",
    "length": 138,
    "eeg_count": 3,
    "medical_count": 2,
    "confidence": "MEDIUM",
    "category": "Edge Cases",
    "question": "Is LPD the same as low-power discharge?"
  }
]"""