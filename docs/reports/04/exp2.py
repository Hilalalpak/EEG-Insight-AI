# api/main.py - Mac Air 8GB Final Optimized with YouTube Integration
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
# NEW: YouTube video knowledge collection
video_collection = chroma_client.get_or_create_collection(
    name="eeg_medical_video_knowledge",
    metadata={"hnsw:space": "cosine"}
)
print("Connected to ChromaDB and all collections.")

app = FastAPI(title="EEG RAG API")


class QueryRequest(BaseModel):
    query: str
    n_results: int = 5
    n_definitions: int = 3
    filters: dict | None = None


def extract_patient_id(query: str) -> str | None:
    """Extract patient/EEG ID from query"""
    patterns = [
        r'patient\s+(?:id\s+)?(\d+)',
        r'patient_id[:\s]+(\d+)',
        r'\b(\d{10})\b'  # 10-digit number
    ]

    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return match.group(1)
    return None


def extract_eeg_id(query: str) -> str | None:
    """Extract EEG ID from query"""
    patterns = [
        r'eeg\s+(?:id\s+)?(\d+)',
        r'eeg[:\s]+(\d+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return match.group(1)
    return None


def should_use_video_knowledge(query: str) -> bool:
    """
    Videos are ACNS expert interpretations of EEG signals.
    Use them for:
    1. EEG interpretation questions (patterns, characteristics)
    2. Patient data analysis (how to read signals)
    3. Medical terminology

    DON'T use for:
    - Pure metadata queries (patient demographics)
    - Time-based filtering only
    """
    query_lower = query.lower()

    # Always use for interpretation/analysis
    interpretation_keywords = [
        'pattern', 'signal', 'characteristic', 'interpret', 'analyze',
        'lpd', 'gpd', 'lrda', 'grda', 'birds', 'seizure',
        'ictal', 'interictal', 'rhythm', 'frequency', 'amplitude',
        'what is', 'explain', 'show', 'find', 'compare', 'tell me about',
        'clinical significance', 'typical', 'normal', 'abnormal'
    ]

    return any(keyword in query_lower for keyword in interpretation_keywords)


def call_llm(prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
    """Mac Air optimized"""
    payload = {
        "model": "gemma:2b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 40,
            "num_predict": max_tokens,
            "num_ctx": 3072,
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


def hybrid_rag_pipeline(query: str, eeg_docs: list[str], medical_docs: list[str],
                        video_docs: list[str] = None) -> tuple[str, dict]:
    """Context carefully sized for Mac Air - now with optional video knowledge"""

    medical_text = "\n".join(medical_docs[:2])[:600] if medical_docs else ""
    eeg_text = "\n".join(eeg_docs[:3])[:800] if eeg_docs else ""

    # NEW: Include video knowledge if provided (for terminology queries)
    video_text = ""
    if video_docs:
        video_text = "\n".join(video_docs[:2])[:500]  # Keep it concise

    # Build prompt based on available context
    if video_text and not eeg_text:
        # Pure terminology query - use video + medical definitions
        prompt = f"""Medical definitions: {medical_text}

Expert video explanations: {video_text}

Q: {query}

A (explain clearly based on medical knowledge):"""
    elif video_text and eeg_text:
        # Mixed query - combine all sources
        prompt = f"""Medical info: {medical_text}

Video knowledge: {video_text}

Patient data: {eeg_text}

Q: {query}

A (explain term + describe findings):"""
    else:
        # Original patient-focused prompt
        prompt = f"""Medical info: {medical_text}

Patient data: {eeg_text}

Q: {query}

A (explain term + describe findings):"""

    answer = call_llm(prompt, temperature=0.2, max_tokens=512)

    validation_info = {
        "passed": len(answer) > 50,
        "confidence": "HIGH" if len(answer) > 150 else "MEDIUM",
        "details": f"Length: {len(answer)}",
        "corrected": False,
        "used_video_knowledge": bool(video_text)  # NEW: Track if video was used
    }

    return answer, validation_info


@app.post("/query")
def query_eeg_data(request: QueryRequest):
    try:
        # Extract IDs (patient treated as EEG ID)
        patient_id = extract_patient_id(request.query)
        eeg_id = extract_eeg_id(request.query)
        target_eeg_id = eeg_id or patient_id

        # NEW: Check if this is a medical terminology query
        is_terminology_query = should_use_video_knowledge(request.query)

        where_filter = request.filters if request.filters else None

        # Try metadata filter first
        if target_eeg_id and not where_filter:
            where_filter = {"eeg_id": {"$eq": target_eeg_id}}

        query_embedding = embed_model.encode(request.query, normalize_embeddings=True).tolist()

        # Query EEG collection (reduced for terminology queries)
        n_eeg_results = 2 if is_terminology_query else min(request.n_results, 5)
        eeg_results = eeg_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_eeg_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Query medical definitions
        medical_results = medical_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(request.n_definitions, 2),
            include=["documents"]
        )

        # NEW: Query video knowledge ONLY for terminology queries
        video_results = None
        if is_terminology_query:
            video_results = video_collection.query(
                query_embeddings=[query_embedding],
                n_results=2,  # Keep it minimal to preserve performance
                include=["documents", "metadatas"]
            )

        eeg_docs = eeg_results.get("documents", [[]])[0] if eeg_results else []
        medical_docs = medical_results.get("documents", [[]])[0] if medical_results else []
        video_docs = video_results.get("documents", [[]])[0] if video_results else []

        # FALLBACK: If specific ID requested but filter found nothing, try semantic search
        if target_eeg_id and not eeg_docs:
            semantic_query = f"patient {target_eeg_id} EEG recording"
            semantic_embedding = embed_model.encode(semantic_query, normalize_embeddings=True).tolist()

            eeg_results = eeg_collection.query(
                query_embeddings=[semantic_embedding],
                n_results=5,
                include=["documents", "metadatas", "distances"]
            )

            eeg_docs = eeg_results.get("documents", [[]])[0] if eeg_results else []

            if not eeg_docs:
                return {
                    "retrieved_eeg_segments": {"documents": [[]], "metadatas": [[]], "distances": [[]]},
                    "retrieved_medical_definitions": medical_results,
                    "retrieved_video_knowledge": video_results if video_results else {"documents": [[]]},
                    "llm_response": f"No data found for EEG/Patient {target_eeg_id}.",
                    "validation": {"confidence": "N/A", "corrected": False, "used_video_knowledge": False}
                }

        if not eeg_docs and not medical_docs and not video_docs:
            return {
                "retrieved_eeg_segments": {"documents": [[]], "metadatas": [[]], "distances": [[]]},
                "retrieved_medical_definitions": {"documents": [[]]},
                "retrieved_video_knowledge": {"documents": [[]]},
                "llm_response": "No relevant data found.",
                "validation": {"confidence": "N/A", "corrected": False, "used_video_knowledge": False}
            }

        # NEW: Pass video docs to pipeline
        final_answer, validation_info = hybrid_rag_pipeline(
            request.query,
            eeg_docs=eeg_docs,
            medical_docs=medical_docs,
            video_docs=video_docs if is_terminology_query else None
        )

        return {
            "retrieved_eeg_segments": eeg_results,
            "retrieved_medical_definitions": medical_results,
            "retrieved_video_knowledge": video_results if video_results else {"documents": [[]]},
            "llm_response": final_answer,
            "validation": {
                "confidence": validation_info["confidence"],
                "corrected": validation_info["corrected"],
                "used_video_knowledge": validation_info.get("used_video_knowledge", False)
            },
            "queried_eeg_id": target_eeg_id,
            "query_type": "terminology" if is_terminology_query else "patient_specific"
        }

    except Exception as e:
        return {"error": f"API error: {str(e)}"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "mode": "mac_air_optimized",
        "collections": ["eeg_insights", "medical_definitions", "eeg_medical_video_knowledge"]
    }



"""[
  {
    "success": true,
    "time": 41.264678955078125,
    "answer": "Sure, here's the answer to your question:\n\n**Answer:** Lateralized periodic discharges (LPDs) are a pattern of rhythmic and periodic electrical activity that is commonly seen in critically ill patients. The pattern is characterized by lateralized activity on an EEG, with the activity being more prominent in the central and parietotemporal lobes. LPDs are often seen in patients with epilepsy, but they can also be seen in other conditions such as stroke, brain tumors, and metabolic disorders.",
    "length": 495,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Medical Terms",
    "question": "What is LPD?"
  },
  {
    "success": true,
    "time": 34.05901002883911,
    "answer": "Sure, here's the explanation of GPD and its clinical significance:\n\n**GPD (Generalized Periodic Discharges)** is a pattern of rhythmic and periodic discharges on an EEG that is characterized by its high amplitude, fast frequency, and clear spectral characteristics.\n\n**Clinical Significance:**\n\n* GPD is a clinically significant pattern because it can indicate a variety of conditions, including epilepsy, intracranial hemorrhage, metabolic disorders, and drug abuse.\n* It is often seen in patients with focal seizures, especially in patients with epilepsy.\n* GPD can also be seen in patients with intracranial hemorrhage, metabolic disorders, and drug abuse.\n* The clinical significance of GPD makes it an important pattern to recognize on an EEG.\n\n**Additional Notes:**\n\n* GPD is typically seen as a series of sharp, regular discharges on an EEG.\n* The spectral characteristics of GPD are typically fast, high-amplitude, and have a clear edge frequency.\n* GPD is a highly diagnostic pattern, but it can also be seen in patients with other conditions.",
    "length": 1052,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Medical Terms",
    "question": "Explain GPD and its clinical significance"
  },
  {
    "success": true,
    "time": 32.30974078178406,
    "answer": "The context does not provide any information about what BIRDs are, so I cannot answer this question from the provided context.",
    "length": 126,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "MEDIUM",
    "category": "Medical Terms",
    "question": "What are BIRDs in EEG terminology?"
  },
  {
    "success": true,
    "time": 22.781142711639404,
    "answer": "The passage does not provide information about the difference between LRDA and GRDA, so I cannot answer this question from the provided context.",
    "length": 144,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "MEDIUM",
    "category": "Medical Terms",
    "question": "Differentiate between LRDA and GRDA"
  },
  {
    "success": true,
    "time": 22.308406829833984,
    "answer": "The context does not provide any information about the Ictal-Interictal Continuum, so I cannot answer this question from the provided context.",
    "length": 142,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "MEDIUM",
    "category": "Medical Terms",
    "question": "What is the Ictal-Interictal Continuum?"
  },
  {
    "success": true,
    "time": 29.25277805328369,
    "answer": "Patient 1002379034 is a critically ill individual whose EEG segment at second 37 shows very high amplitude activity with a signal-to-noise ratio of 11.12 and a spectral edge frequency at 10.47 Hz, indicating alpha-range activity consistent with relaxed wakefulness or posterior rhythms.",
    "length": 286,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Patient Queries",
    "question": "Tell me about patient 1002379034"
  },
  {
    "success": true,
    "time": 37.99019503593445,
    "answer": "The EEG segment from patient 1001717358 at second 173 contains high-amplitude activity with a mean power of 1370.15 and a signal-to-noise ratio of 10.57, indicating excellent signal quality. The spectral edge frequency is at 11.79 Hz, representing alpha-range activity consistent with relaxed wakefulness or posterior rhythms.\n\nThe EEG segment from patient 1001717358 at second 119 contains high-amplitude activity with a mean power of 1882.66 and a signal-to-noise ratio of 10.57, indicating excellent signal quality. The spectral edge frequency is at 11.79 Hz, representing alpha-range activity consistent with relaxed wakefulness or posterior rhythms.",
    "length": 654,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Patient Queries",
    "question": "What patterns are in patient 1001717358?"
  },
  {
    "success": true,
    "time": 27.507410287857056,
    "answer": "The context does not provide any information about seizure events for patient 1001717358, so I cannot generate the requested information from the context.",
    "length": 154,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Patient Queries",
    "question": "Show seizure events for patient 1001717358"
  },
  {
    "success": true,
    "time": 26.89800190925598,
    "answer": "The patient data is not provided in the context, so I cannot answer this question from the provided context.",
    "length": 108,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "MEDIUM",
    "category": "Patient Queries",
    "question": "Find patient 42516 data"
  },
  {
    "success": true,
    "time": 26.964579820632935,
    "answer": "The confidence for patient 1002379034's LPD is not provided in the context, so I cannot answer this question from the provided context.",
    "length": 135,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "MEDIUM",
    "category": "Patient Queries",
    "question": "What is the confidence for patient 1002379034's LPD?"
  },
  {
    "success": true,
    "time": 45.516010999679565,
    "answer": "Sure, here are the high-confidence seizure segments from patient 1002197945 at second 20 and 5:\n\n**Patient 1002197945 - Second 20:**\n\n* Mean power of 2559.95 indicating very high amplitude activity\n* Signal-to-noise ratio of 6.47 showing good signal quality\n* Spectral edge frequency at 16.37 Hz representing beta-dominant activity suggesting arousal or ictal patterns\n\n**Patient 1002197945 - Second 5:**\n\n* Mean power of 2559.95 indicating very high amplitude activity\n* Signal-to-noise ratio of 6.47 showing good signal quality\n* Spectral edge frequency at 16.37 Hz representing beta-dominant activity suggesting arousal or ictal patterns",
    "length": 640,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Pattern Analysis",
    "question": "Show high-confidence seizure segments"
  },
  {
    "success": true,
    "time": 31.257874965667725,
    "answer": "Typical seizure signal characteristics include:\n\n- Mean power of 4681.43 indicating very high amplitude activity\n- Signal-to-noise ratio of 26.28 showing excellent signal quality with minimal artifact\n- Spectral edge frequency at 5.21 Hz representing theta-dominant activity seen in drowsiness or temporal lobe pathology",
    "length": 320,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Pattern Analysis",
    "question": "What are typical seizure signal characteristics?"
  },
  {
    "success": true,
    "time": 30.60086464881897,
    "answer": "Sure, here's the answer to your question:\n\nA segment from patient 1002142157 at second 46 was classified as GPD with mixed expert opinions (53% agreement from 15 expert votes), indicating an edge case or transitional pattern. Alternative interpretation as OTHER (20%) suggests this segment exhibits characteristics bridging both patterns, which is clinically significant for understanding seizure evolution or interictal-ictal transitions.",
    "length": 439,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Pattern Analysis",
    "question": "Find segments with mixed expert opinions"
  },
  {
    "success": true,
    "time": 60.308504819869995,
    "answer": "Error: HTTPConnectionPool(host='eegi-ollama', port=11434): Read timed out. (read timeout=60)",
    "length": 92,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "MEDIUM",
    "category": "Pattern Analysis",
    "question": "What patterns have high amplitude and fast activity?"
  },
  {
    "success": true,
    "time": 30.785118103027344,
    "answer": "Sure, here are the segments with clean signal quality:\n\n- Patient 1001487592 at second 35\n- Patient 1002197945 at second 20",
    "length": 123,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "MEDIUM",
    "category": "Pattern Analysis",
    "question": "Show segments with clean signal quality"
  },
  {
    "success": true,
    "time": 43.503068923950195,
    "answer": "Sure, here's the comparison between LPD and GPD patterns:\n\n**Lateralized Periodic Discharges (LPDs):**\n\n* Characterized by bilateral, symmetric, and regular patterns of high-amplitude, fast-frequency activity.\n* Typically seen in patients with focal seizures, especially in association with focal lesions in the temporal lobe.\n* Can also be seen in patients with other conditions such as intracranial hemorrhage, meningitis, and metabolic disorders.\n\n**Generalized Periodic Discharges (GPDs):**\n\n* Characterized by bilateral, symmetric, and regular patterns of high-amplitude, fast-frequency activity.\n* Typically seen in patients with generalized tonic-clonic seizures, especially in association with lesions in the temporal lobe.\n* Can also be seen in patients with other conditions such as intracranial hemorrhage, meningitis, and metabolic disorders.\n\nIn this case, the EEG segment shows characteristics of both LPDs and GPDs, suggesting a possible bridging pattern between the two patterns. This is clinically significant because it indicates that the segment exhibits characteristics of both focal and generalized seizures, which can be important for understanding the underlying cause of the seizures.",
    "length": 1208,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Comparisons",
    "question": "Compare LPD vs GPD patterns"
  },
  {
    "success": true,
    "time": 41.64867830276489,
    "answer": "Sure. Comparing seizure in EEG 1001717358 vs 1002197945:\n\n**EEG 1001717358:**\n\n* Classified as SEIZURE with high expert consensus (100% agreement from 6 expert votes).\n* Mean power of 2727.99 indicating very high amplitude activity.\n* Signal-to-noise ratio of 14.85 showing excellent signal quality with minimal artifact.\n* Spectral edge frequency at 10.11 Hz representing alpha-range activity consistent with relaxed wakefulness or posterior rhythms.\n\n**EEG 1002197945:**\n\n* Not provided in the context, so I cannot compare it to the first case.",
    "length": 546,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Comparisons",
    "question": "Compare seizure in EEG 1001717358 vs 1002197945"
  },
  {
    "success": true,
    "time": 57.268107652664185,
    "answer": "Sure, here's the difference between high and low confidence classifications:\n\n- **High-confidence classification (92% agreement from 12 expert votes):**\n  - The EEG segment from patient 1001487592 at second 0 was classified as GPD with high expert consensus (92% agreement from 12 expert votes). This indicates that the pattern is clear, well-defined, and consistent with the expected clinical presentation of generalized tonic-clonic seizures (GPDs).\n  - The signal analysis reveals specific characteristics that are consistent with GPDs, including high mean power, good signal-to-noise ratio, and spectral edge frequency in the beta-dominant range.\n\n- **Low-confidence classification (92% agreement from 24 expert votes):**\n  - The EEG segment from patient 1001487592 at second 12 was classified as GPD with high expert consensus (92% agreement from 24 expert votes). This indicates that the pattern is also clear and well-defined, but there is some variability in the expert reviews.\n  - The signal analysis also reveals some characteristics that are consistent with GPDs, but there is also some disagreement among the experts. For example, the mean power is slightly lower than in the high-confidence classification, and the spectral edge frequency is closer to the alpha-dominant range.\n\n**In summary, the high-confidence classification indicates a clear and consistent pattern, while the low-confidence classification indicates a more variable pattern that may be due to some factors such as variability in the expert reviews.**",
    "length": 1534,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Comparisons",
    "question": "Difference between high and low confidence classifications?"
  },
  {
    "success": true,
    "time": 27.505083799362183,
    "answer": "The context does not provide any information about patient 999999999, so I cannot answer this question from the provided context.",
    "length": 129,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "MEDIUM",
    "category": "Edge Cases",
    "question": "Tell me about patient 999999999"
  },
  {
    "success": true,
    "time": 24.971265077590942,
    "answer": "The context does not provide any information about a pattern called XYZ, so I cannot answer this question from the provided context.",
    "length": 132,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "MEDIUM",
    "category": "Edge Cases",
    "question": "What is XYZ pattern?"
  },
  {
    "success": true,
    "time": 31.88077211380005,
    "answer": "The passage does not specify whether LPD and low-power discharge are the same, so I cannot answer this question from the provided context.",
    "length": 138,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "MEDIUM",
    "category": "Edge Cases",
    "question": "Is LPD the same as low-power discharge?"
  }
]"""