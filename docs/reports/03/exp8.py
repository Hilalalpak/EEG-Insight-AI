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


def hybrid_rag_pipeline(query: str, eeg_docs: list[str], medical_docs: list[str]) -> tuple[str, dict]:
    """Context carefully sized for Mac Air"""

    medical_text = "\n".join(medical_docs[:2])[:600] if medical_docs else ""
    eeg_text = "\n".join(eeg_docs[:3])[:800] if eeg_docs else ""

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
        # Extract IDs (patient treated as EEG ID)
        patient_id = extract_patient_id(request.query)
        eeg_id = extract_eeg_id(request.query)

        target_eeg_id = eeg_id or patient_id

        where_filter = request.filters if request.filters else None

        # Try metadata filter first
        if target_eeg_id and not where_filter:
            where_filter = {"eeg_id": {"$eq": target_eeg_id}}

        query_embedding = embed_model.encode(request.query, normalize_embeddings=True).tolist()

        # Query collections
        eeg_results = eeg_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(request.n_results, 5),
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        medical_results = medical_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(request.n_definitions, 2),
            include=["documents"]
        )

        eeg_docs = eeg_results.get("documents", [[]])[0] if eeg_results else []
        medical_docs = medical_results.get("documents", [[]])[0] if medical_results else []

        # FALLBACK: If specific ID requested but filter found nothing, try semantic search
        if target_eeg_id and not eeg_docs:
            # Text contains "patient {id}", search semantically
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
                    "llm_response": f"No data found for EEG/Patient {target_eeg_id}.",
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
            },
            "queried_eeg_id": target_eeg_id
        }

    except Exception as e:
        return {"error": f"API error: {str(e)}"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "mode": "mac_air_optimized"}




"""[
  {
    "success": true,
    "time": 25.102575063705444,
    "answer": "Lateralized periodic discharges (LPDs) are a pattern of rhythmic and periodic electrical activity in the EEG that is highly associated with acute seizures. The pattern is characterized by lateralized activity, typically involving bilateral temporal electrodes, with a frequency range of 2-10 Hz.",
    "length": 295,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Medical Terms",
    "question": "What is LPD?"
  },
  {
    "success": true,
    "time": 34.956167221069336,
    "answer": "Sure, here's the explanation of GPD and its clinical significance:\n\n**GPD (Generalized Periodic Discharges):**\n\nGeneralized periodic discharges (GPDs) are a type of rhythmic and periodic pattern (RPP) that is characterized by the presence of multiple, closely spaced, regular discharges on an EEG.\n\n**Clinical Significance of GPD:**\n\nGPDs are typically seen in patients with epilepsy, particularly in those with focal epilepsy or status epilepticus. They are often seen as a sign of ongoing seizures or as a transition between seizures. The clinical significance of GPDs is that they can provide valuable information about the underlying cause of the epilepsy.\n\n**Findings in the Patient's EEG:**\n\nThe EEG segment in question shows characteristics of GPD, including multiple, closely spaced, regular discharges on an EEG. The signal is of high amplitude, with a signal-to-noise ratio of 1.58, suggesting a noisy signal. The spectral edge frequency is at 33.84 Hz, which is consistent with muscle artifact or high-frequency seizure components.\n\n**Conclusion:**\n\nThe EEG segment in this patient shows clinical significance as a GPD, indicating the presence of multiple, closely spaced, regular discharges on an EEG. This finding is consistent with the clinical presentation of seizures and provides valuable information about the underlying cause of the epilepsy.",
    "length": 1361,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Medical Terms",
    "question": "Explain GPD and its clinical significance"
  },
  {
    "success": true,
    "time": 27.50188398361206,
    "answer": "The context does not provide any information about what BIRDs are in EEG terminology, so I cannot answer this question from the provided context.",
    "length": 145,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "MEDIUM",
    "category": "Medical Terms",
    "question": "What are BIRDs in EEG terminology?"
  },
  {
    "success": true,
    "time": 21.941715002059937,
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
    "time": 31.71720600128174,
    "answer": "The Ictal-Interictal Continuum refers to the gradual transition between ictal and interictal patterns of epilepsy. The patient's EEG segment exhibits characteristics of both ictal and interictal patterns, suggesting a transitional pattern. The mean power of 740.52, signal-to-noise ratio of 13.54, and spectral edge frequency at 8.63 Hz indicate a complex pattern that partially resembles both ictal and interictal features. This assessment is based on multiple expert review windows, providing multiple temporal perspectives on the same brain activity.",
    "length": 553,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Medical Terms",
    "question": "What is the Ictal-Interictal Continuum?"
  },
  {
    "success": true,
    "time": 29.319815158843994,
    "answer": "Patient 1002379034 is a critically ill individual whose EEG segment at second 37 showed high amplitude activity with a signal-to-noise ratio of 11.12 and a spectral edge frequency of 10.47 Hz, consistent with relaxed wakefulness or posterior rhythms. The mean power was 1687.01, indicating very high amplitude activity.",
    "length": 319,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Patient Queries",
    "question": "Tell me about patient 1002379034"
  },
  {
    "success": true,
    "time": 36.26424598693848,
    "answer": "The EEG segment from patient 1001717358 at second 173 contains high-amplitude activity with a mean power of 1370.15 and a signal-to-noise ratio of 10.57, indicating excellent signal quality. The spectral edge frequency is at 11.79 Hz, which is consistent with alpha-range activity.\n\nThe EEG segment from patient 1001717358 at second 119 contains very high-amplitude activity with a mean power of 1882.66 and a signal-to-noise ratio of 10.57, indicating excellent signal quality. The spectral edge frequency is at 11.79 Hz, which is consistent with alpha-range activity.",
    "length": 569,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Patient Queries",
    "question": "What patterns are in patient 1001717358?"
  },
  {
    "success": true,
    "time": 26.81563377380371,
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
    "time": 27.35454297065735,
    "answer": "The data for patient 42516 is not mentioned in the context, so I cannot answer this question from the provided context.",
    "length": 119,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "MEDIUM",
    "category": "Patient Queries",
    "question": "Find patient 42516 data"
  },
  {
    "success": true,
    "time": 26.258684873580933,
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
    "time": 44.35436201095581,
    "answer": "Sure, here are the high-confidence seizure segments from patient 1002197945 at second 20 and 5:\n\n**Patient 1002197945 - Second 20:**\n- Mean power of 2559.95 indicating very high amplitude activity\n- Signal-to-noise ratio of 6.47 showing good signal quality\n- Spectral edge frequency at 16.37 Hz representing beta-dominant activity suggesting arousal or ictal patterns\n\n**Patient 1002197945 - Second 5:**\n- Mean power of 2559.95 indicating very high amplitude activity\n- Signal-to-noise ratio of 6.47 showing good signal quality\n- Spectral edge frequency at 16.37 Hz representing beta-dominant activity suggesting arousal or ictal patterns",
    "length": 638,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Pattern Analysis",
    "question": "Show high-confidence seizure segments"
  },
  {
    "success": true,
    "time": 32.35248303413391,
    "answer": "Sure, here's a summary of the typical seizure signal characteristics:\n\n- Mean power of 4681.43 indicating very high amplitude activity\n- Signal-to-noise ratio of 26.28 showing excellent signal quality with minimal artifact\n- Spectral edge frequency at 5.21 Hz representing theta-dominant activity seen in drowsiness or temporal lobe pathology",
    "length": 342,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Pattern Analysis",
    "question": "What are typical seizure signal characteristics?"
  },
  {
    "success": true,
    "time": 32.50842595100403,
    "answer": "Sure, here's the answer to your question:\n\nA segment with mixed expert opinions is a segment where multiple experts disagree on the classification of the signal. In this case, patient 1002142157 at second 46 was classified as GPD with mixed expert opinions (53% agreement from 15 expert votes). This indicates an edge case or transitional pattern.",
    "length": 347,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Pattern Analysis",
    "question": "Find segments with mixed expert opinions"
  },
  {
    "success": true,
    "time": 49.992748975753784,
    "answer": "The first EEG segment from patient 1002858110 at second 4 classified as OTHER with high expert consensus (100% agreement from 20 expert votes), indicating a clear, well-defined pattern. Signal analysis reveals: mean power of 1164.24 indicating very high amplitude activity, signal-to-noise ratio of 4.10 showing acceptable quality with some noise, and spectral edge frequency at 21.63 Hz representing beta-dominant activity suggesting arousal or ictal patterns.\n\nThe second EEG segment from patient 1002858110 at second 5 classified as OTHER with high expert consensus (100% agreement from 20 expert votes), indicating a clear, well-defined pattern. Signal analysis reveals: mean power of 1048.72 indicating very high amplitude activity, signal-to-noise ratio of 4.10 showing acceptable quality with some noise, and spectral edge frequency at 21.63 Hz representing beta-dominant activity suggesting arousal or ictal patterns.",
    "length": 925,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Pattern Analysis",
    "question": "What patterns have high amplitude and fast activity?"
  },
  {
    "success": true,
    "time": 28.908663749694824,
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
    "time": 38.80463886260986,
    "answer": "Sure, here's a comparison between LPD and GPD patterns:\n\n**Lateralized Periodic Discharges (LPDs)**\n\n* Characterized by bilateral, symmetric, and regular patterns of high-amplitude, fast-frequency activity.\n* Typically seen in patients with focal seizures, especially those with intracranial lesions in the temporal lobe.\n* Can also be seen in patients with other conditions such as intracranial hemorrhage, meningitis, and metabolic disorders.\n\n**Generalized Periodic Discharges (GPDs)**\n\n* Characterized by bilateral, symmetric, and regular patterns of high-amplitude, fast-frequency activity.\n* Typically seen in patients with non-focal seizures, such as carotid artery stenosis, atrial fibrillation, and ventricular tachycardia.\n* Can also be seen in patients with intracranial lesions in conditions such as subdural hemorrhage and intracerebral hemorrhage.\n\nIn this case, the EEG segment shows characteristics of both LPDs and GPDs, suggesting a clinical significance for understanding seizure evolution or interictal-ictal transitions.",
    "length": 1041,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Comparisons",
    "question": "Compare LPD vs GPD patterns"
  },
  {
    "success": true,
    "time": 51.8949499130249,
    "answer": "Sure, here's the comparison between seizure in EEG 1001717358 and 1002197945:\n\n**EEG segment from patient 1001717358 at second 104:**\n\n* Classified as SEIZURE with high expert consensus (100% agreement from 6 expert votes).\n* Mean power of 2727.99 indicating very high amplitude activity.\n* Signal-to-noise ratio of 14.85 showing excellent signal quality with minimal artifact.\n* Spectral edge frequency at 10.11 Hz representing alpha-range activity consistent with relaxed wakefulness or posterior rhythms.\n\n**EEG segment from patient 1002197945 at second 104:**\n\n* Not provided in the context.\n\n**Therefore, the seizure in EEG 1001717358 is classified as SEIZURE with high expert consensus, indicating a clear, well-defined pattern.**",
    "length": 736,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Comparisons",
    "question": "Compare seizure in EEG 1001717358 vs 1002197945"
  },
  {
    "success": true,
    "time": 46.52214789390564,
    "answer": "Sure, here's the difference between high and low confidence classifications:\n\n**High-confidence classification:**\n\n* Expert consensus: 92% agreement from 12 expert votes.\n* Signal characteristics: mean power of 1602.44, signal-to-noise ratio of 7.02, and spectral edge frequency at 18.42 Hz representing beta-dominant activity suggesting arousal or ictal patterns.\n\n**Low-confidence classification:**\n\n* Expert consensus: 92% agreement from 24 expert votes.\n* Signal characteristics: mean power of 1602.44, signal-to-noise ratio of 7.02, and spectral edge frequency at 18.42 Hz representing beta-dominant activity suggesting arousal or ictal patterns.\n\n**Conclusion:**\n\nThe EEG segment from patient 1001487592 at second 0 was classified as GPD with high expert consensus, indicating a clear, well-defined pattern. The segment from second 12 was classified as GPD with lower expert consensus (92% agreement from 24 expert votes), suggesting a less clear or less well-defined pattern.",
    "length": 982,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Comparisons",
    "question": "Difference between high and low confidence classifications?"
  },
  {
    "success": true,
    "time": 29.49790120124817,
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
    "time": 25.311750888824463,
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
    "time": 28.062920093536377,
    "answer": "No, LPD and low-power discharge are not the same. \n\nThe passage describes LPD as lateralized periodic discharges (LPDs) with moderate agreement (67% agreement from 6 expert votes) suggesting typical presentation with some variability. \n\nOn the other hand, the passage describes low-power discharge as a segment exhibiting characteristics bridging both patterns, which is clinically significant for understanding seizure evolution or interictal-ictal transitions.",
    "length": 462,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "HIGH",
    "category": "Edge Cases",
    "question": "Is LPD the same as low-power discharge?"
  }
]"""