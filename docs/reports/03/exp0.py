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



"""[
  {
    "question": "What is LPD?",
    "answer": "**General Definition:**\n\nLPD, or low-power delta activity, refers to a type of epilepsy waveform characterized by the presence of very high amplitude, slow-wave activity on EEG recordings. This activity is typically observed in certain temporal lobe regions, such as the frontal lobe and temporal lobes, and is often associated with focal epilepsy and certain types of seizures.\n\n**Analysis of Provided Context:**\n\nThe context provides multiple EEG segments classified as LPD by different expert reviewers. These segments exhibit characteristics such as:\n\n* **High amplitude activity:** EEG segments show significant average power, indicating abnormal brain activity.\n* **Fast-frequency activity:** EEG segments contain components with spectral edge frequencies in the fast range (30-40 Hz), suggesting abnormal electrical activity.\n* **Variability in agreement:** The degree of agreement among expert reviewers varies, indicating some disagreement in interpreting the EEG findings.\n\n**Synthesis and Illustration:**\n\nThe findings suggest that these EEG segments may represent a complex interplay of different seizure types. The high amplitude activity could be related to generalized epileptic activity, while the fast-frequency components may indicate specific types of seizures or interictal activity. The variability in agreement suggests that the exact nature of these seizures may be challenging to determine definitively.\n\n**Conclusion:**\n\nThe EEG segments classified as LPD in the context provide valuable insights into the clinical presentation and complexity of focal epilepsy. The variability in agreement among expert reviewers highlights the importance of considering multiple perspectives on EEG analysis and emphasizes the need for further research to elucidate the exact mechanisms underlying these seizures.",
    "validation": {},
    "status": "success"
  },
  {
    "question": "Tell me about patient 9999999999",
    "answer": "Sure, here is the response to the user's query:\n\nPatient 9999999999 exhibits several characteristics indicative of cognitive arousal or ictal activity.\n\n**High amplitude activity:** EEG segments from patients 1002576868 and 1003330515 show consistently high power values, suggesting significant electrical activity.\n\n**Beta-dominant activity:** The spectral edge frequency at 17.89 Hz in segment from patient 1003330515 is characteristic of beta-dominant activity, which is associated with cognitive arousal and attention.\n\n**Mixed patterns:** The presence of both high amplitude activity and alpha-range activity in segment from patient 1003330515 suggests a mixed pattern, which could indicate a combination of cognitive processes or an underlying cognitive disorder.\n\n**Unlabeled baseline:** The lack of expert annotations for the segment from patient 1003330515 at second 49 indicates that the activity may represent an artifact or an uninterpretable pattern.\n\n**GPD pattern:** The GPD classification with high expert consensus (92% agreement) for patient 1001487592 suggests a clear and well-defined pattern, indicating a cognitive event.\n\nOverall, these findings suggest that patient 9999999999 may be exhibiting cognitive arousal or ictal activity, based on the presence of specific EEG patterns and characteristics. However, without further clinical context or information, it is difficult to make a definitive diagnosis.",
    "validation": {},
    "status": "success"
  },
  {
    "question": "I heard LPD means low-power discharge. Is that correct?",
    "answer": "Sure, here's the explanation you requested:\n\n## LPD means Low-Power Discharge\n\n**Definition:** Based on the context, LPD stands for **Low-Power Discharge**. It's not a specific diagnosis, but a general term used to describe an abnormal pattern in the EEG that indicates a potentially abnormal brain activity.\n\n**Explanation:** The EEG segments you provided show various patterns, but they all share some characteristics:\n\n* **High amplitude:** Several segments show mean power values above 2000, indicating significant electrical activity.\n* **Fast frequency:** Spectral edge frequencies in many segments are in the range of 30-40 Hz, which is characteristic of fast frequency activity associated with muscle artifacts or high-frequency seizure components.\n* **Noisy signal:** Some segments show noisy patterns, suggesting possible artifacts or low-voltage activity.\n\n**Conclusion:** These findings collectively suggest that the EEG segments represent abnormal brain activity with characteristics suggestive of muscle artifacts, high-frequency seizures, or other conditions that cause low-power discharge. Further analysis and interpretation by a neurophysiologist are necessary to determine the specific cause and provide a diagnosis.",
    "validation": {},
    "status": "success"
  },
  {
    "question": "Compare seizure patterns in EEG 1001717358 vs 1002197945",
    "answer": "**General Definition:**\n\nComparing seizure patterns in EEG segments 1001717358 and 1002197945 reveals significant differences in the frequency and amplitude of the electrical activity. \n\n**Analysis of Provided Context:**\n\n* **EEG segment 1001717358:**\n    * High amplitude activity with a high signal-to-noise ratio, indicating clean and well-defined seizure activity.\n    * Spectral edge frequency at 10.11 Hz suggests a posterior rhythm.\n* **EEG segment 1002197945:**\n    * Moderate amplitude activity with a lower signal-to-noise ratio compared to segment 1001717358.\n    * Spectral edge frequency at 11.68 Hz also suggests an alpha-range activity.\n    * Lower amplitude activity and higher signal-to-noise ratio compared to segment 1001717358 indicate a less clear and potentially more ambiguous seizure pattern.\n\n**Synthesis and Illustration:**\n\n* The seizure patterns in the two segments exhibit differences in both the frequency and amplitude of the electrical activity.\n* Segment 1001717358 shows high amplitude activity with a posterior rhythm, indicating a more defined and typical seizure pattern.\n* Segment 1002197945 displays a moderate amplitude activity with an alpha-range edge frequency, suggesting a less well-defined seizure pattern.\n\n**Conclusion:**\n\nThe comparison of these EEG segments highlights the importance of considering both the frequency and amplitude of the seizure patterns when assessing the severity and characteristics of the seizure.",
    "validation": {},
    "status": "success"
  },
  {
    "question": "Show me seizure segments from patient 1001717358",
    "answer": "Sure, here is the response to the user's query:\n\n**General Definition:**\n\nA seizure is an abnormal, sudden spike in brain activity that is characterized by high amplitude and synchronized neuronal discharges. It is a medical emergency that requires immediate attention.\n\n**Analysis of Provided Context:**\n\nThe context provides a series of EEG segments from patient 1001717358 at different time points. Each segment is classified as a seizure with high expert consensus (100% agreement from 6 expert votes).\n\n**Synthesis and Illustration:**\n\nThe EEG segments show the following characteristics:\n\n- **High amplitude activity:** The mean power of the activity in some segments is very high, indicating a significant electrical signal.\n- **Alpha-range activity:** The spectral edge frequency of some segments falls within the alpha range (10-13 Hz), which is associated with relaxed wakefulness or posterior rhythms.\n- **Synchronized neuronal discharges:** The EEG segments show clear, synchronized neuronal discharges, indicating a seizure.\n\nThese findings suggest that the seizures are caused by abnormal synchronized neuronal activity, which is a common feature of seizures.",
    "validation": {},
    "status": "success"
  }
]"""