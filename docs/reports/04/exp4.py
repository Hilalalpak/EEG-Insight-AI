# api/main.py - Enhanced Multi-Query & Proactive Error Handling
import re
import requests
import chromadb
import ast
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

print("Loading embedding model...")
embed_model = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
print("Embedding model loaded.")

print("Connecting to ChromaDB...")
chroma_client = chromadb.HttpClient(host="eegi-chroma", port=8000)

eeg_collection = chroma_client.get_or_create_collection(
    name="eeg_insights", metadata={"hnsw:space": "cosine"}
)
medical_collection = chroma_client.get_or_create_collection(
    name="medical_definitions", metadata={"hnsw:space": "cosine"}
)
video_collection = chroma_client.get_or_create_collection(
    name="eeg_medical_video_knowledge", metadata={"hnsw:space": "cosine"}
)
print("Connected to ChromaDB and all collections.")

app = FastAPI(title="EEG RAG API - Advanced Multi-Query")


class QueryRequest(BaseModel):
    query: str
    n_results_per_query: int = 3
    n_definitions: int = 2
    n_videos: int = 2
    filters: dict | None = None


def call_llm(prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
    """Optimized LLM call."""
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
        response = requests.post("http://eegi-ollama:11434/api/generate", json=payload, timeout=90)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"Error: {e}"


# NEW: Smart Sub-Query Generator
def generate_sub_queries(query: str) -> list[str]:
    """Use an LLM to decompose a complex query into simpler, targeted sub-queries for different scenarios."""
    prompt = f"""
    You are an expert in decomposing complex medical questions into simple, effective search queries for a vector database.
    Your task is to break down the user's question into 2-3 distinct, simple sub-queries. The goal is to retrieve separate pieces of information that can be combined to answer the original question.
    The output MUST be ONLY a Python list of strings.

    Handle these cases:
    1.  **Comparison:** For "differentiate A and B", create queries for both "what is A" and "what is B".
    2.  **Attribute Association:** For "what is the [attribute] of [entity]?", create queries for the entity itself and the general definition of the attribute.
    3.  **Abstract Concepts:** For "patterns with [property]", create queries for concrete examples of patterns known to have that property.

    ---
    ### Examples:

    **Original Question:** "Differentiate between LRDA and GRDA"
    **Output:** ["what are the characteristics of LRDA in EEG?", "what are the characteristics of GRDA in EEG?"]

    **Original Question:** "What is the confidence for patient 1002379034's LPD?"
    **Output:** ["EEG data for patient 1002379034 showing LPD", "how is confidence of LPD patterns measured in EEG analysis?", "what is LPD?"]

    **Original Question:** "What patterns have high amplitude and fast activity?"
    **Output:** ["examples of high amplitude EEG patterns", "examples of fast frequency activity in EEG signals", "what are the signal characteristics of a seizure?"]

    ---
    ### Your Task:

    **Original Question:** "{query}"
    **Output:**
    """
    try:
        response_str = call_llm(prompt, max_tokens=200)
        sub_queries = ast.literal_eval(response_str)
        if isinstance(sub_queries, list) and sub_queries:
            sub_queries.append(query)
            return list(set(sub_queries))
        return [query]
    except Exception:
        return [query]


def build_synthesis_prompt(query: str, eeg_docs: list, medical_docs: list, video_docs: list) -> str:
    """Builds the final synthesis prompt for the main LLM."""
    context_parts = []
    if medical_docs:
        context_parts.append(f"### Medical Definitions (Theory):\n" + "\n".join(medical_docs)[:600])
    if video_docs:
        context_parts.append(f"### Expert Video Commentary (Practical Insight):\n" + "\n".join(video_docs)[:500])
    if eeg_docs:
        context_parts.append(f"### Relevant Patient EEG Findings (Case Data):\n" + "\n".join(eeg_docs)[:800])

    instruction = (
        "\n\n### Task:\nAct as a neurology expert and integrate all provided evidence (Theory, Practical Insight, Case Data). "
        "Use this evidence to create a comprehensive, synthesized, and clear answer to the user's original question. "
        "If you cannot find enough information on a topic, state this clearly but still provide any other relevant findings you can."
    )
    return "\n\n".join(context_parts) + instruction + f"\n\n### Original Question:\n{query}\n\n### Answer (Expert Analysis):"


@app.post("/query")
def query_eeg_data(request: QueryRequest):
    try:
        sub_queries = generate_sub_queries(request.query)

        unique_eeg_docs, unique_medical_docs, unique_video_docs = set(), set(), set()

        for sub_q in sub_queries:
            query_embedding = embed_model.encode(sub_q, normalize_embeddings=True).tolist()

            # Search across all collections
            for collection, doc_set, n_results in [
                (eeg_collection, unique_eeg_docs, request.n_results_per_query),
                (medical_collection, unique_medical_docs, request.n_definitions),
                (video_collection, unique_video_docs, request.n_videos)
            ]:
                results = collection.query(query_embeddings=[query_embedding], n_results=n_results,
                                           include=["documents"])
                for doc in results.get("documents", [[]])[0]:
                    doc_set.add(doc)

        # NEW: Reasonable Fallback
        total_docs_found = len(unique_eeg_docs) + len(unique_medical_docs) + len(unique_video_docs)
        if total_docs_found < 2:
            helpful_response = f"Not enough information found in the knowledge base for your query: '{request.query}'. Could you specify which terms you want to explore further?"
            return {"llm_response": helpful_response, "sub_queries_used": sub_queries}

        final_prompt = build_synthesis_prompt(
            query=request.query,
            eeg_docs=list(unique_eeg_docs),
            medical_docs=list(unique_medical_docs),
            video_docs=list(unique_video_docs)
        )

        final_answer = call_llm(final_prompt)

        return {
            "llm_response": final_answer,
            "retrieved_doc_counts": {
                "eeg_segments": len(unique_eeg_docs),
                "medical_definitions": len(unique_medical_docs),
                "video_knowledge": len(unique_video_docs)
            },
            "sub_queries_used": sub_queries
        }

    except Exception as e:
        return {"error": f"API error: {str(e)}"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "mode": "mac_air_optimized"}


"""[
  {
    "success": true,
    "time": 100.22564506530762,
    "answer": "Lateralized periodic discharges (LPDs) are a pattern of rhythmic and periodic electrical activity observed in the electroencephalogram (EEG) of critically ill patients. They are highly associated with seizures and are often seen in patients with acute seizures.",
    "length": 261,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Medical Terms",
    "question": "What is LPD?"
  },
  {
    "success": true,
    "time": 106.55667185783386,
    "answer": "Sure, here's an explanation of GPD and its clinical significance:\n\n**GPD (Generalized Periodic Discharges):**\n\nGeneralized periodic discharges (GPDs) are a type of rhythmic electrical activity that is seen in many neurological disorders, including epilepsy. They are characterized by the appearance of regular, periodic discharges on an EEG, typically in the delta frequency range (0.5-4 Hz).\n\n**Clinical Significance of GPDs:**\n\nGPDs are often seen in patients with epilepsy, particularly in those with focal epilepsy or status epilepticus. They can be a sign of underlying brain dysfunction, and can provide valuable information about the underlying cause of epilepsy.\n\n**Additional Relevant Findings:**\n\nThe EEG segment described in the case data exhibits characteristics of both LPD and GPD. This suggests that the signal is complex and may represent a combination of both LPD and GPD activity. The high amplitude, fast-activity seen in the segment is consistent with muscle artifact, while the presence of a clear periodic discharge pattern is consistent with LPD.\n\n**Conclusion:**\n\nThe EEG findings in this case suggest that the patient may have a complex epilepsy with a combination of LPD and GPD activity. Further investigation and analysis would be necessary to determine the underlying cause of the patient's epilepsy.",
    "length": 1329,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Medical Terms",
    "question": "Explain GPD and its clinical significance"
  },
  {
    "success": true,
    "time": 78.50770497322083,
    "answer": "BIRDs are a type of Ictal Rhythmic Discharges (IRDs) that exhibit a pattern of evolution or \"evolving BIRDs\". They are characterized by the appearance of multiple sharp waves in each burst, with at least one wave in more than half of the burst. This pattern is often seen in patients with epilepsy and can be a sign of interictal-ictal transitions or other conditions that affect brain activity.",
    "length": 395,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Medical Terms",
    "question": "What are BIRDs in EEG terminology?"
  },
  {
    "success": true,
    "time": 77.2771999835968,
    "answer": "The context does not provide any information about LRDA and GRDA, so I cannot answer this question from the provided context.",
    "length": 125,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Medical Terms",
    "question": "Differentiate between LRDA and GRDA"
  },
  {
    "success": true,
    "time": 112.69735431671143,
    "answer": "The Ictal-Interictal Continuum (EEC) refers to the gradual transition between seizures and periods of relative calm or wakefulness. It encompasses various patterns of brain activity that alternate between periods of increased and decreased activity.\n\n**Key characteristics of the EEC include:**\n\n* **Mixed expert opinions:** While some experts may classify a pattern as an Ictal or an Interictal, others may classify it as a different pattern altogether.\n* **Variable presentation:** The EEC can manifest with different patterns in different patients and at different stages of epilepsy.\n* **Clinical significance:** The EEC is often associated with seizures, but it can also be observed in patients without epilepsy.\n* **Multifocal activity:** The EEC can involve activity in multiple brain regions, both on the same side and across the brain.\n\n**Additional relevant findings:**\n\n* The EEG segment from patient 1002142157 exhibits characteristics of the EEC, with mixed expert opinions and a complex pattern.\n* This case highlights the variability of the EEC and the importance of considering multiple perspectives on its interpretation.\n* The EEC is a complex phenomenon that is still not fully understood, but it is an important clinical pattern that can help neurologists diagnose and manage epilepsy.",
    "length": 1305,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Medical Terms",
    "question": "What is the Ictal-Interictal Continuum?"
  },
  {
    "success": true,
    "time": 94.90719699859619,
    "answer": "I am unable to access external sources or provide medical advice, so I cannot answer this question.",
    "length": 99,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Patient Queries",
    "question": "Tell me about patient 1002379034"
  },
  {
    "success": true,
    "time": 92.6227056980133,
    "answer": "The context does not provide any information about patterns in patient 1001717358, so I cannot answer this question from the provided context.",
    "length": 142,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Patient Queries",
    "question": "What patterns are in patient 1001717358?"
  },
  {
    "success": true,
    "time": 92.3686261177063,
    "answer": "The context does not provide any information about seizure events for patient 1001717358, so I cannot generate the requested answer from the provided context.",
    "length": 158,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Patient Queries",
    "question": "Show seizure events for patient 1001717358"
  },
  {
    "success": true,
    "time": 86.80795001983643,
    "answer": "The context does not provide any information about patient 42516, so I cannot generate the requested answer from the provided context.",
    "length": 134,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Patient Queries",
    "question": "Find patient 42516 data"
  },
  {
    "success": true,
    "time": 85.58870506286621,
    "answer": "The confidence for patient 1002379034's LPD is not provided in the context, so I cannot answer this question from the provided context.",
    "length": 135,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Patient Queries",
    "question": "What is the confidence for patient 1002379034's LPD?"
  },
  {
    "success": true,
    "time": 115.96960520744324,
    "answer": "Sure, here's the answer to the user's question:\n\nThe provided evidence suggests that two EEG segments from patients 1002197945 and 1001717358 show high-confidence seizure activity. The segments are classified as seizures by the experts, indicating clear and well-defined patterns.\n\nThe mean power of the segments is 2271.64, indicating very high amplitude activity. The signal-to-noise ratio is 7.93, showing good signal quality. The spectral edge frequency is at 16.63 Hz, representing beta-dominant activity, suggesting arousal or ictal patterns.\n\nThese findings suggest that the seizures in both patients are abnormal synchronized neuronal discharges, indicating a clear and immediate clinical need for attention.",
    "length": 716,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Pattern Analysis",
    "question": "Show high-confidence seizure segments"
  },
  {
    "success": true,
    "time": 105.15242910385132,
    "answer": "Sure, here's a summary of typical seizure signal characteristics:\n\n* **High amplitude activity**\n* **Clean and well-defined pattern**\n* **High signal-to-noise ratio**\n* **Spectral edge frequency at a specific frequency** (usually theta-dominant activity)\n\nThese characteristics are indicative of a seizure and can help neurologists differentiate between different types of seizures.",
    "length": 382,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Pattern Analysis",
    "question": "What are typical seizure signal characteristics?"
  },
  {
    "success": false,
    "time": 120.0243980884552,
    "error": "HTTPConnectionPool(host='localhost', port=8001): Read timed out. (read timeout=120)",
    "category": "Pattern Analysis",
    "question": "Find segments with mixed expert opinions"
  },
  {
    "success": true,
    "time": 111.04568409919739,
    "answer": "EEG segment from patient 1002858110 at second 4 classified as OTHER with high expert consensus (100% agreement from 20 expert votes), indicating a clear, well-defined pattern. Signal analysis reveals: mean power of 1164.24 indicating very high amplitude activity, signal-to-noise ratio of 4.10 showing acceptable quality with some noise, and spectral edge frequency at 21.63 Hz representing beta-dominant activity suggesting arousal or ictal patterns. Keywords: other, high-confidence, high-amplitude, fast-activity.\n\nEEG segment from patient 1001717358 at second 123 classified as SEIZURE with high expert consensus (100% agreement from 3 expert votes), indicating a clear, well-defined pattern. Signal analysis reveals: mean power of 3076.72 indicating very high amplitude activity, signal-to-noise ratio of 4.10 showing acceptable quality with some noise, and spectral edge frequency at 21.63 Hz representing beta-dominant activity suggesting arousal or ictal patterns.",
    "length": 972,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Pattern Analysis",
    "question": "What patterns have high amplitude and fast activity?"
  },
  {
    "success": true,
    "time": 91.59106993675232,
    "answer": "The EEG segments from patients 1003330515 and 1001487592 both exhibit clean signal quality, with mean power values of 998860.59 and 1104.22, respectively. This indicates that the signals are highly amplified and of good quality, meeting the criteria for a clean signal.",
    "length": 269,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Pattern Analysis",
    "question": "Show segments with clean signal quality"
  },
  {
    "success": true,
    "time": 87.44826316833496,
    "answer": "Sure, here's a comparison between LPD and GPD patterns:\n\n**Lateralized Periodic Discharges (LPDs):**\n* Characterized by bilateral, symmetric, and regular patterns of rhythmic electrical activity.\n* Typically observed in patients with focal neurological lesions, such as stroke or brain tumors.\n* Can be associated with seizures, but the exact relationship is not always clear.\n\n**Generalized Periodic Discharges (GPDs):**\n* Characterized by bilateral, symmetric, and regular patterns of rhythmic electrical activity.\n* Typically observed in patients with generalized epilepsy or certain types of focal lesions.\n* Can also be associated with seizures, but the exact relationship is not always clear.\n\nIn this case data, the EEG segment classified as LPD with moderate agreement (67% agreement from 3 expert votes) suggests typical presentation with some variability. However, the clinical context and EEG findings are suggestive of GPD pattern. Therefore, the answer is GPD.",
    "length": 973,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Comparisons",
    "question": "Compare LPD vs GPD patterns"
  },
  {
    "success": false,
    "time": 120.02219223976135,
    "error": "HTTPConnectionPool(host='localhost', port=8001): Read timed out. (read timeout=120)",
    "category": "Comparisons",
    "question": "Compare seizure in EEG 1001717358 vs 1002197945"
  },
  {
    "success": true,
    "time": 92.96778798103333,
    "answer": "Sure, here's the answer to the user's question:\n\nThe difference between high and low confidence classifications in this case is not explicitly mentioned in the context, so I cannot provide a detailed answer to this question from the provided context.",
    "length": 250,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Comparisons",
    "question": "Difference between high and low confidence classifications?"
  },
  {
    "success": true,
    "time": 85.54859709739685,
    "answer": "I am unable to provide a response to the original question as I am unable to access external sources or provide patient-specific information.",
    "length": 141,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Edge Cases",
    "question": "Tell me about patient 999999999"
  },
  {
    "success": true,
    "time": 84.88222002983093,
    "answer": "The context does not provide any information about the XYZ pattern, so I cannot answer this question from the provided context.",
    "length": 127,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Edge Cases",
    "question": "What is XYZ pattern?"
  },
  {
    "success": true,
    "time": 72.65847301483154,
    "answer": "No, LPD and low-power discharge are not the same. While both are periodic discharges, they have different characteristics. LPDs are characterized by high amplitude and fast frequency activity, while low-power discharges are characterized by lower amplitude and slower frequency activity.",
    "length": 287,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Edge Cases",
    "question": "Is LPD the same as low-power discharge?"
  }
]"""