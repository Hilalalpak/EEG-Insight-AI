# api/main.py - Enhanced Multi-Query & Video-Prioritized EEG Synthesis
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

app = FastAPI(title="EEG RAG API - Video-Prioritized Multi-Query")


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


def generate_sub_queries(query: str) -> list[str]:
    """Decompose complex queries into simpler sub-queries for EEG, medical, and video sources."""
    prompt = f"""
    You are an expert in decomposing complex medical questions into simple, effective search queries.
    Break down the user's question into 2-3 distinct, targeted sub-queries for retrieving info from EEG data, medical definitions, and video commentary.
    Output MUST be a Python list of strings.

    Examples:
    Question: "Differentiate between LRDA and GRDA"
    Output: ["what are the characteristics of LRDA in EEG?", "what are the characteristics of GRDA in EEG?"]

    Question: "What is the confidence for patient 1002379034's LPD?"
    Output: ["EEG data for patient 1002379034 showing LPD", "how is confidence of LPD patterns measured in EEG analysis?", "what is LPD?"]

    Original Question: "{query}"
    Output:
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
    """Builds a synthesis prompt emphasizing video commentary for EEG interpretation."""
    context_parts = []
    if medical_docs:
        context_parts.append(
            f"### Medical Definitions (Theory):\n" + "\n".join(medical_docs)[:600]
        )
    if video_docs:
        context_parts.append(
            f"### Expert Video Commentary (Practical Insight - PRIORITIZE EEG interpretation examples and terminology from doctors):\n"
            + "\n".join(video_docs)[:500]
        )
    if eeg_docs:
        context_parts.append(
            f"### Relevant Patient EEG Findings (Case Data):\n" + "\n".join(eeg_docs)[:800]
        )

    instruction = (
        "\n\n### Task:\n"
        "Act as a neurology expert and integrate all provided evidence (Theory, Practical Insight, Case Data). "
        "For EEG-specific questions, give PRIORITY to examples, interpretations, and terminology discussed in Expert Video Commentary. "
        "Synthesize all information into a comprehensive, clear, expert-level answer. "
        "Do not copy-paste directly; summarize and integrate insights from all sources. "
        "If insufficient info is found for a part, note it but provide all other relevant findings."
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

        # Fallback if insufficient data
        total_docs_found = len(unique_eeg_docs) + len(unique_medical_docs) + len(unique_video_docs)
        if total_docs_found < 2:
            helpful_response = (
                f"Not enough information found in the knowledge base for your query: '{request.query}'. "
                "Please specify which terms or aspects you want to explore further."
            )
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
    return {"status": "healthy", "mode": "video-prioritized-eeg"}


"""[
  {
    "success": true,
    "time": 108.4331169128418,
    "answer": "Sure, here's the answer to your question:\n\n**Left-sided Periodic Discharges (LPDs)** are a type of epilepsy seizure characterized by the appearance of periodic discharges on an EEG. They are typically unilateral, meaning they appear only in one hemisphere of the brain. LPDs are commonly observed in patients with epilepsy and can provide valuable information about the underlying seizure activity.",
    "length": 398,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Medical Terms",
    "question": "What is LPD?"
  },
  {
    "success": true,
    "time": 111.48184609413147,
    "answer": "Error: HTTPConnectionPool(host='eegi-ollama', port=11434): Read timed out. (read timeout=90)",
    "length": 92,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Medical Terms",
    "question": "Explain GPD and its clinical significance"
  },
  {
    "success": true,
    "time": 83.57542705535889,
    "answer": "BIRDs in EEG terminology refer to brief, potentially Ictol rhythmic discharges. According to the standardized Critical Care EEG Terminology L. J. Hirsch, et al., BIRDs are characterized by negativity graded in four hertz, focal or generalized, and have to be between 0.5 and 10 seconds in duration.",
    "length": 298,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Medical Terms",
    "question": "What are BIRDs in EEG terminology?"
  },
  {
    "success": true,
    "time": 66.66279315948486,
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
    "time": 68.95881080627441,
    "answer": "Sure, here's the answer to the original question:\n\nThe Ictal-Interictal Continuum refers to the gradual transition between electrographic seizures (ESz) and electroclinical status epilepticus (ECSE). This continuum represents the gradual evolution of seizure activity from ESz to ECSE, with the transition being characterized by a gradual increase in amplitude and complexity of the EEG signal.",
    "length": 394,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Medical Terms",
    "question": "What is the Ictal-Interictal Continuum?"
  },
  {
    "success": true,
    "time": 96.19422101974487,
    "answer": "Sure, here's the answer to the question:\n\nPatient 1002379034 is described as having an EEG segment with a mean power of 1687.01 and a signal-to-noise ratio of 14.37, indicating excellent signal quality with minimal artifact. The spectral edge frequency is at 14.11 Hz, suggesting alpha-range activity consistent with relaxed wakefulness or posterior rhythms.\n\nThe absence of expert annotations for this timepoint makes it difficult to provide a comprehensive interpretation. However, the high mean power and clean signal quality suggest a potentially significant finding related to the patient's cognitive state or underlying pathology.\n\nThe case data provides a glimpse into the potential clinical significance of this EEG finding, but further analysis and interpretation require additional context and expert expertise.",
    "length": 821,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Patient Queries",
    "question": "Tell me about patient 1002379034"
  },
  {
    "success": true,
    "time": 82.78945684432983,
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
    "time": 69.32999014854431,
    "answer": "The context does not provide any information about seizure events for patient 1001717358, so I cannot generate the requested information from the context.",
    "length": 154,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Patient Queries",
    "question": "Show seizure events for patient 1001717358"
  },
  {
    "success": true,
    "time": 112.75702786445618,
    "answer": "Error: HTTPConnectionPool(host='eegi-ollama', port=11434): Read timed out. (read timeout=90)",
    "length": 92,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Patient Queries",
    "question": "Find patient 42516 data"
  },
  {
    "success": true,
    "time": 102.17655801773071,
    "answer": "The context does not provide any information about the confidence for patient 1002379034's LPD, so I cannot answer this question from the provided context.",
    "length": 155,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Patient Queries",
    "question": "What is the confidence for patient 1002379034's LPD?"
  },
  {
    "success": true,
    "time": 82.32778716087341,
    "answer": "Sure, here's the answer to the question:\n\n**High-Confidence Seizure Segments:**\n\n* Patient 1001717358: Second 100, classified as seizure with high expert consensus (100% agreement from 3 expert votes).\n* Patient 1002197945: Second 5, classified as seizure with high expert consensus (100% agreement from 3 expert votes).\n\n**Additional Notes:**\n\n* Both patients presented with abnormal synchronized neuronal discharges, indicating seizure activity.\n* Patient 1001717358's seizure had a high mean power, signal-to-noise ratio, and spectral edge frequency, suggesting a clear and well-defined pattern.\n* Patient 1002197945's seizure had a high mean power and a relatively low signal-to-noise ratio, but still met the criteria for a seizure based on the expert's assessment.",
    "length": 770,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Pattern Analysis",
    "question": "Show high-confidence seizure segments"
  },
  {
    "success": true,
    "time": 77.77049326896667,
    "answer": "The passage does not provide information about typical seizure signal characteristics, so I cannot answer this question from the provided context.",
    "length": 146,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Pattern Analysis",
    "question": "What are typical seizure signal characteristics?"
  },
  {
    "success": true,
    "time": 85.20384693145752,
    "answer": "The EEG segment from patient 1002142157 at second 33 classified as GPD with mixed expert opinions (53% agreement from 75 expert votes), indicating an edge case or transitional pattern. Alternative interpretation as OTHER (20%) suggests this segment exhibits characteristics bridging both patterns, which is clinically significant for understanding seizure evolution or interictal-ictal transitions. This assessment is based on 5 overlapping 50-second expert review windows, providing multiple temporal perspectives on the same brain activity.",
    "length": 542,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Pattern Analysis",
    "question": "Find segments with mixed expert opinions"
  },
  {
    "success": true,
    "time": 68.82785701751709,
    "answer": "Sure, based on the provided information, the patterns with high amplitude and fast activity are:\n\n* Lateralized with the McDelt activity plus S, L-R-D-A plus S.\n* High-amplitude, fast-activity.",
    "length": 193,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Pattern Analysis",
    "question": "What patterns have high amplitude and fast activity?"
  },
  {
    "success": true,
    "time": 75.84973978996277,
    "answer": "Sure, here are the segments with clean signal quality:\n\n- Patient 1003330515 at second 36: Mean power of 998860.59, signal-to-noise ratio of 19.12, and spectral edge frequency at 8.89 Hz are consistent with relaxed wakefulness or posterior rhythms.\n\n- Patient 1001487592 at second 35: Mean power of 1104.22, signal-to-noise ratio of 19.12, and spectral edge frequency at 8.89 Hz are also consistent with relaxed wakefulness or posterior rhythms.",
    "length": 445,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Pattern Analysis",
    "question": "Show segments with clean signal quality"
  },
  {
    "success": true,
    "time": 112.40108513832092,
    "answer": "Sure, here's a comparison between LPD and GPD patterns:\n\n**Generalized Periodic Discharges (GPDs):**\n- Characterized by multiple, isolated, high-amplitude, periodic discharges with clear morphology.\n- Typically observed in focal brain regions, particularly in temporal lobe epilepsy.\n- Can be associated with focal seizures, especially when they are multiple and isolated.\n\n**Lateralized Periodic Discharges (LPDs):**\n- Characterized by multiple, isolated, high-amplitude, periodic discharges with clear morphology.\n- Typically observed in temporal lobe epilepsy, but can also be observed in other focal and diffuse brain regions.\n- Can be associated with focal seizures, but are more likely to be observed in patients with epilepsy with focal onset.\n\n**Comparison:**\n\n| Feature | LPD | GPD |\n|---|---|---|\n| Morphology | Multiple, isolated | Multiple, isolated |\n| Location | Temporal lobe | Focal and diffuse |\n| Associated with | Focal seizures | Focal seizures |\n| Prevalence | Temporal lobe epilepsy | Focal and diffuse epilepsy |\n\n**Additional Notes:**\n\n- LPDs are typically more variable than GPDs in terms of the number and morphology of the discharges.\n- Both LPDs and GPDs can be associated with focal seizures, but GPDs are more likely to be observed in patients with focal onset.\n- The presence of both LPDs and GPDs in the same EEG recording can suggest a complex seizure pattern.",
    "length": 1393,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Comparisons",
    "question": "Compare LPD vs GPD patterns"
  },
  {
    "success": false,
    "time": 120.0440571308136,
    "error": "HTTPConnectionPool(host='localhost', port=8001): Read timed out. (read timeout=120)",
    "category": "Comparisons",
    "question": "Compare seizure in EEG 1001717358 vs 1002197945"
  },
  {
    "success": true,
    "time": 92.89934015274048,
    "answer": "The passage does not provide information about the difference between high and low confidence classifications, so I cannot answer this question from the provided context.",
    "length": 170,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Comparisons",
    "question": "Difference between high and low confidence classifications?"
  },
  {
    "success": true,
    "time": 86.62891006469727,
    "answer": "I am unable to provide a response to the original question as I am unable to access external sources or provide expert opinions or interpretations.",
    "length": 147,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Edge Cases",
    "question": "Tell me about patient 999999999"
  },
  {
    "success": true,
    "time": 83.40779280662537,
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
    "time": 63.246830224990845,
    "answer": "No, the answer is no. \n\nThe passage describes LPDs as bilateral periodic discharges, while low-power discharge is not mentioned.",
    "length": 128,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "N/A",
    "category": "Edge Cases",
    "question": "Is LPD the same as low-power discharge?"
  }
]"""