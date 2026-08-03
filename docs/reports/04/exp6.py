# api/main.py - Hybrid EEG RAG System with Intelligent Query Routing
import re
import requests
import chromadb
import ast
from enum import Enum
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import Optional, Dict, List, Tuple

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
video_collection = chroma_client.get_or_create_collection(
    name="eeg_medical_video_knowledge",
    metadata={"hnsw:space": "cosine"}
)
print("Connected to ChromaDB. All collections ready.")

app = FastAPI(title="EEG RAG API - Hybrid Intelligence System")


class QueryType(Enum):
    PATIENT_SPECIFIC = "patient_specific"
    TERMINOLOGY = "terminology"
    COMPARISON = "comparison"
    PATTERN_ANALYSIS = "pattern_analysis"


class QueryRequest(BaseModel):
    query: str
    n_results: int = 5
    n_definitions: int = 3
    n_videos: int = 2
    filters: Optional[Dict] = None
    force_strategy: Optional[str] = None  # Override auto-routing


# ============================================================================
# QUERY CLASSIFICATION & ROUTING
# ============================================================================

def classify_query(query: str) -> QueryType:
    """
    Intelligent query classifier - routes to optimal pipeline
    Based on exp1-5 analysis findings
    """
    q = query.lower()

    # PATIENT SPECIFIC: Use exp1 strategy (fast, reliable)
    patient_patterns = [
        r'patient\s+\d+',
        r'patient_id',
        r'tell me about patient',
        r'find patient',
        r'show.*patient',
        r'eeg\s+\d+'
    ]
    if any(re.search(p, q) for p in patient_patterns):
        return QueryType.PATIENT_SPECIFIC

    # COMPARISON: Use exp3 strategy (structured, concise)
    comparison_keywords = [
        'compare', 'difference between', 'differentiate',
        'vs', 'versus', 'contrast', 'vs.', 'what is the difference'
    ]
    if any(kw in q for kw in comparison_keywords):
        return QueryType.COMPARISON

    # TERMINOLOGY: Use exp5 strategy (video-prioritized, multi-query)
    terminology_keywords = [
        'what is', 'what are', 'explain', 'define', 'meaning of',
        'lpd', 'gpd', 'lrda', 'grda', 'birds', 'seizure',
        'ictal', 'interictal', 'continuum', 'terminology',
        'clinical significance', 'interpretation'
    ]
    if any(kw in q for kw in terminology_keywords):
        return QueryType.TERMINOLOGY

    # PATTERN ANALYSIS: Default to balanced approach
    return QueryType.PATTERN_ANALYSIS


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_patient_id(query: str) -> Optional[str]:
    """Extract patient/EEG ID from query"""
    patterns = [
        r'patient\s+(?:id\s+)?(\d+)',
        r'patient_id[:\s]+(\d+)',
        r'eeg\s+(?:id\s+)?(\d+)',
        r'\b(\d{10})\b'
    ]
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return match.group(1)
    return None


def call_llm(prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
    """Optimized LLM call with timeout handling"""
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


# ============================================================================
# STRATEGY 1: PATIENT-SPECIFIC (exp1 approach)
# ============================================================================

def patient_specific_pipeline(
        query: str,
        eeg_docs: List[str],
        medical_docs: List[str]
) -> Tuple[str, Dict]:
    """
    Fast and reliable patient-specific queries
    exp1 proven best for: patient data, specific EEG segments
    """
    medical_text = "\n".join(medical_docs[:2])[:600] if medical_docs else ""
    eeg_text = "\n".join(eeg_docs[:3])[:800] if eeg_docs else ""

    prompt = f"""Medical info: {medical_text}

Patient data: {eeg_text}

Q: {query}

A (explain term + describe findings):"""

    answer = call_llm(prompt, temperature=0.2, max_tokens=512)

    validation = {
        "strategy": "patient_specific",
        "confidence": "HIGH" if len(answer) > 150 else "MEDIUM",
        "sources_used": {
            "eeg_segments": len(eeg_docs),
            "medical_defs": len(medical_docs),
            "video_knowledge": 0
        }
    }

    return answer, validation


# ============================================================================
# STRATEGY 2: TERMINOLOGY (exp5 approach with improvements)
# ============================================================================

def generate_sub_queries_for_terminology(query: str) -> List[str]:
    """
    Smart sub-query generation for terminology
    Focuses on ACNS video content and medical definitions
    """
    prompt = f"""You are an expert in EEG terminology from ACNS Critical Care guidelines.
Break down this question into 2-3 simple search queries to find:
1. Definition from medical sources
2. Clinical interpretation from expert videos
3. Practical examples from patient data

Output MUST be a Python list of strings.

Examples:
Q: "What is LPD?"
A: ["lateralized periodic discharges definition", "LPD clinical interpretation EEG", "examples of LPD in patient recordings"]

Q: "What are BIRDs in EEG?"
A: ["brief potentially ictal rhythmic discharges", "BIRDs ACNS terminology", "BIRDs EEG pattern characteristics"]

Q: "{query}"
A:"""

    try:
        response = call_llm(prompt, temperature=0.0, max_tokens=150)
        sub_queries = ast.literal_eval(response)
        if isinstance(sub_queries, list) and sub_queries:
            sub_queries.append(query)  # Include original
            return list(set(sub_queries))[:3]  # Max 3 to control latency
        return [query]
    except:
        return [query]


def terminology_pipeline(
        query: str,
        eeg_docs: List[str],
        medical_docs: List[str],
        video_docs: List[str]
) -> Tuple[str, Dict]:
    """
    Video-prioritized terminology explanation
    exp5 proven best for: LPD, GPD, BIRDs, Ictal-Interictal
    Critical: ACNS video content contains expert interpretation
    """
    # Prioritize video content for terminology
    video_text = "\n".join(video_docs[:3])[:700] if video_docs else ""
    medical_text = "\n".join(medical_docs[:2])[:500] if medical_docs else ""
    eeg_text = "\n".join(eeg_docs[:2])[:600] if eeg_docs else ""

    prompt = f"""### ACNS Critical Care EEG Expert Commentary (PRIORITY SOURCE):
{video_text}

### Medical Definitions:
{medical_text}

### Relevant Patient Examples (if applicable):
{eeg_text}

### Task:
You are an EEG expert trained on ACNS Critical Care EEG Terminology 2021.
Use the Expert Commentary above as your PRIMARY source to explain the terminology.
Provide a clear, clinically accurate definition with interpretation guidelines.
If patient examples are relevant, mention how the concept appears in practice.

### Question: {query}

### Expert Answer:"""

    answer = call_llm(prompt, temperature=0.15, max_tokens=600)

    validation = {
        "strategy": "terminology",
        "confidence": "HIGH" if video_docs else "MEDIUM",
        "sources_used": {
            "eeg_segments": len(eeg_docs),
            "medical_defs": len(medical_docs),
            "video_knowledge": len(video_docs)
        }
    }

    return answer, validation


# ============================================================================
# STRATEGY 3: COMPARISON (exp3 approach)
# ============================================================================

def comparison_pipeline(
        query: str,
        eeg_docs: List[str],
        medical_docs: List[str],
        video_docs: List[str]
) -> Tuple[str, Dict]:
    """
    Structured comparison with table format
    exp3 proven best for: LPD vs GPD, pattern comparisons
    """
    video_text = "\n".join(video_docs[:2])[:500] if video_docs else ""
    medical_text = "\n".join(medical_docs[:2])[:500] if medical_docs else ""
    eeg_text = "\n".join(eeg_docs[:3])[:700] if eeg_docs else ""

    prompt = f"""### Expert Video Knowledge:
{video_text}

### Medical Definitions:
{medical_text}

### Patient Data Examples:
{eeg_text}

### Task:
Compare the items in the question using the sources above.
Structure your answer clearly:
1. Brief definition of each item (2-3 sentences each)
2. Key differences in a concise format
3. Clinical significance of the differences

Keep the response focused and structured. Avoid excessive length.

### Question: {query}

### Comparative Analysis:"""

    answer = call_llm(prompt, temperature=0.2, max_tokens=512)

    validation = {
        "strategy": "comparison",
        "confidence": "MEDIUM",
        "sources_used": {
            "eeg_segments": len(eeg_docs),
            "medical_defs": len(medical_docs),
            "video_knowledge": len(video_docs)
        }
    }

    return answer, validation


# ============================================================================
# STRATEGY 4: PATTERN ANALYSIS (Balanced approach)
# ============================================================================

def pattern_analysis_pipeline(
        query: str,
        eeg_docs: List[str],
        medical_docs: List[str],
        video_docs: List[str]
) -> Tuple[str, Dict]:
    """
    Balanced approach for general pattern analysis
    Uses all sources with equal weight
    """
    video_text = "\n".join(video_docs[:2])[:400] if video_docs else ""
    medical_text = "\n".join(medical_docs[:2])[:400] if medical_docs else ""
    eeg_text = "\n".join(eeg_docs[:4])[:800] if eeg_docs else ""

    prompt = f"""Medical Knowledge: {medical_text}

Expert Guidance: {video_text}

Patient EEG Data: {eeg_text}

Q: {query}

A (analyze patterns using all available evidence):"""

    answer = call_llm(prompt, temperature=0.2, max_tokens=512)

    validation = {
        "strategy": "pattern_analysis",
        "confidence": "MEDIUM",
        "sources_used": {
            "eeg_segments": len(eeg_docs),
            "medical_defs": len(medical_docs),
            "video_knowledge": len(video_docs)
        }
    }

    return answer, validation


# ============================================================================
# MAIN QUERY ENDPOINT
# ============================================================================

@app.post("/query")
def query_eeg_data(request: QueryRequest):
    try:
        # Step 1: Classify query type
        if request.force_strategy:
            query_type = QueryType(request.force_strategy)
        else:
            query_type = classify_query(request.query)

        # Step 2: Extract patient ID if applicable
        patient_id = extract_patient_id(request.query)
        where_filter = request.filters

        if patient_id and not where_filter and query_type == QueryType.PATIENT_SPECIFIC:
            where_filter = {"eeg_id": {"$eq": patient_id}}

        # Step 3: Adaptive retrieval based on query type
        query_embedding = embed_model.encode(request.query, normalize_embeddings=True).tolist()

        # For terminology: Use sub-queries and prioritize video
        if query_type == QueryType.TERMINOLOGY:
            sub_queries = generate_sub_queries_for_terminology(request.query)

            unique_video_docs = set()
            unique_medical_docs = set()
            unique_eeg_docs = set()

            for sub_q in sub_queries:
                sub_embedding = embed_model.encode(sub_q, normalize_embeddings=True).tolist()

                # Video is PRIORITY for terminology
                v_results = video_collection.query(
                    query_embeddings=[sub_embedding],
                    n_results=3,
                    include=["documents"]
                )
                for doc in v_results.get("documents", [[]])[0]:
                    unique_video_docs.add(doc)

                m_results = medical_collection.query(
                    query_embeddings=[sub_embedding],
                    n_results=2,
                    include=["documents"]
                )
                for doc in m_results.get("documents", [[]])[0]:
                    unique_medical_docs.add(doc)

                # Less EEG data for terminology
                e_results = eeg_collection.query(
                    query_embeddings=[sub_embedding],
                    n_results=1,
                    include=["documents"]
                )
                for doc in e_results.get("documents", [[]])[0]:
                    unique_eeg_docs.add(doc)

            eeg_docs = list(unique_eeg_docs)
            medical_docs = list(unique_medical_docs)
            video_docs = list(unique_video_docs)

            eeg_results = {"documents": [eeg_docs]}
            medical_results = {"documents": [medical_docs]}
            video_results = {"documents": [video_docs]}

        # For patient-specific: Focus on EEG data (exp1 approach)
        elif query_type == QueryType.PATIENT_SPECIFIC:
            eeg_results = eeg_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(request.n_results, 5),
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )

            medical_results = medical_collection.query(
                query_embeddings=[query_embedding],
                n_results=2,
                include=["documents"]
            )

            # Minimal video for patient queries
            video_results = {"documents": [[]]}

            eeg_docs = eeg_results.get("documents", [[]])[0]
            medical_docs = medical_results.get("documents", [[]])[0]
            video_docs = []

            # Fallback semantic search if no results
            if not eeg_docs and patient_id:
                semantic_query = f"patient {patient_id} EEG recording"
                semantic_embedding = embed_model.encode(semantic_query, normalize_embeddings=True).tolist()

                eeg_results = eeg_collection.query(
                    query_embeddings=[semantic_embedding],
                    n_results=5,
                    include=["documents", "metadatas", "distances"]
                )
                eeg_docs = eeg_results.get("documents", [[]])[0]

        # For comparison and pattern analysis: Balanced retrieval
        else:
            eeg_results = eeg_collection.query(
                query_embeddings=[query_embedding],
                n_results=4,
                include=["documents", "metadatas", "distances"]
            )

            medical_results = medical_collection.query(
                query_embeddings=[query_embedding],
                n_results=2,
                include=["documents"]
            )

            video_results = video_collection.query(
                query_embeddings=[query_embedding],
                n_results=2,
                include=["documents"]
            )

            eeg_docs = eeg_results.get("documents", [[]])[0]
            medical_docs = medical_results.get("documents", [[]])[0]
            video_docs = video_results.get("documents", [[]])[0]

        # Step 4: Check if we have sufficient data
        if not eeg_docs and not medical_docs and not video_docs:
            return {
                "llm_response": f"No relevant data found for: '{request.query}'",
                "strategy_used": query_type.value,
                "retrieved_sources": {"eeg": 0, "medical": 0, "video": 0}
            }

        # Step 5: Route to appropriate pipeline
        if query_type == QueryType.PATIENT_SPECIFIC:
            final_answer, validation = patient_specific_pipeline(
                request.query, eeg_docs, medical_docs
            )
        elif query_type == QueryType.TERMINOLOGY:
            final_answer, validation = terminology_pipeline(
                request.query, eeg_docs, medical_docs, video_docs
            )
        elif query_type == QueryType.COMPARISON:
            final_answer, validation = comparison_pipeline(
                request.query, eeg_docs, medical_docs, video_docs
            )
        else:  # PATTERN_ANALYSIS
            final_answer, validation = pattern_analysis_pipeline(
                request.query, eeg_docs, medical_docs, video_docs
            )

        # Step 6: Return comprehensive response
        return {
            "llm_response": final_answer,
            "strategy_used": query_type.value,
            "validation": validation,
            "retrieved_sources": {
                "eeg_segments": eeg_results,
                "medical_definitions": medical_results,
                "video_knowledge": video_results
            },
            "queried_patient_id": patient_id
        }

    except Exception as e:
        return {"error": f"API error: {str(e)}"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "system": "Hybrid EEG RAG",
        "strategies": [s.value for s in QueryType],
        "collections": ["eeg_insights", "medical_definitions", "eeg_medical_video_knowledge"]
    }


@app.get("/classify")
def classify_query_endpoint(query: str):
    """Debug endpoint to test query classification"""
    query_type = classify_query(query)
    patient_id = extract_patient_id(query)

    return {
        "query": query,
        "classified_as": query_type.value,
        "extracted_patient_id": patient_id,
        "recommended_strategy": {
            QueryType.PATIENT_SPECIFIC: "Fast retrieval (exp1)",
            QueryType.TERMINOLOGY: "Video-prioritized (exp5)",
            QueryType.COMPARISON: "Structured format (exp3)",
            QueryType.PATTERN_ANALYSIS: "Balanced approach"
        }[query_type]
    }


"""[
  {
    "success": true,
    "time": 107.6408040523529,
    "answer": "Sure, here's the definition of LPD according to the ACNS Critical Care EEG Terminology 2021:\n\n**Left Periodic Discharges (LPDs)**: Bilateral periodic discharges seen in one hemisphere with a clear and consistent pattern.",
    "length": 220,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "HIGH",
    "category": "Medical Terms",
    "question": "What is LPD?"
  },
  {
    "success": true,
    "time": 87.19141364097595,
    "answer": "Sure, here's the explanation of GPD and its clinical significance based on the ACNS Critical Care EEG Terminology 2021:\n\n**GPD (Generalized Periodic Discharges Plusfast Activity):**\n\nGeneralized Periodic Discharges Plusfast Activity (GPD) is a pattern characterized by the presence of periodic discharges with a faster frequency than typical periodic discharges. This pattern is typically seen in patients with epilepsy and can be associated with seizures or other neurological disorders.\n\n**Clinical Significance:**\n\nGPD is a clinically significant pattern because it can indicate a variety of underlying neurological conditions, including epilepsy, intracranial neoplasia, and metabolic disorders. Early diagnosis and appropriate management of GPD is essential for preventing potential complications.\n\n**Clinical Examples:**\n\nThe patient example provided in the context is a case of GPD. The EEG segment shows periodic discharges with a faster frequency than typical periodic discharges, suggesting GPD. This pattern is consistent with the clinical presentation of epilepsy and can be associated with seizures.",
    "length": 1112,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "HIGH",
    "category": "Medical Terms",
    "question": "Explain GPD and its clinical significance"
  },
  {
    "success": true,
    "time": 68.91153693199158,
    "answer": "The passage does not define BIRDS in EEG terminology, so I cannot answer this question from the provided context.",
    "length": 113,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "HIGH",
    "category": "Medical Terms",
    "question": "What are BIRDs in EEG terminology?"
  },
  {
    "success": true,
    "time": 60.42252802848816,
    "answer": "Error: HTTPConnectionPool(host='eegi-ollama', port=11434): Read timed out. (read timeout=60)",
    "length": 92,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "MEDIUM",
    "category": "Medical Terms",
    "question": "Differentiate between LRDA and GRDA"
  },
  {
    "success": true,
    "time": 78.62960720062256,
    "answer": "The Ictal-Interictal Continuum refers to the gradual transition between the electrographic features of an epileptic seizure and the electrographic features of an interictal period. This transition can be subtle and can be difficult to distinguish from a simple seizure. However, any EEG pattern that has a definite clinical correlate, it's time-locked to the pattern, is considered an Ictal-Interictal Continuum.",
    "length": 412,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "HIGH",
    "category": "Medical Terms",
    "question": "What is the Ictal-Interictal Continuum?"
  },
  {
    "success": true,
    "time": 43.58729815483093,
    "answer": "Patient 1002379034 is a critically ill individual whose EEG segment at second 37 shows high amplitude activity with a signal-to-noise ratio of 11.12 and an edge frequency of 10.47 Hz. The segment is described as a mixed-pattern, uncertain, high-amplitude, clean-signal.\n\nFindings:\n\n* Mean power: 1687.01\n* Signal-to-noise ratio: 11.12\n* Spectral edge frequency: 10.47 Hz",
    "length": 370,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "HIGH",
    "category": "Patient Queries",
    "question": "Tell me about patient 1002379034"
  },
  {
    "success": true,
    "time": 49.575406074523926,
    "answer": "The EEG segment from patient 1001717358 at second 173 contains high-amplitude activity with a mean power of 1370.15 and a signal-to-noise ratio of 10.57, indicating excellent signal quality. The spectral edge frequency is at 11.79 Hz, which is consistent with alpha-range activity.\n\nThe EEG segment from patient 1001717358 at second 119 contains very high-amplitude activity with a mean power of 1882.66 and a signal-to-noise ratio of 10.57, indicating excellent signal quality. The spectral edge frequency is at 11.79 Hz, which is consistent with alpha-range activity.",
    "length": 569,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "HIGH",
    "category": "Patient Queries",
    "question": "What patterns are in patient 1001717358?"
  },
  {
    "success": true,
    "time": 32.70598387718201,
    "answer": "The context does not provide any seizure events for patient 1001717358, so I cannot generate the requested information from the context.",
    "length": 136,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "MEDIUM",
    "category": "Patient Queries",
    "question": "Show seizure events for patient 1001717358"
  },
  {
    "success": true,
    "time": 32.14083504676819,
    "answer": "The context does not provide any information about patient 42516, so I cannot answer this question from the provided context.",
    "length": 125,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "MEDIUM",
    "category": "Patient Queries",
    "question": "Find patient 42516 data"
  },
  {
    "success": true,
    "time": 33.3649206161499,
    "answer": "The confidence for patient 1002379034's LPD is not explicitly mentioned in the context, so I cannot answer this question from the provided context.",
    "length": 147,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "MEDIUM",
    "category": "Patient Queries",
    "question": "What is the confidence for patient 1002379034's LPD?"
  },
  {
    "success": true,
    "time": 88.06677198410034,
    "answer": "Sure, here are the high-confidence seizure segments from patient 1002197945 at second 20:\n\n- Mean power of 2559.95 indicating very high amplitude activity\n- Signal-to-noise ratio of 6.47 showing good signal quality\n- Spectral edge frequency at 16.37 Hz representing beta-dominant activity suggesting arousal or ictal patterns",
    "length": 325,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "HIGH",
    "category": "Pattern Analysis",
    "question": "Show high-confidence seizure segments"
  },
  {
    "success": true,
    "time": 91.39637589454651,
    "answer": "Error: HTTPConnectionPool(host='eegi-ollama', port=11434): Read timed out. (read timeout=60)",
    "length": 92,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "HIGH",
    "category": "Pattern Analysis",
    "question": "What are typical seizure signal characteristics?"
  },
  {
    "success": true,
    "time": 45.37322783470154,
    "answer": "Sure, here's the answer to your question:\n\nThe EEG segment from patient 1002142157 at second 46 is classified as GPD with mixed expert opinions (53% agreement from 15 expert votes), indicating an edge case or transitional pattern.",
    "length": 230,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "MEDIUM",
    "category": "Pattern Analysis",
    "question": "Find segments with mixed expert opinions"
  },
  {
    "success": true,
    "time": 60.63320207595825,
    "answer": "Error: HTTPConnectionPool(host='eegi-ollama', port=11434): Read timed out. (read timeout=60)",
    "length": 92,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "MEDIUM",
    "category": "Pattern Analysis",
    "question": "What patterns have high amplitude and fast activity?"
  },
  {
    "success": true,
    "time": 59.654690980911255,
    "answer": "The EEG segment from patient 1001487592 at second 35 is clean signal quality. The mean power of 1104.22 indicates very high amplitude activity, the signal-to-noise ratio of 5.06 shows good signal quality, and the spectral edge frequency at 20.00 Hz suggests beta-dominant activity.\n\nThe EEG segment from patient 1002197945 at second 20 is classified as SEIZURE with high expert consensus (100% agreement from 3 expert votes). The mean power of 2559.95 indicates very high amplitude activity, the signal-to-noise ratio of 6.47 shows good signal quality, and the spectral edge frequency at 20.00 Hz suggests beta-dominant activity.",
    "length": 629,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "MEDIUM",
    "category": "Pattern Analysis",
    "question": "Show segments with clean signal quality"
  },
  {
    "success": true,
    "time": 60.3916130065918,
    "answer": "Error: HTTPConnectionPool(host='eegi-ollama', port=11434): Read timed out. (read timeout=60)",
    "length": 92,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "MEDIUM",
    "category": "Comparisons",
    "question": "Compare LPD vs GPD patterns"
  },
  {
    "success": true,
    "time": 60.79839730262756,
    "answer": "Error: HTTPConnectionPool(host='eegi-ollama', port=11434): Read timed out. (read timeout=60)",
    "length": 92,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "MEDIUM",
    "category": "Comparisons",
    "question": "Compare seizure in EEG 1001717358 vs 1002197945"
  },
  {
    "success": true,
    "time": 61.19970202445984,
    "answer": "Error: HTTPConnectionPool(host='eegi-ollama', port=11434): Read timed out. (read timeout=60)",
    "length": 92,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "MEDIUM",
    "category": "Comparisons",
    "question": "Difference between high and low confidence classifications?"
  },
  {
    "success": true,
    "time": 33.69603371620178,
    "answer": "The context does not provide any information about patient 999999999, so I cannot answer this question from the provided context.",
    "length": 129,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "MEDIUM",
    "category": "Edge Cases",
    "question": "Tell me about patient 999999999"
  },
  {
    "success": true,
    "time": 72.88550209999084,
    "answer": "The context does not provide any information about the XYZ pattern, so I cannot answer this question from the provided context.",
    "length": 127,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "HIGH",
    "category": "Edge Cases",
    "question": "What is XYZ pattern?"
  },
  {
    "success": true,
    "time": 68.37681603431702,
    "answer": "No, LPD and low-power discharge are not the same. LPD refers to bilateral periodic discharges, while low-power discharge can be unilateral or bilateral.",
    "length": 152,
    "eeg_count": 0,
    "medical_count": 0,
    "confidence": "HIGH",
    "category": "Edge Cases",
    "question": "Is LPD the same as low-power discharge?"
  }
]"""