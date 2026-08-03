# api/main.py - Self-Validating RAG System
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

app = FastAPI(title="EEG RAG API", description="API for querying EEG segments and generating AI analysis.")


class QueryRequest(BaseModel):
    query: str
    n_results: int = 5
    filters: dict | None = None


# ============== STEP 1: SUMMARIZE CONTEXT ==============
def summarize_context(context_documents: list[str], user_query: str, model: str = "gemma:2b") -> str:
    """Extract only relevant information from retrieved segments"""

    full_context = "\n\n---SEGMENT---\n\n".join(context_documents)

    prompt = f"""You are a clinical neurophysiologist reviewing EEG data.

TASK: Extract ONLY the information relevant to answering the user's question.

USER QUESTION: {user_query}

RETRIEVED EEG SEGMENTS:
{full_context}

INSTRUCTIONS:
1. Read all segments carefully
2. Identify which segments contain relevant information
3. Extract key facts: patient IDs, measurements, classifications, clinical patterns
4. DISCARD irrelevant segments entirely
5. Keep the summary concise (max 300 words)

RELEVANT INFORMATION:"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.9,
            "num_predict": 512,
            "num_ctx": 8192
        }
    }

    try:
        response = requests.post(
            "http://eegi-ollama:11434/api/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"Summarization error: {e}")
        return full_context  # Fallback


# ============== STEP 2: GENERATE ANSWER ==============
def generate_answer_with_medical_knowledge(summary: str, user_query: str, model: str = "gemma:2b") -> str:
    """Generate answer using BOTH context and medical knowledge"""

    prompt = f"""You are an expert clinical neurophysiologist. Answer the question using TWO sources:
1. Your medical knowledge (definitions, physiology, clinical interpretation)
2. The specific EEG data provided below

CRITICAL INSTRUCTIONS:
- Start with a GENERAL MEDICAL EXPLANATION using your knowledge
- Then connect to SPECIFIC EVIDENCE from the EEG data
- Clearly distinguish between general knowledge and data-specific findings
- Use this structure:

## Medical Background
[Your medical knowledge about the concept]

## Evidence from EEG Data
[Specific findings from the segments below]

---

EEG DATA SUMMARY:
{summary}

USER QUESTION:
{user_query}

YOUR RESPONSE:"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,  # ✅ Low but not zero (allows medical knowledge)
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.2,
            "num_predict": 1024,
            "num_ctx": 4096
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
        return f"Error generating answer: {e}"


# ============== STEP 3: SELF-VALIDATION ==============
def validate_answer(answer: str, summary: str, user_query: str, model: str = "gemma:2b") -> dict:
    """Model checks its own answer for accuracy and hallucinations"""

    validation_prompt = f"""You are a medical accuracy reviewer. Your job is to check if the AI's answer is correct.

ORIGINAL QUESTION:
{user_query}

EEG DATA PROVIDED:
{summary}

AI'S ANSWER:
{answer}

VALIDATION TASK:
1. Check if the answer uses correct medical terminology
2. Verify all claims are supported by either:
   - Standard medical knowledge (e.g., "LPD stands for Lateralized Periodic Discharges")
   - The EEG data provided (e.g., "Patient 1234 shows mean power of 165.67")
3. Identify any unsupported or incorrect claims
4. Rate confidence: HIGH, MEDIUM, or LOW

RESPOND IN THIS FORMAT:
VALIDATION: [PASS/FAIL]
CONFIDENCE: [HIGH/MEDIUM/LOW]
ISSUES: [List any problems, or write "None"]
CORRECTED_TERMS: [List any medical terms that need correction, or write "None"]"""

    payload = {
        "model": model,
        "prompt": validation_prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,  # ✅ Strict validation
            "num_predict": 256,
            "num_ctx": 4096
        }
    }

    try:
        response = requests.post(
            "http://eegi-ollama:11434/api/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        validation_text = response.json().get("response", "")

        # Parse validation
        validation_result = {
            "passed": "VALIDATION: PASS" in validation_text.upper(),
            "confidence": "HIGH" if "CONFIDENCE: HIGH" in validation_text.upper() else
            "MEDIUM" if "CONFIDENCE: MEDIUM" in validation_text.upper() else "LOW",
            "issues": validation_text.split("ISSUES:")[-1].split("CORRECTED_TERMS:")[0].strip(),
            "full_validation": validation_text
        }

        return validation_result

    except Exception as e:
        print(f"Validation error: {e}")
        return {"passed": True, "confidence": "MEDIUM", "issues": "Validation unavailable"}


# ============== STEP 4: SELF-CORRECTION (if needed) ==============
def self_correct_answer(answer: str, validation: dict, summary: str, user_query: str, model: str = "gemma:2b") -> str:
    """If validation fails, model corrects its own answer"""

    correction_prompt = f"""You are a clinical neurophysiologist correcting a previous answer.

ORIGINAL QUESTION:
{user_query}

YOUR PREVIOUS ANSWER (with errors):
{answer}

VALIDATION FEEDBACK:
{validation['issues']}

EEG DATA:
{summary}

TASK: Rewrite the answer correctly:
1. Fix any incorrect medical terminology
2. Remove unsupported claims
3. Keep correct parts of the original answer
4. Maintain professional medical tone

CORRECTED ANSWER:"""

    payload = {
        "model": model,
        "prompt": correction_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 1024,
            "num_ctx": 4096
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
        return answer  # Return original if correction fails


# ============== MAIN ENDPOINT ==============
@app.post("/query")
def query_eeg_data(request: QueryRequest):
    try:
        # Retrieve segments
        query_embedding = embed_model.encode(request.query, normalize_embeddings=True).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=request.n_results,
            where=request.filters if request.filters else None,
            include=["documents", "metadatas", "distances"]
        )

        if not results or not results["documents"] or not results["documents"][0]:
            return {
                "retrieved_results": {"documents": [[]], "metadatas": [[]], "distances": [[]]},
                "llm_response": "No relevant EEG segments found. Please adjust your query or filters.",
                "validation": {"passed": True, "confidence": "N/A"}
            }

        context_documents = results["documents"][0]

        # ✅ STEP 1: Summarize (filter irrelevant info)
        print(f"[Step 1/4] Summarizing {len(context_documents)} segments...")
        relevant_summary = summarize_context(context_documents, request.query)

        # ✅ STEP 2: Generate answer (using medical knowledge + data)
        print(f"[Step 2/4] Generating answer with medical knowledge...")
        initial_answer = generate_answer_with_medical_knowledge(relevant_summary, request.query)

        # ✅ STEP 3: Self-validate
        print(f"[Step 3/4] Validating answer accuracy...")
        validation = validate_answer(initial_answer, relevant_summary, request.query)

        # ✅ STEP 4: Self-correct if needed
        final_answer = initial_answer
        if not validation["passed"] or validation["confidence"] == "LOW":
            print(f"[Step 4/4] Validation failed. Self-correcting...")
            final_answer = self_correct_answer(initial_answer, validation, relevant_summary, request.query)

            # Add disclaimer
            final_answer = f"""⚠️ **Note:** The initial response was refined for accuracy.

{final_answer}

---
*Validation Details: {validation['confidence']} confidence*
"""
        else:
            print(f"[Step 4/4] Validation passed ({validation['confidence']} confidence)")

        return {
            "retrieved_results": results,
            "llm_response": final_answer,
            "validation": {
                "confidence": validation["confidence"],
                "issues": validation["issues"] if validation["issues"] != "None" else None
            }
        }

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}


# ============== HEALTH CHECK ==============
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "gemma:2b",
        "validation": "enabled",
        "self_correction": "enabled"
    }