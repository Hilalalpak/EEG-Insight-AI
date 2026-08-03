# api/main.py
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

app = FastAPI(title="EEG RAG API", description="API for querying EEG segments and generating AI analysis.",)



class QueryRequest(BaseModel):
    query: str
    n_results: int = 5
    filters: dict | None = None


# api/main.py - İyileştirilmiş LLM Response Function

def generate_llm_response(user_query: str, context_documents: list[str], model: str = "gemma:2b") -> str:
    """
    Generates LLM response with strict grounding to prevent hallucination.
    """

    # Context'i numaralandır ve yapılandır
    numbered_context = []
    for i, doc in enumerate(context_documents, 1):
        numbered_context.append(f"[SEGMENT {i}]\n{doc}\n")

    context = "\n".join(numbered_context)

    prompt = f"""You are a clinical neurophysiologist analyzing EEG data. You MUST follow these rules:

CRITICAL RULES:
1. Answer ONLY using information from the EEG segments below
2. ALWAYS cite segment numbers when making claims (e.g., "According to Segment 2...")
3. If the segments don't contain the answer, respond: "The provided EEG segments do not contain sufficient information to answer this question."
4. DO NOT make up medical definitions or terminology
5. DO NOT invent patient data or measurements
6. If asked for a comparison, ONLY compare segments that are actually provided

EEG SEGMENT DATA:
---
{context}
---

USER QUESTION:
{user_query}

INSTRUCTIONS FOR YOUR RESPONSE:
- Start with a direct answer to the question
- Reference specific segment numbers as evidence
- Quote actual measurements (power, SNR, SEF values) from the segments
- If comparing patients, only use data from the segments above
- Use clear section headers (## Overview, ## Findings, ## Clinical Notes)

YOUR RESPONSE:"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,  # ⬇️ Halüsinasyonu minimize et
            "top_p": 0.9,
            "top_k": 20,  # ✅ Response çeşitliliğini sınırla
            "repeat_penalty": 1.2,  # ✅ Tekrarları önle
            "num_predict": 1024,  # ⬇️ Çok uzun yanıtları önle
            "num_ctx": 8192,
            "stop": ["USER QUESTION:", "CRITICAL RULES:"]  # ✅ Prompt leak'i önle
        }
    }

    try:
        response = requests.post(
            "http://eegi-ollama:11434/api/generate",
            json=payload,
            timeout=180
        )
        response.raise_for_status()
        data = response.json()
        llm_output = data.get("response", "No response from LLM.")

        # ✅ Post-processing: Halüsinasyon tespiti
        if _contains_hallucination(llm_output, context_documents):
            return "⚠️ The AI response contained unsupported claims. Please rephrase your question or adjust filters."

        return llm_output

    except requests.exceptions.RequestException as e:
        return f"Error connecting to LLM service: {e}"
    except Exception as e:
        return f"An unexpected error occurred during LLM generation: {e}"


def _contains_hallucination(llm_response: str, context_docs: list[str]) -> bool:
    """
    Basit halüsinasyon kontrolü - geliştirilmeli
    """
    # Yaygın halüsinasyon kalıpları
    suspicious_patterns = [
        "low-amplitude, fast-activity",  # LPD için yanlış tanım
        "GEP", "HFE",  # Var olmayan kısaltmalar
        "I believe", "I think", "probably",  # Belirsizlik ifadeleri
    ]

    llm_lower = llm_response.lower()

    for pattern in suspicious_patterns:
        if pattern.lower() in llm_lower:
            return True

    # Context'te olmayan hasta ID'leri kontrol et
    mentioned_eeg_ids = set()
    for line in llm_response.split('\n'):
        if 'eeg' in line.lower() and any(char.isdigit() for char in line):
            # Basit EEG ID extraction (geliştirilmeli)
            import re
            ids = re.findall(r'\b\d{10}\b', line)
            mentioned_eeg_ids.update(ids)

    context_eeg_ids = set()
    for doc in context_docs:
        import re
        ids = re.findall(r'patient (\d{10})', doc)
        context_eeg_ids.update(ids)

    # Olmayan ID'lerden bahsediyorsa
    if mentioned_eeg_ids - context_eeg_ids:
        return True

    return False

@app.post("/query")
def query_eeg_data(request: QueryRequest):
    try:
        query_embedding = embed_model.encode(request.query, normalize_embeddings=True).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=request.n_results,
            where=request.filters if request.filters else None,
            include=["documents", "metadatas", "distances"])

        llm_response = ""
        if results and results["documents"] and results["documents"][0]:
            llm_response = generate_llm_response(
                user_query=request.query,
                context_documents=results["documents"][0])
        else:
            results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}  # Boş yapı
            llm_response = "No relevant EEG segments were found for your query. Please try adjusting your query or filters."

        return {
            "retrieved_results": results,
            "llm_response": llm_response}
    except Exception as e:
        return {"error": f"An error occurred in the API: {str(e)}"}