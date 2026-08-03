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


def generate_llm_response(user_query: str, context_documents: list[str], model: str = "gemma:2b") -> str:

    context = "\n\n---\n\n".join(context_documents)

    prompt = f"""
    You are an expert clinical neurophysiologist explaining a concept to a medical student.

    Your task is to answer the user's query by following these steps precisely:
    1.  **Provide a General Definition:** First, use your own internal knowledge to provide a clear, comprehensive, and fundamental definition of the user's query ("{user_query}"). Explain the core concept in simple terms.
    2.  **Analyze the Provided Context:** Next, review the specific EEG segment summaries provided in the context below.
    3.  **Synthesize and Illustrate:** Finally, connect your general definition to the specific examples from the context. Explain how the findings in the EEG segments (like 'very high amplitude activity', 'beta-dominant activity', 'synchronized neuronal discharges') are practical manifestations of the general definition you provided. Use the context as evidence to support your explanation.

    CONTEXT:
    ---
    {context}
    ---

    USER QUERY:
    {user_query}

    YOUR EXPERT RESPONSE (structured with markdown):
    """

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.6,
            "top_p": 0.9,
            "num_predict": 2048,
            "num_ctx": 8192 }}

    try:
        response = requests.post("http://eegi-ollama:11434/api/generate", json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "No response from LLM.")

    except requests.exceptions.RequestException as e:
        return f"Error connecting to LLM service: {e}"
    except Exception as e:
        return f"An unexpected error occurred during LLM generation: {e}"


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