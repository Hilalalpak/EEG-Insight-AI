# api/main.py - Mac Air 8GB Final Optimized with CONTEXTUAL YouTube Integration
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
video_collection = chroma_client.get_or_create_collection(
    name="eeg_medical_video_knowledge",
    metadata={"hnsw:space": "cosine"}
)
print("Connected to ChromaDB and all collections.")

app = FastAPI(title="EEG RAG API")


class QueryRequest(BaseModel):
    query: str
    n_results: int = 5
    n_definitions: int = 2
    n_videos: int = 2  # Video'dan alınacak sonuç sayısını da kontrol edelim
    filters: dict | None = None


def extract_patient_id(query: str) -> str | None:
    patterns = [r'patient\s+(?:id\s+)?(\d+)', r'patient_id[:\s]+(\d+)', r'\b(\d{10})\b']
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return match.group(1)
    return None


# YENİ ve DAHA AKILLI YAKLAŞIM: Dinamik Prompt Oluşturma
def build_contextual_prompt(query: str, eeg_docs: list[str], medical_docs: list[str], video_docs: list[str]) -> str:
    """
    Bulunan kaynaklara göre LLM için en uygun prompt'u dinamik olarak oluşturur.
    Bu, LLM'e sadece veri vermekle kalmaz, veriyi nasıl kullanacağını da söyler.
    """
    prompt_parts = []

    # 1. Tıbbi ve Video Tanımları: Bunlar temel bilgiyi sağlar.
    if medical_docs:
        medical_text = "\n".join(medical_docs)[:600]
        prompt_parts.append(f"Tıbbi Tanımlar:\n{medical_text}")

    if video_docs:
        video_text = "\n".join(video_docs)[:500]
        prompt_parts.append(f"Uzman Video Açıklamaları:\n{video_text}")

    # 2. Hasta Verisi: Bu, vaka özelindeki kanıttır.
    if eeg_docs:
        eeg_text = "\n".join(eeg_docs)[:800]
        prompt_parts.append(f"İlgili Hasta EEG Bulguları:\n{eeg_text}")

    # 3. LLM için Talimat: Prompt'un en kritik kısmı.
    # LLM'e, elindeki bilgileri nasıl sentezleyeceğini adım adım anlatıyoruz.
    instruction = "\n\n---"
    if video_docs and eeg_docs:
        instruction += "\nTalimat: Yukarıdaki Uzman Video Açıklamalarını ve Tıbbi Tanımları kullanarak sorudaki anahtar kavramı bir nörolog gibi açıkla. Ardından, bu kavramın Hasta EEG Bulgularında nasıl ortaya çıktığını analiz ederek somut bir cevap oluştur."
    elif video_docs:
        instruction += "\nTalimat: Yukarıdaki Uzman Video Açıklamalarını ve Tıbbi Tanımları temel alarak soruyu kapsamlı bir şekilde yanıtla."
    elif eeg_docs:
        instruction += "\nTalimat: Yukarıdaki Hasta EEG Bulgularını analiz et ve soruya bu bulgulara dayanarak cevap ver."
    else:
        instruction += "\nTalimat: Elindeki bilgilere dayanarak soruyu cevapla."

    final_prompt = "\n\n".join(prompt_parts) + instruction + f"\n\nSoru: {query}\n\nCevap:"
    return final_prompt


def call_llm(prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
    """Mac Air optimized"""
    payload = {
        "model": "gemma:2b", "prompt": prompt, "stream": False,
        "options": {"temperature": temperature, "top_p": 0.9, "top_k": 40, "num_predict": max_tokens, "num_ctx": 3072,
                    "num_thread": 4}
    }
    try:
        response = requests.post("http://eegi-ollama:11434/api/generate", json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"Error: {e}"


@app.post("/query")
def query_eeg_data(request: QueryRequest):
    try:
        patient_id = extract_patient_id(request.query)

        # Filtreleme mantığı aynı kalıyor
        where_filter = request.filters
        if patient_id and not where_filter:
            where_filter = {"eeg_id": {"$eq": patient_id}}

        query_embedding = embed_model.encode(request.query, normalize_embeddings=True).tolist()

        # ARTIK HER ZAMAN TÜM KAYNAKLARI SORGULUYORUZ
        # Karmaşık kurallar yerine, her kaynaktan en ilgili bilgileri alıp
        # LLM'in bunları akıllıca kullanmasını sağlıyoruz.
        eeg_results = eeg_collection.query(
            query_embeddings=[query_embedding], n_results=request.n_results, where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        medical_results = medical_collection.query(
            query_embeddings=[query_embedding], n_results=request.n_definitions,
            include=["documents"]
        )
        video_results = video_collection.query(
            query_embeddings=[query_embedding], n_results=request.n_videos,
            include=["documents"]
        )

        eeg_docs = eeg_results.get("documents", [[]])[0]
        medical_docs = medical_results.get("documents", [[]])[0]
        video_docs = video_results.get("documents", [[]])[0]

        # Eğer hiçbir şey bulunamazsa, standart cevap ver
        if not eeg_docs and not medical_docs and not video_docs:
            return {"llm_response": "Üzgünüm, bu soruyla ilgili herhangi bir veri bulamadım."}

        # YENİ YAKLAŞIM: Dinamik ve bağlamsal prompt oluştur
        final_prompt = build_contextual_prompt(
            query=request.query, eeg_docs=eeg_docs, medical_docs=medical_docs, video_docs=video_docs
        )

        final_answer = call_llm(final_prompt)

        return {
            "retrieved_eeg_segments": eeg_results,
            "retrieved_medical_definitions": medical_results,
            "retrieved_video_knowledge": video_results,
            "llm_response": final_answer,
            "queried_eeg_id": patient_id
        }

    except Exception as e:
        return {"error": f"API error: {str(e)}"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "mode": "mac_air_optimized"}


"""[
  {
    "success": true,
    "time": 42.791906118392944,
    "answer": "Lateralized periodic discharges (LPDs) are a pattern of rhythmic and periodic electrical activity that is highly associated with seizures in critically ill patients.",
    "length": 165,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "N/A",
    "category": "Medical Terms",
    "question": "What is LPD?"
  },
  {
    "success": true,
    "time": 35.72167181968689,
    "answer": "GPD stands for generalized periodic discharge. It is a pattern of rhythmic and periodic activity in the EEG that is seen in many types of epilepsy, including focal epilepsy and temporal lobe epilepsy.\n\nClinical significance of GPD is that it can be a sign of a seizure or an interictal-ictal transition. It is often seen in patients with epilepsy who are taking anti-epileptic medications.",
    "length": 389,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "N/A",
    "category": "Medical Terms",
    "question": "Explain GPD and its clinical significance"
  },
  {
    "success": true,
    "time": 46.27640223503113,
    "answer": "The passage does not specify what BIRDs are, so I cannot answer this question from the provided context.",
    "length": 104,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "N/A",
    "category": "Medical Terms",
    "question": "What are BIRDs in EEG terminology?"
  },
  {
    "success": true,
    "time": 40.133405923843384,
    "answer": "Sure, here's the answer to the question:\n\nThe passage does not provide information about the difference between LRDA and GRDA, so I cannot answer this question from the provided context.",
    "length": 186,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "N/A",
    "category": "Medical Terms",
    "question": "Differentiate between LRDA and GRDA"
  },
  {
    "success": true,
    "time": 53.72273516654968,
    "answer": "The Ictal-Interictal Continuum is a term used to describe the transition between seizures and periods of normal brain activity. It is characterized by a gradual change in the EEG pattern that occurs as the seizure transitions from one phase to another. This process can be observed in patients with epilepsy who have frequent seizures, and can also be seen in patients who have epilepsy that is triggered by certain stimuli, such as stress or alcohol.\n\nThe Ictal-Interictal Continuum is a complex and challenging area of research, but it is important for understanding how epilepsy progresses and how it can be treated. By studying the Ictal-Interictal Continuum, researchers can learn more about the disease and develop new treatments for epilepsy.",
    "length": 749,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "N/A",
    "category": "Medical Terms",
    "question": "What is the Ictal-Interictal Continuum?"
  },
  {
    "success": true,
    "time": 60.28196573257446,
    "answer": "Error: HTTPConnectionPool(host='eegi-ollama', port=11434): Read timed out. (read timeout=60)",
    "length": 92,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "N/A",
    "category": "Patient Queries",
    "question": "Tell me about patient 1002379034"
  },
  {
    "success": true,
    "time": 42.69649577140808,
    "answer": "The patterns in patient 1001717358 are lateralized rhythmicdelta activity (LRDA) and lateralized periodic discharges (LPDs).",
    "length": 124,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "N/A",
    "category": "Patient Queries",
    "question": "What patterns are in patient 1001717358?"
  },
  {
    "success": true,
    "time": 42.90370798110962,
    "answer": "The context does not provide any information about seizure events for patient 1001717358, so I cannot generate the requested answer from the context.",
    "length": 149,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "N/A",
    "category": "Patient Queries",
    "question": "Show seizure events for patient 1001717358"
  },
  {
    "success": true,
    "time": 39.822556257247925,
    "answer": "The context does not provide any information about patient 42516, so I cannot answer this question from the provided context.",
    "length": 125,
    "eeg_count": 0,
    "medical_count": 2,
    "confidence": "N/A",
    "category": "Patient Queries",
    "question": "Find patient 42516 data"
  },
  {
    "success": true,
    "time": 48.6178240776062,
    "answer": "The confidence for patient 1002379034's LPD is not provided in the context, so I cannot answer this question from the provided context.",
    "length": 135,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "N/A",
    "category": "Patient Queries",
    "question": "What is the confidence for patient 1002379034's LPD?"
  },
  {
    "success": true,
    "time": 49.93296408653259,
    "answer": "Sure, here's the answer to the question:\n\nThe high-confidence seizure segments from patient 1002197945 at second 20 and 5 are classified as seizures with high expert consensus (100% agreement from 3 expert votes). These segments represent abnormal synchronized neuronal discharges requiring immediate clinical attention.",
    "length": 320,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "N/A",
    "category": "Pattern Analysis",
    "question": "Show high-confidence seizure segments"
  },
  {
    "success": true,
    "time": 46.740362882614136,
    "answer": "Suretypical seizure signal characteristics include:\n\n- Mean power of 4681.43 indicating very high amplitude activity\n- Signal-to-noise ratio of 26.28 showing excellent signal quality with minimal artifact\n- Spectral edge frequency at 5.21 Hz representing theta-dominant activity seen in drowsiness or temporal lobe pathology",
    "length": 324,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "N/A",
    "category": "Pattern Analysis",
    "question": "What are typical seizure signal characteristics?"
  },
  {
    "success": true,
    "time": 45.12387681007385,
    "answer": "Segments with mixed expert opinions indicate that the pattern is not clearly defined and that there is a lack of clear consensus among the experts. This could be due to a number of factors, including the complexity of the pattern, the heterogeneity of the data, or the presence of multiple underlying conditions.",
    "length": 312,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "N/A",
    "category": "Pattern Analysis",
    "question": "Find segments with mixed expert opinions"
  },
  {
    "success": true,
    "time": 60.2973051071167,
    "answer": "Error: HTTPConnectionPool(host='eegi-ollama', port=11434): Read timed out. (read timeout=60)",
    "length": 92,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "N/A",
    "category": "Pattern Analysis",
    "question": "What patterns have high amplitude and fast activity?"
  },
  {
    "success": true,
    "time": 44.95865535736084,
    "answer": "The passage describes EEG segments with clean signal quality, which means that the signal-to-noise ratio is high and the spectral edge frequency is within the normal range for beta-dominant activity.",
    "length": 199,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "N/A",
    "category": "Pattern Analysis",
    "question": "Show segments with clean signal quality"
  },
  {
    "success": true,
    "time": 47.677278995513916,
    "answer": "Sure, here's the answer to the question:\n\n**Comparison of LPD and GPD patterns:**\n\n| Feature | LPD | GPD |\n|---|---|---|\n| Appearance | Lateralized | Bilateral |\n| Signal-to-noise ratio | High | Moderate |\n| Spectral edge frequency | 30.47 Hz | Not specified |\n| Amplitude | 1409.46 | Not specified |\n| Interpretation | Typical presentation with some variability | Bridge between LPD and RPP patterns |",
    "length": 402,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "N/A",
    "category": "Comparisons",
    "question": "Compare LPD vs GPD patterns"
  },
  {
    "success": true,
    "time": 60.23468279838562,
    "answer": "Error: HTTPConnectionPool(host='eegi-ollama', port=11434): Read timed out. (read timeout=60)",
    "length": 92,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "N/A",
    "category": "Comparisons",
    "question": "Compare seizure in EEG 1001717358 vs 1002197945"
  },
  {
    "success": true,
    "time": 43.30169177055359,
    "answer": "The difference between high and low confidence classifications indicates the level of agreement among the experts. A high confidence classification indicates that the experts agree on the classification, while a low confidence classification indicates that there is disagreement among the experts.",
    "length": 297,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "N/A",
    "category": "Comparisons",
    "question": "Difference between high and low confidence classifications?"
  },
  {
    "success": true,
    "time": 44.3394501209259,
    "answer": "The context does not provide any information about patient 999999999, so I cannot answer this question from the provided context.",
    "length": 129,
    "eeg_count": 0,
    "medical_count": 2,
    "confidence": "N/A",
    "category": "Edge Cases",
    "question": "Tell me about patient 999999999"
  },
  {
    "success": true,
    "time": 51.286739110946655,
    "answer": "The context does not provide any information about the XYZ pattern, so I cannot answer this question from the provided context.",
    "length": 127,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "N/A",
    "category": "Edge Cases",
    "question": "What is XYZ pattern?"
  },
  {
    "success": true,
    "time": 45.69014501571655,
    "answer": "No, LPD and low-power discharge are not the same thing. LPD is classified as a periodic lateralized epileptic form, while low-power discharge is classified as a generalized periodic discharge.",
    "length": 192,
    "eeg_count": 5,
    "medical_count": 2,
    "confidence": "N/A",
    "category": "Edge Cases",
    "question": "Is LPD the same as low-power discharge?"
  }
]"""