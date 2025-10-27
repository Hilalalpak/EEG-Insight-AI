-----

# EEG Insight AI: A RAG System for Clinical EEG Analysis (In Development)

## Overview

**EEG Insight AI** is an advanced Retrieval-Augmented Generation (RAG) system we are actively developing to support neurological pattern recognition and seizure risk assessment in clinical contexts. Our goal is to blend classic signal processing with medical knowledge retrieval and Large Language Model (LLM) reasoning to analyze patient **EEG data**.

We are currently processing multi-patient EEG recordings (sampled at $200\,\text{Hz}$) from the **HMS Harmful Brain Activity Classification dataset**. We've indexed three crucial knowledge sources into our ChromaDB vector store:

1.  **EEG Signal Segments:** $1$-second windows of EEG data, complete with extracted features (**power**, **SNR**, **spectral edge frequency**) and aggregated expert consensus labels.
2.  **Medical Terminology:** The official **ACNS 2021 standardized critical care EEG terminology**, processed from a PDF.
3.  **Expert Lectures:** Transcribed YouTube videos covering practical clinical interpretation patterns and reasoning.

The system uses a sophisticated **hybrid retrieval strategy** (dense vector search, BM25 keyword search, and reranking) to address complex clinical queries. For instance, we can answer questions like *"Show seizure segments with high amplitude"* by retrieving raw signal data, or *"What differentiates LPD from LRDA?"* by combining formal definitions with expert clinical discussion.

-----

## Architecture

Our system is centered around a **FastAPI** service that manages the flow from a user query to the final clinical answer.

```
User Query → FastAPI → Hybrid Retriever (SapBERT + BM25 + BGE Reranker)
                    ↓
              ChromaDB (3 collections)
                    ↓
              Context Assembly → Ollama LLM → Clinical Answer
```

### Key Components and Design Decisions

  * **Signal Processing:** We use a **Bandpass filter ($0.5-40\,\text{Hz}$) and a $60\,\text{Hz}$ notch filter** to clean the raw EEG data before feature extraction.
  * **Semantic Chunking:** We built a **custom regex-based parser** to logically structure the ACNS document chunks, preserving section titles and ensuring semantic integrity.
  * **Metadata Enrichment:** To deal with the complexity of the dataset, we **aggregate votes** from overlapping $10$-second expert annotation windows to assign a consensus label to each $1$-second segment.
  * **RRF Fusion:** We implement **Reciprocal Rank Fusion (with $k=60$)** to effectively merge the results from our dense (semantic) and sparse (keyword) retrievers.
  * **Query Classification:** The system automatically analyzes the user's intent—is it asking for **Reasoning** (expert transcripts), a **Definition** (ACNS documents), or **Patient Data** (signal segments)? This decision guides the retrieval process to prioritize the most relevant knowledge source.

-----

## Project Structure

We maintain a standard structure to separate configuration, core logic, and deployment assets.

```
├── infrastructure/
│   ├── conf/                    # All YAML configurations (ChromaDB, RAG strategy, LLM settings)
│   └── docker/                  # Dockerfiles for each service container
├── src/
│   ├── api/                    # FastAPI application and main retrieval logic
│   ├── data_pipeline/          # Scripts for data download (Kaggle) and upload (MinIO)
│   └── rag/
│       ├── core/               # Utilities like the Embedder and ChromaDB wrapper
│       └── ingestion/          # The three distinct data ingestion pipelines
└── ui/                         # Streamlit user interface code
```

-----

## Setup

### Prerequisites

  * Docker & Docker Compose
  * A system with **$16\,\text{GB}+\,\text{RAM}$** (recommended for running the embedding models)
  * Your **Kaggle API credentials** set up for data access.

### Installation

1.  Clone the repository and move into the project directory:

<!-- end list -->

```bash
git clone https://github.com/yourusername/EEGInsightAI.git
cd EEGInsightAI
```

2.  Configure your local environment by creating and editing the `.env` file:

> **NOTE:** Structural variables (Bucket Name, S3 and Ollama URLs) are now managed in the YAML files (`base.yml`, `env_dev.yaml`). Only sensitive and runtime flags remain in `.env`.

```bash
# MinIO (S3-compatible storage) Access Credentials (Sensitive)
S3_ACCESS_KEY_ID=eegi_admin
S3_SECRET_ACCESS_KEY=admin_eegi

# Runtime Flag: Set to true to reset ChromaDB collections if you need to re-index
OVERWRITE_COLLECTION=false
```

3.  Download the EEG data (requires the Kaggle API to be configured):

<!-- end list -->

```bash
pip install kaggle python-dotenv boto3 pandas pyarrow
# Place your kaggle.json file in ~/.kaggle/
python src/data_pipeline/data_pipeline.py
```

4.  Start the core services (Database, Storage, LLM Server):

<!-- end list -->

```bash
docker-compose up -d chromadb minio ollama
# Wait about 30 seconds for the services to stabilize

# Pull the base LLM model using the container's shell
docker exec -it eegi-ollama ollama pull gemma:2b

# Run the ingestion pipelines to populate ChromaDB
docker-compose up ingest-document ingest-transcript ingest-signal

# Start the API and the User Interface
docker-compose up -d api ui
```

You can now access the system at:

  * **UI (Streamlit)**: http://localhost:8501
  * **API (Docs)**: http://localhost:8001/docs
  * **MinIO Console**: http://localhost:9001

-----

## Usage

### Streamlit UI

Use the interface to ask natural language questions:

  * "Show EEG segments from patient 1002136740 labeled as seizure"
  * "What are the diagnostic criteria for GPD?"
  * "How do experts differentiate between LRDA and seizure activity?"

Filters for label, confidence, and result count can be adjusted in the sidebar.

### API Example

You can interact directly with the FastAPI endpoint:

```python
import requests

response = requests.post("http://localhost:8001/query", json={
    "query": "Explain spectral edge frequency in seizure detection",
    "n_results": 5,
    "n_definitions": 3,
    "n_videos": 2,
    "filters": {"expert_consensus": {"$eq": "Seizure"}}
})

print(response.json()["llm_response"])
```

**Available Filter Options**:

  * `expert_consensus`: Seizure, LPD, GPD, LRDA, GRDA, Other
  * `is_high_confidence`: Boolean (confidence $\ge 0.8$)
  * `has_high_amplitude`: Boolean (power $> 20$)
  * `mean_sef`: Float range (e.g., `{"$gte": 15}` for beta-dominant activity)

-----

## Configuration

### RAG Parameters (`infrastructure/conf/rag_strategy.yaml`)

We manage our retrieval strategy settings here:

```yaml
rag:
  n_search: 15           # Initial pool size for retrieval
  n_final_signal: 5      # Top signal segments sent to the LLM
  n_final_document: 2    # Top definition chunks sent to the LLM
  n_final_transcript: 2  # Top video transcript chunks sent to the LLM
  rrf_k: 60              # RRF constant used for fusion
```

### Core Models (`infrastructure/conf/llm/models.yml`)

  * **Embedding:** **SapBERT** (chosen for its strong performance on biomedical vocabulary)
  * **Reranker:** **BGE-reranker-base**
  * **LLM:** **Gemma 2B** (served via Ollama)

You can swap out models by updating `models.yml` and restarting the relevant containers.

-----

## Data Pipeline Details

### EEG Processing

For every $1$-second segment, the pipeline performs:

1.  **Filtering:** Notch ($\mathbf{60\,\text{Hz}}$) followed by Bandpass ($\mathbf{0.5-40\,\text{Hz}}$) using the zero-phase `filtfilt` method.
2.  **Feature Extraction:** Calculation of per-channel power, SNR, and **SEF-$95$**.
3.  **Label Aggregation:** A consensus label is determined by combining votes from overlapping $10$-second annotation windows.
4.  **Summarization:** A compact clinical narrative is generated to embed the signal's core features along with its context.

The resulting metadata includes over $30$ fields detailing the signal characteristics and labeling confidence.

-----

## Known Limitations (Current Development Focus)

1.  **Small Dataset:** We are currently using only $10$ EEG recordings. We plan to expand this by modifying `target_files` in `data_pipeline.py`.
2.  **Tokenization:** We use a simple `.split()` method for BM25. A proper medical-domain tokenizer needs to be implemented.
3.  **Context Window:** The LLM context is currently limited to about $\mathbf{800\,\text{chars}}$ per source, which may constrain reasoning.
4.  **Performance:** The **Gemma 2B** model has limited reasoning capacity for complex cases. We are evaluating an upgrade to a larger LLM.
5.  **Label Overlap:** Ambiguity can occur in low-confidence segments due to the multiple overlapping annotation windows.

-----

## Troubleshooting

**If you encounter ChromaDB connection errors**:

```bash
# Check if the service is running
docker ps | grep chroma
# View the service logs
docker logs eegi-chroma
```

**If the embedding model download hangs**:

```bash
# Pre-download models outside of the Docker container to debug
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')"
```

**If you get empty retrieval results**:

  * Verify that collections are populated by checking the API `/health` endpoint.
  * Check the ChromaDB volume data: `docker exec -it eegi-chroma ls /data`
  * Re-run the ingestion process with `OVERWRITE_COLLECTION=true` in your `.env` file.

-----

## Future Work

We are prioritizing the following features:

  * [ ] Integrate time-series anomaly detection (e.g., isolation forests) for novel pattern identification.
  * [ ] Implement **cross-patient similarity search** to find comparable historical cases.
  * [ ] Support **EEG-to-EEG comparison queries** using signal embeddings.
  * [ ] Fine-tune the reranker on actual medical Q\&A pairs for better relevance scoring.
  * [ ] Add confidence calibration to the LLM's final outputs.
  * [ ] Explore integration for **real-time streaming EEG input**.

-----

## License

This project utilizes data from the [HMS - Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification) Kaggle competition. Please adhere to the competition's data usage terms.

## Acknowledgments

We thank the following for making this project possible:

  * The **ACNS** for providing standardized EEG terminology (2021 revision).
  * **Dr. Lawrence Hirsch** for his invaluable educational video series on critical care EEG.
  * **Kaggle** and **HMS** for providing the foundational dataset.

-----

*Built with: FastAPI, ChromaDB, SentenceTransformers, Ollama, Streamlit, Docker* 🛠️