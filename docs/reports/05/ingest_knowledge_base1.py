#ingestion/eeg_pipeline/ingest_knowledge_base.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

from eeg_pipeline.embeddings import EmbeddingService
from eeg_pipeline.chroma import ChromaRetriever

class MedicalKnowledgeIndexer:
    def __init__(self, collection_name="medical_definitions", chroma_host="localhost"):
        self.chroma_host = chroma_host
        self.collection_name = collection_name
        self.chrome_service = ChromaRetriever(self.collection_name, self.chroma_host)
        self.embedding_service = EmbeddingService()

    def index_pdf(self, filepath, chunk_size=1000, chunk_overlap=200, batch_size=32):
        """Index a medical PDF into ChromaDB"""

        loader = PyPDFLoader(filepath)
        raw_docs = loader.load()

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len)

        splitted_docs = text_splitter.split_documents(raw_docs)

        # Check if already indexed
        if self.chrome_service.collection.count() > 0:
            print(f"Collection '{self.collection_name}' already has {self.chrome_service.collection.count()} documents. Skipping.")
            return

        # Batch processing
        print(f"Indexing {len(splitted_docs)} chunks in batches of {batch_size}...")
        for i in tqdm(range(0, len(splitted_docs), batch_size)):
            batch_chunks = splitted_docs[i:i + batch_size]

            batch_texts = [doc.page_content for doc in batch_chunks]
            batch_embeddings = self.embedding_service.embed(batch_texts)
            batch_ids = [f"acns_def_{j}" for j in range(i, i + len(batch_chunks))]
            batch_metadatas = [{"source": "ACNS_Terminology_2021.pdf","page": doc.metadata.get("page", -1),"chunk_index": j}
                for j, doc in enumerate(batch_chunks, start=i)]

            self.chrome_service.index_segment(document=batch_texts, embedding=batch_embeddings,
                                              metadata=batch_metadatas, doc_id=)

        print(f"Indexed {self.chrome_service.collection.count()} chunks to '{self.collection_name}'")


if __name__ == "__main__":
    indexer = MedicalKnowledgeIndexer()
    indexer.index_pdf(filepath="ingestion/data/ACNSStandardizedCriticalCareEEGTerminology_rev2021.pdf")