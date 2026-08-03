# ingestion/eeg_pipeline/ingest_knowledge_base_adaptive.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

from eeg_pipeline.embeddings import EmbeddingService
from eeg_pipeline.chroma import ChromaRetriever


class MedicalKnowledgeIndexer:
    def __init__(self, collection_name="medical_definitions_multisize", chroma_host="localhost"):
        self.chroma_host = chroma_host
        self.collection_name = collection_name
        self.chrome_service = ChromaRetriever(self.collection_name, self.chroma_host)
        self.embedding_service = EmbeddingService()

    def index_pdf(self, filepath, batch_size=32):
        """Multi-size chunking: small + large için 2 strateji"""

        loader = PyPDFLoader(filepath)
        raw_docs = loader.load()

        # Check if already indexed
        if self.chrome_service.collection.count() > 0:
            print(f"Collection already has {self.chrome_service.collection.count()} docs. Skipping.")
            return

        # 2 strateji: küçük (definition) + büyük (comparison)
        strategies = {
            'small': {'size': 800, 'overlap': 150},
            'large': {'size': 1800, 'overlap': 300}
        }

        all_chunks = []
        for strategy_name, params in strategies.items():
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=params['size'],
                chunk_overlap=params['overlap'],
                length_function=len
            )
            chunks = splitter.split_documents(raw_docs)

            for idx, doc in enumerate(chunks):
                doc.metadata['strategy'] = strategy_name
                doc.metadata['chunk_id'] = f"{strategy_name}_{idx}"
                all_chunks.append(doc)

        print(
            f"Created {len(all_chunks)} chunks (small: {sum(1 for c in all_chunks if c.metadata['strategy'] == 'small')}, large: {sum(1 for c in all_chunks if c.metadata['strategy'] == 'large')})")

        # Batch indexing
        for i in tqdm(range(0, len(all_chunks), batch_size)):
            batch = all_chunks[i:i + batch_size]
            texts = [doc.page_content for doc in batch]
            embeddings = self.embedding_service.embed(texts)
            ids = [doc.metadata['chunk_id'] for doc in batch]
            metadatas = [{"source": "ACNS_Terminology_2021.pdf",
                          "page": doc.metadata.get("page", -1),
                          "strategy": doc.metadata['strategy']}
                         for doc in batch]

            self.chrome_service.index_segment(document=texts, embedding=embeddings, metadata=metadatas, doc_id=)

        print(f"Indexed {self.chrome_service.collection.count()} total chunks")


if __name__ == "__main__":
    indexer = MedicalKnowledgeIndexer()
    indexer.index_pdf(filepath="ingestion/data/ACNSStandardizedCriticalCareEEGTerminology_rev2021.pdf")