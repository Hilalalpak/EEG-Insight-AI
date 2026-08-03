# ingestion/eeg_pipeline/ingest_knowledge_base2.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import re
from eeg_pipeline.embeddings import EmbeddingService
from eeg_pipeline.chroma import ChromaRetriever


class MedicalKnowledgeIndexer:
    def __init__(self, collection_name="medical_definitions_semantic", chroma_host="localhost"):

        self.chroma_host = chroma_host
        self.collection_name = collection_name
        self.chrome_service = ChromaRetriever(self.collection_name, self.chroma_host)
        self.embedding_service = EmbeddingService()

    def detect_section_boundaries(self, raw_docs):

        sections = []
        current_section = {"title": "Introduction", "content": "", "pages": []}

        section_patterns = [
            r'^[A-Z]\.\s+[A-Z\s]+$',       # A. EEG BACKGROUND
            r'^[0-9]+\.\s+[A-Z][a-zA-Z\s]+',  # 1. Generalized Periodic Discharges
            r'^[A-Z]{2,}\s*\[NEW',         # BIRDs [NEW, 2021]
            r'^\d+[a-z]?\.\s+[A-Z]',       # 4b. Possible ECSE
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+', # e.g., Clinical Context
        ]

        for doc in raw_docs:
            lines = doc.page_content.split('\n')
            page_num = doc.metadata.get('page', -1)

            for line in lines:
                line = line.strip()

                # Header detection
                is_header = any(re.match(pattern, line) for pattern in section_patterns)
                if is_header and len(line) < 100:
                    if current_section["content"].strip():
                        sections.append(current_section)
                    current_section = {"title": line, "content": "", "pages": [page_num]}
                else:
                    current_section["content"] += line + "\n"
                    if page_num not in current_section["pages"]:
                        current_section["pages"].append(page_num)

        # Add last
        if current_section["content"].strip():
            sections.append(current_section)

        return sections

    def create_hierarchical_chunks(self, sections):

        all_chunks = []
        for section in sections:
            content = section["content"].strip()
            if len(content) < 100:
                continue

            # Adaptive chunking
            if len(content) < 2000:
                chunk_size = len(content)
                chunk_overlap = 0
            else:
                chunk_size = 1200
                chunk_overlap = 300

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", "; ", ", ", " "],
                length_function=len)

            chunks = splitter.split_text(content)

            for i, chunk_text in enumerate(chunks):
                all_chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "section_title": section["title"],
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "pages": ", ".join(map(str, section["pages"])),
                        "source": "ACNS_Terminology_2021.pdf",
                        "is_complete_section": len(chunks) == 1}})

        return all_chunks

    def index_pdf(self, filepath, batch_size=32, use_semantic_chunking=True):

        loader = PyPDFLoader(filepath)
        raw_docs = loader.load()

        # Collection check
        count = self.chrome_service.collection.count()
        if count > 0:
            print(f"⚠️ Collection '{self.collection_name}' already has {count} documents.")
            confirm = input("Delete and re-index? (yes/no): ").strip().lower()
            if confirm == "yes":
                print("🗑️ Deleting existing collection...")
                self.chrome_service.collection.delete()
            else:
                print("⏭️ Skipping indexing.")
                return

        # Semantic vs basic chunking
        if use_semantic_chunking:
            print("🔍 Detecting semantic sections...")
            sections = self.detect_section_boundaries(raw_docs)
            print(f"📘 Found {len(sections)} semantic sections.")
            for sec in sections[:5]:
                print(f"   • {sec['title']} ({len(sec['content'])} chars)")
            splitted_docs = self.create_hierarchical_chunks(sections)
        else:
            print("⚙️ Using fallback: basic chunking (no semantic structure).")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300, length_function=len)
            docs = splitter.split_documents(raw_docs)
            splitted_docs = [{"text": doc.page_content, "metadata": {"page": doc.metadata.get("page", -1)}} for doc in docs]

        # Batch embedding + insertion
        print(f"\n📥 Indexing {len(splitted_docs)} chunks (batch={batch_size})...")
        for i in tqdm(range(0, len(splitted_docs), batch_size)):
            batch = splitted_docs[i:i + batch_size]
            texts = [c["text"] for c in batch]
            embeddings = self.embedding_service.embed(texts)
            ids = [f"acns_chunk_{i+j}" for j in range(len(batch))]
            metadatas = [c["metadata"] for c in batch]

            self.chrome_service.index_segment(document=texts, embedding=embeddings, metadata=metadatas, doc_id=)

        print(f"✅ Indexed {self.chrome_service.collection.count()} chunks to '{self.collection_name}'.")

        complete_sections = sum(1 for c in splitted_docs if c["metadata"].get("is_complete_section"))
        print(f"📈 Stats → {complete_sections} complete sections | {len(splitted_docs)} total chunks.")


if __name__ == "__main__":
    indexer = MedicalKnowledgeIndexer(chroma_host="localhost")
    indexer.index_pdf(filepath="ingestion/data/ACNSStandardizedCriticalCareEEGTerminology_rev2021.pdf", use_semantic_chunking=True)
