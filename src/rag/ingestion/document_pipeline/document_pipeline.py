"""
Pipeline to load, semantically chunk, and index PDF knowledge base docs (e.g. ACNS).
"""
import logging
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from infrastructure.conf.interfaces import (DBConfigInterface, LLMConfigInterface, PipelineConfigInterface)

from src.rag.core.embedder import TextEmbedder
from src.rag.core.retriever import ChromaDB
from src.rag.ingestion.document_pipeline.parser import DocumentParser

class DocumentPipeline:

    def __init__(self,
                 db_config: DBConfigInterface,
                 llm_config: LLMConfigInterface,
                 pipeline_config: PipelineConfigInterface,
                 logger: logging.Logger):

        self.logger = logger
        self.pipeline_config = pipeline_config

        self.collection_name = db_config.get_collection_name("document")
        self.chroma_host = db_config.get_chroma_host()
        self.chroma_port = db_config.get_chroma_port()

        embed_model_name = llm_config.get_embedding_model_name()

        self.chroma_db = ChromaDB(self.logger, self.collection_name, self.chroma_host, self.chroma_port)
        self.embedder = TextEmbedder(self.logger, embed_model_name)

        self.parser = DocumentParser(pipeline_config, logger)


    def index_pdf(self, filepath: str, batch_size: int = 32, use_semantic_chunking: bool = True):
        """Main method to load, chunk and index a PDF file"""
        source_filename = os.path.basename(filepath)
        self.logger.info(f"Starting PDF indexing for: {source_filename}")

        loader = PyPDFLoader(filepath)
        raw_docs = loader.load()

        # Check collection status
        try:
            count = self.chroma_db.get_collection_count()
        except Exception as e:
            self.logger.critical(f"DB count check failed: {e}", exc_info=True)
            return

        if count > 0:
            self.logger.warning(f"Collection '{self.collection_name}' already has {count} documents.")
            overwrite_flag = os.getenv("OVERWRITE_COLLECTION", "false").lower()
            if overwrite_flag == "true":
                self.logger.info("Clearing collection...")
                self.chroma_db.clear_collection()
            else:
                self.logger.info("Skipping indexing.")
                return

        # chunking
        if use_semantic_chunking:
            self.logger.info("Running semantic chunking (regex)...")
            splitted_docs = self.parser.get_chunks(raw_docs, source_filename)
        else:
            self.logger.warning("No semantic chunking. Using basic text split.")
            chunk_params = self.pipeline_config.get_document_chunk_params()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_params.get("chunk_size"),
                chunk_overlap=chunk_params.get("chunk_overlap"),
                length_function=len)

            docs = splitter.split_documents(raw_docs)
            splitted_docs = [{"text": doc.page_content,
                              "metadata": {"page": doc.metadata.get("page", -1),
                                           "source": source_filename}} for doc in docs]

        # Indexing in batches
        self.logger.info(f"Indexing {len(splitted_docs)} chunks in batches of {batch_size}...")
        for i in tqdm(range(0, len(splitted_docs), batch_size), desc="Indexing Batches"):
            try:
                batch = splitted_docs[i:i + batch_size]
                texts = [c["text"] for c in batch]
                metadatas = [c["metadata"] for c in batch]
                ids = [f"acns_chunk_{i + j}" for j in range(len(batch))]

                embeddings = self.embedder.embed(texts)

                self.chroma_db.index_segment(document=texts, embedding=embeddings, metadata=metadatas, doc_id=ids)
            except Exception as e:
                self.logger.error(f"Batch index failed (chunk {i} to {i+batch_size}): {e}")

        final_count = self.chroma_db.get_collection_count()
        self.logger.info(f"Indexing complete: {final_count} docs in collection")

        if use_semantic_chunking:
            complete_sections = sum(1 for c in splitted_docs if c["metadata"].get("is_complete_section"))
            self.logger.info(f"Stats: {complete_sections} complete sections, {len(splitted_docs)} total chunks")
