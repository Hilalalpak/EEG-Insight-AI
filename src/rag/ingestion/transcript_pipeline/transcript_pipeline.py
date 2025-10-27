"""
Pipeline to fetch, chunk and index YouTube video transcripts
"""
import os
import logging
from typing import Any
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tqdm import tqdm

from infrastructure.conf.interfaces import (DBConfigInterface, LLMConfigInterface, PipelineConfigInterface)

from src.rag.core.embedder import TextEmbedder
from src.rag.core.retriever import ChromaDB
from src.rag.ingestion.transcript_pipeline.metadata_ext import MetadataExtractor

class TranscriptPipeline:

    def __init__(self,
                 db_config: DBConfigInterface,
                 llm_config: LLMConfigInterface,
                 pipeline_config: PipelineConfigInterface,
                 logger: logging.Logger):

        self.logger = logger
        self.pipeline_config = pipeline_config

        self.collection_name = db_config.get_collection_name("transcript")
        self.chroma_host = db_config.get_chroma_host()
        self.chroma_port = db_config.get_chroma_port()

        embed_model_name = llm_config.get_embedding_model_name()

        self.embedder = TextEmbedder(self.logger, embed_model_name)
        self.chroma_db = ChromaDB(self.logger, self.collection_name, self.chroma_host, self.chroma_port)

        self.video_list = pipeline_config.get_youtube_videos()
        self.metadata_extractor = MetadataExtractor()

        self.logger.info(f"TranscriptPipeline initialized. Collection: '{self.collection_name}'")

    def _get_video_transcript(self, url: str) -> list[Any]:
        """Takes transcripts using YoutubeLoader"""
        try:
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=False, language=["en"])
            documents = loader.load()

            if not documents or not documents[0].page_content.strip():
                self.logger.warning(f"No transcript found for: {url}")
                return []

            for doc in documents:
                doc.metadata["video_url"] = url
                doc.metadata["source"] = "youtube"
            return documents

        except Exception as e:
            self.logger.error(f"YoutubeLoader failed for {url}: {e}")
            return []

    def _chunk_transcript(self, documents: list[Any]) -> list[Document]:
        """Splits the transcript documents into overlapping chunks."""
        chunk_params = self.pipeline_config.get_transcript_chunk_params()
        chunk_size = chunk_params.get("chunk_size", 1500)
        chunk_overlap = chunk_params.get("chunk_overlap", 300)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " "]) # try to split on sentences
        chunks = splitter.split_documents(documents)
        self.logger.info(f"Transcript split into {len(chunks)} chunks.")
        return chunks

    def run_pipeline(self):
        self.logger.info(f"Starting transcript pipeline for {len(self.video_list)} videos...")

        try:
            count = self.chroma_db.get_collection_count()
        except Exception as e:
            self.logger.critical(f"DB count check failed: {e}", exc_info=True)
            return

        if count > 0:
            self.logger.warning(f"Collection '{self.collection_name}' already has {count} documents.")
            overwrite_flag = os.getenv("OVERWRITE_COLLECTION", "false").lower()
            if overwrite_flag == "true":
                self.logger.info("Clearing existing collection...")
                self.chroma_db.clear_collection()
            else:
                self.logger.info("Skipping indexing.")
                return

        total_chunks_indexed = 0
        stats = {'reasoning': 0, 'examples': 0, 'definitions': 0, 'risk': 0}

        for video_url in self.video_list:
            self.logger.info(f"Processing video: {video_url}")

            video_id = video_url.split("watch?v=")[-1].split("&")[0]
            first_chunk_key = f"video_{video_id}_chunk_0"

            if self.chroma_db.is_doc_indexed(first_chunk_key):
                self.logger.info(f"Video {video_id} is already indexed. Skipping.")
                continue

            docs = self._get_video_transcript(video_url)
            if not docs:
                self.logger.error(f"Failed to get transcript: {video_id}")
                continue

            chunks = self._chunk_transcript(docs)
            if not chunks:
                self.logger.warning(f"No chunks created: {video_id}")
                continue

            self.logger.info(f"Indexing {len(chunks)} chunks: {video_id}...")

            for idx, chunk in enumerate(tqdm(chunks, desc=f"Indexing {video_id}")):
                try:
                    text = chunk.page_content

                    extracted_meta = self.metadata_extractor.extract_all(text)

                    # Update stats
                    if extracted_meta['has_step_by_step']: stats['reasoning'] += 1
                    if extracted_meta['has_example']: stats['examples'] += 1
                    if extracted_meta['has_definition']: stats['definitions'] += 1
                    if extracted_meta['has_risk']: stats['risk'] += 1

                    embedding = self.embedder.embed(text)

                    metadata = {
                        "source": "video_transcript",
                        "video_id": video_id,
                        "video_url": video_url,
                        "chunk_index": idx,
                        **extracted_meta}

                    self.chroma_db.index_segment(document=text, embedding=embedding, metadata=metadata,
                                                 doc_id=f"video_{video_id}_chunk_{idx}")
                    total_chunks_indexed += 1

                except Exception as e:
                    self.logger.error(f"Chunk failed: {video_id} (chunk {idx}): {e}")

            self.logger.info(f"Finished processing video {video_id}.")

