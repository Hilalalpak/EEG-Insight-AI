"""
Document-specific semantic parsing and chunking logic
"""
import re
import logging
from typing import Any, List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from infrastructure.conf.interfaces import PipelineConfigInterface

class DocumentParser:

    def __init__(self, pipeline_config: PipelineConfigInterface, logger: logging.Logger):
        self.logger = logger
        self.pipeline_config = pipeline_config

        # TODO: This specific and dependent on the ACNS pdf. A more general solution can be considered
        self.section_patterns = [
            r'^[A-Z]\.\s+[A-Z\s]+$',  # A. EEG BACKGROUND
            r'^[0-9]+\.\s+[A-Z][a-zA-Z\s]+',  # 1. Generalized Periodic Discharges
            r'^[A-Z]{2,}\s*\[NEW',  # BIRDs [NEW, 2021]
            r'^\d+[a-z]?\.\s+[A-Z]',  # 4b. Possible ECSE
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+',  # e.g., Clinical Context
        ]

    def _find_sections(self, raw_docs: list[Any]) -> list[dict]:
        """Extracts semantic sections with regex from PDF."""
        sections = []
        current_section = {"title": "Introduction", "content": "", "pages": []}

        for doc in raw_docs:
            lines = doc.page_content.split('\n')
            page_num = doc.metadata.get('page', -1)

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check if the line matches any header pattern
                is_header = any(re.match(pattern, line) for pattern in self.section_patterns)
                if is_header and len(line) < 100:
                    if current_section["content"].strip():
                        sections.append(current_section)
                    current_section = {"title": line, "content": "", "pages": [page_num]}
                else:
                    current_section["content"] += line + "\n"
                    if page_num not in current_section["pages"]:
                        current_section["pages"].append(page_num)

        if current_section["content"].strip():
            sections.append(current_section)

        return sections

    def _build_chunks_from_sections(self, sections: list[dict], source_filename: str) -> list[dict]:
        """The content of each chapter adaptively chunks."""
        all_chunks = []

        chunk_params = self.pipeline_config.get_document_chunk_params()
        small_section_threshold = chunk_params.get("small_section_threshold", 2000)
        default_chunk_size = chunk_params.get("chunk_size", 1200)
        default_chunk_overlap = chunk_params.get("chunk_overlap", 300)

        for section in sections:
            content = section["content"].strip()
            if len(content) < 100:  # Skip short sections
                continue

            if len(content) < small_section_threshold:
                chunk_size = len(content)  # Store as one chunk
                chunk_overlap = 0
            else:
                chunk_size = default_chunk_size
                chunk_overlap = default_chunk_overlap

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
                        "source": source_filename,
                        "is_complete_section": len(chunks) == 1}})

        return all_chunks

    def get_chunks(self, raw_docs: List[Any], source_filename: str) -> List[Dict]:
        """Public method to run the parsing"""
        sections = self._find_sections(raw_docs)
        splitted_docs = self._build_chunks_from_sections(sections, source_filename)
        return splitted_docs