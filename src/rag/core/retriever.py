"""
ChromaDB connection helper. Used for indexing, querying, and clearing the collection.
"""

import chromadb
import logging
from typing import Any, Optional

class ChromaDB:

    def __init__(self, logger: logging.Logger, collection_name: str, host: str, port: int):
        """
        Starts the ChromaDB client and the collection.
        """
        self.logger = logger
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.collection: Optional[chromadb.Collection] = None
        try:
            self.client = chromadb.HttpClient(host=self.host, port=self.port)
            self.collection = self.client.get_or_create_collection(name=self.collection_name,metadata={"hnsw:space": "cosine"})
            self.logger.info(f"Chroma connected: {self.host}:{self.port}, collection='{self.collection_name}'")
        except Exception as e:
            self.logger.critical(f"ChromaDB connection failed ({self.host}:{self.port}). Collection: '{self.collection_name}'. Error: {e}")
            raise ConnectionError(f"ChromaDB connection failed for collection '{self.collection_name}'") from e

    def is_doc_indexed(self, document_id):
        """Checks if a single document ID already exists in the collection."""
        if not self.collection:
            self.logger.error("Cannot check index status: ChromaDB collection is not initialized.")
            raise ConnectionError("ChromaDB collection is not available.")
        try:
            results = self.collection.get(ids=[document_id], limit=1, include=[])
            return len(results.get("ids", [])) > 0
        except Exception as e:
            self.logger.error(f"DB check failed for doc_id '{document_id}'. Error: {e}")
            raise

    def index_segment(self, document: Any, embedding: Any, metadata: Any, doc_id: Any):
        """
        Indexes a document (or batch) to ChromaDB.
        """
        if embedding and isinstance(embedding, list) and isinstance(embedding[0], list):
            self.collection.add(
                documents=document,
                embeddings=embedding,
                ids=doc_id,
                metadatas=metadata)
        else:
            self.collection.add(
                documents=[document],
                embeddings=[embedding],
                ids=[doc_id],
                metadatas=[metadata])

    def get_collection_count(self) -> int:
        """Returns the total number of items in the collection."""
        try:
            return self.collection.count()
        except Exception as e:
            self.logger.error(f"DB count failed for collection '{self.collection_name}'. Error: {e}")
            raise

    def clear_collection(self):
        """Deletes all items from the collection."""
        self.logger.warning(f"Deleting all items from collection: '{self.collection_name}'...")
        try:
            # Deleting with an empty filter deletes all items
            self.collection.delete(where={})
            self.logger.info(f"Collection '{self.collection_name}' is empty.")
        except Exception as e:
            self.logger.error(f"Failed to clear collection '{self.collection_name}': {e}", exc_info=True)
            raise