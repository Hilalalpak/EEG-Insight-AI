"""
Loads search indices (ChromaDB collections & BM25) at API startup.
"""
import chromadb
import logging
from rank_bm25 import BM25Okapi
from typing import Dict, Tuple, List, Optional
from infrastructure.conf.interfaces import DBConfigInterface

logger = logging.getLogger(__name__)

def load_collections(host: str, port: int, db_config: DBConfigInterface) -> Dict[str, Optional[chromadb.Collection]]:
    """Connect to ChromaDB and load collections."""
    collections = {}
    logger.info(f"Connecting to ChromaDB at {host}:{port}...")

    collection_names_map = {
        "signal": db_config.get_collection_name("signal"),
        "document": db_config.get_collection_name("document"),
        "transcript": db_config.get_collection_name("transcript")}

    try:
        chroma_client = chromadb.HttpClient(host=host, port=port)

        for key, name in collection_names_map.items():
            try:
                collections[key] = chroma_client.get_collection(name=name)
                logger.debug(f"Retrieved collection: '{name}'")
            except Exception:
                logger.error(f"Collection '{name}' not found - {key} searches disabled")
                collections[key] = None

        return collections

    except Exception as e:
        logger.critical(f"ChromaDB connection failed at {host}:{port} - {e}", exc_info=True)
        return {key: None for key in collection_names_map}

def build_bm25_indices(collections: Dict[str, Optional[chromadb.Collection]]) -> Tuple[Dict[str, Optional[BM25Okapi]], Dict[str, List[str]]]:
    """
    Build BM25 indices. Warning: Loads ALL documents into memory.
    """
    bm25_indices = {}
    all_docs = {}

    logger.info("Building BM25 sparse indices...")

    for key, collection in collections.items():
        if collection is None:
            logger.warning(f"Skipping BM25 for {key} - collection missing")
            bm25_indices[key] = None
            all_docs[key] = []
            continue

        try:
            # Fetch documents
            docs_result = collection.get(include=['documents'])['documents'][0]
            if not docs_result:
                logger.warning(f"No documents found in {key} collection")
                bm25_indices[key] = None
                all_docs[key] = []
                continue

            all_docs[key] = docs_result

            # Tokenize and build BM25 index
            tokenized_corpus = [doc.lower().split(" ") for doc in docs_result]
            bm25_indices[key] = BM25Okapi(tokenized_corpus)

            logger.info(f"BM25 ready for {key}: {len(docs_result)} docs")

        except Exception as e:
            logger.error(f"BM25 build failed for {key}: {e}", exc_info=True)
            bm25_indices[key] = None
            all_docs[key] = []

    logger.info("BM25 indexing complete")
    return bm25_indices, all_docs