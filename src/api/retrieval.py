"""
Handles hybrid retrieval (dense + sparse), fusion and reranking
"""
import logging
from typing import List, Tuple, Optional, Dict
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from src.api.hybridizer import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

class HybridRetriever:

    def __init__(self,
                 embed_model: SentenceTransformer,
                 rerank_model: CrossEncoder,
                 signal_collection: Optional[chromadb.Collection],
                 document_collection: Optional[chromadb.Collection],
                 transcript_collection: Optional[chromadb.Collection],
                 signal_bm25: Optional[BM25Okapi],
                 document_bm25: Optional[BM25Okapi],
                 transcript_bm25: Optional[BM25Okapi],
                 signal_all_docs: List[str],
                 document_all_docs: List[str],
                 transcript_all_docs: List[str],
                 rrf_k: int):

        self.embed_model = embed_model
        self.rerank_model = rerank_model
        self.rrf_k = rrf_k

        # DB Collections
        self.signal_collection = signal_collection
        self.document_collection = document_collection
        self.transcript_collection = transcript_collection

        # Indices
        self.signal_bm25 = signal_bm25
        self.document_bm25 = document_bm25
        self.transcript_bm25 = transcript_bm25

        # BM25 doc lists (needed to map indices to text)
        self.signal_all_docs = signal_all_docs
        self.document_all_docs = document_all_docs
        self.transcript_all_docs = transcript_all_docs

    def _dense_retrieval(self,
                        query_embedding: List[float],
                        n_results: int,
                        where_filter: Optional[dict] = None) -> Tuple[List[Tuple[str, float]],
                                                                    List[Tuple[str, float]],
                                                                    List[Tuple[str, float]]]:
        """
        Runs vector search on all collections
        """
        signal_dense = []
        if self.signal_collection:
            signal_dense_results = self.signal_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=["documents"])

            # ChromaDB's 'distance' score cannot be compared to BM25, so we send 0.0
            signal_dense = [(doc, 0.0) for doc in signal_dense_results.get("documents", [[]])[0]]

        document_dense = []
        if self.document_collection:
            document_dense_results = self.document_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents"])

            document_dense = [(doc, 0.0) for doc in document_dense_results.get("documents", [[]])[0]]

        transcript_dense = []
        if self.transcript_collection:
            transcript_dense_results = self.transcript_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents"])

            transcript_dense = [(doc, 0.0) for doc in transcript_dense_results.get("documents", [[]])[0]]

        return signal_dense, document_dense, transcript_dense

    def _sparse_retrieval(self,
                          query_tokenized: List[str],
                          n_results: int) -> Tuple[List[Tuple[str, float]],
                                                    List[Tuple[str, float]],
                                                    List[Tuple[str, float]]]:
        """Runs BM25 keyword search on all collections"""
        signal_sparse = []
        if self.signal_bm25 and self.signal_all_docs:
            signal_sparse_scores = self.signal_bm25.get_scores(query_tokenized)
            signal_sparse_indices = (-signal_sparse_scores).argsort()[:n_results]
            signal_sparse = [(self.signal_all_docs[i], signal_sparse_scores[i]) for i in signal_sparse_indices]

        document_sparse = []
        if self.document_bm25 and self.document_all_docs:
            document_sparse_scores = self.document_bm25.get_scores(query_tokenized)
            document_sparse_indices = (-document_sparse_scores).argsort()[:n_results]
            document_sparse = [(self.document_all_docs[i], document_sparse_scores[i]) for i in
                               document_sparse_indices]

        transcript_sparse = []
        if self.transcript_bm25 and self.transcript_all_docs:
            transcript_sparse_scores = self.transcript_bm25.get_scores(query_tokenized)
            transcript_sparse_indices = (-transcript_sparse_scores).argsort()[:n_results]
            transcript_sparse = [(self.transcript_all_docs[i], transcript_sparse_scores[i]) for i in
                                 transcript_sparse_indices]

        return signal_sparse, document_sparse, transcript_sparse

    def _rerank_docs(self, query: str, documents: List[str], final_count: int) -> List[str]:
        """
        Rerank docs using cross-encoder model
        """
        if not documents or final_count == 0:
            return []

        sentence_pairs = [[query, doc] for doc in documents]
        scores = self.rerank_model.predict(sentence_pairs, show_progress_bar=False)
        doc_scores = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

        return [doc for doc, score in doc_scores][:final_count]

    def search(self,
                 query: str,
                 n_search: int,
                 n_final_signal: int,
                 n_final_document: int,
                 n_final_transcript: int,
                 where_filter: dict | None = None) -> dict:
        """Run the full hybrid search"""
        query_embed = self.embed_model.encode(query, normalize_embeddings=True).tolist()

        query_tokenized = query.lower().split(" ") # FIXME: ".split()" tokenizer is bad

        # Retrieve
        signal_dense, document_dense, transcript_dense = self._dense_retrieval(query_embed, n_search, where_filter)
        signal_sparse, document_sparse, transcript_sparse = self._sparse_retrieval(query_tokenized, n_search)

        # Hybrid fusion (RRF)
        signal_fused = reciprocal_rank_fusion(signal_dense, signal_sparse, k=self.rrf_k)[:n_final_signal]
        document_fused = reciprocal_rank_fusion(document_dense, document_sparse, k=self.rrf_k)[:n_final_document]
        transcript_fused = reciprocal_rank_fusion(transcript_dense, transcript_sparse, k=self.rrf_k)[:n_final_transcript]

        # Rerank
        signal_docs = self._rerank_docs(query, signal_fused, n_final_signal)
        document_docs = self._rerank_docs(query, document_fused, n_final_document)
        transcript_docs = self._rerank_docs(query, transcript_fused, n_final_transcript)

        logger.info(f"Hybrid retrieval results: S={len(signal_docs)}, D={len(document_docs)}, T={len(transcript_docs)}")
        return {
            "signal": signal_docs,
            "document": document_docs,
            "transcript": transcript_docs}