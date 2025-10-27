"""
Implements RRF to combine results from dense and sparse search.
"""
from collections import defaultdict

def reciprocal_rank_fusion(dense_results: list,
                           sparse_results: list,
                           k: int) -> list:
    """
    **Scores are ignored, only rank is used
    """

    fused_scores = defaultdict(float)

    for rank, (document, _) in enumerate(dense_results):
        fused_scores[document] += 1.0 / (k + rank + 1)

    for rank, (document, _) in enumerate(sparse_results):
        fused_scores[document] += 1.0 / (k + rank + 1)

    # Sort by score and rotate documents
    ranked_documents = sorted(fused_scores.keys(), key=lambda doc: fused_scores[doc], reverse=True)

    return ranked_documents