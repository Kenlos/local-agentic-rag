# retrieval/hybrid.py
from config import RRF_K, DENSE_WEIGHT, SPARSE_WEIGHT, TOP_K_FINAL

def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
) -> list[dict]:
    """
    Fuses dense and sparse results using weighted RRF.
    Each doc scores: weight * (1 / (k + rank))
    Documents appearing in both lists accumulate scores.
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for rank, doc in enumerate(dense_results):
        key = doc["content"]
        scores[key] = scores.get(key, 0) + DENSE_WEIGHT * (1.0 / (RRF_K + rank))
        docs[key] = doc

    for rank, doc in enumerate(sparse_results):
        key = doc["content"]
        scores[key] = scores.get(key, 0) + SPARSE_WEIGHT * (1.0 / (RRF_K + rank))
        docs[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for key, score in ranked[:TOP_K_FINAL]:
        doc = docs[key].copy()
        doc["rrf_score"] = round(score, 6)
        results.append(doc)

    return results