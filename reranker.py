# reranker.py
from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL, TOP_K_FINAL

_reranker = None

def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker

def rerank(query: str, chunks: list[dict]) -> list[dict]:
    """
    Re-scores chunks against the query using a cross-encoder.
    Returns TOP_K_FINAL chunks sorted by reranker score.
    """
    if not chunks:
        return []

    reranker = get_reranker()
    pairs = [(query, chunk["content"]) for chunk in chunks]
    scores = reranker.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:TOP_K_FINAL]