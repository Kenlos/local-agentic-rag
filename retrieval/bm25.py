# retrieval/bm25.py
import bm25s
import numpy as np
from config import TOP_K_SPARSE

# In-memory index — rebuilt on each ingestion call
# We'll persist this to disk in a later iteration
_index = None
_corpus = []  # list of dicts with 'content', 'source', etc.

def build_index(chunks: list[dict]) -> None:
    """Builds a BM25 index from a list of chunk dicts."""
    global _index, _corpus
    _corpus = chunks
    corpus_tokens = bm25s.tokenize(
        [c["content"] for c in chunks],
        stopwords="en"
    )
    _index = bm25s.BM25()
    _index.index(corpus_tokens)

def sparse_search(query: str, top_k: int = TOP_K_SPARSE) -> list[dict]:
    """Returns top_k chunks by BM25 score."""
    if _index is None or not _corpus:
        return []

    query_tokens = bm25s.tokenize(query, stopwords="en")
    results, scores = _index.retrieve(query_tokens, k=min(top_k, len(_corpus)))

    output = []
    for idx, score in zip(results[0], scores[0]):
        chunk = _corpus[idx].copy()
        chunk["score"] = float(score)
        output.append(chunk)

    return output