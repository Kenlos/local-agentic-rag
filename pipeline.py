# pipeline.py
from ingestion.web import ingest_url
from ingestion.local import ingest_file
from retrieval.embedder import embed
from retrieval.vector_store import insert_chunks, dense_search
from retrieval.bm25 import build_index, sparse_search
from retrieval.hybrid import reciprocal_rank_fusion
from reranker import rerank
from query_expansion import expand_query
from generator import generate

def ingest(source: str) -> dict:
    """
    Ingests a URL or local file path into the knowledge base.
    Automatically detects source type.
    """
    if source.startswith("http://") or source.startswith("https://"):
        chunks = ingest_url(source)
    else:
        chunks = ingest_file(source)

    embeddings = embed([c["content"] for c in chunks])
    insert_chunks(chunks, embeddings)
    build_index(chunks)

    return {"ingested": len(chunks), "source": source}

def query(question: str) -> dict:
    """
    Runs the full RAG pipeline for a question.
    Returns answer, sources, and retrieved chunks.
    """
    # 1 — expand query into multiple variants
    queries = expand_query(question)

    # 2 — retrieve candidates for each variant, merge results
    all_dense, all_sparse = [], []
    for q in queries:
        q_embedding = embed([q])[0]
        all_dense.extend(dense_search(q_embedding))
        all_sparse.extend(sparse_search(q))

    # 3 — deduplicate by content before fusion
    seen = set()
    dense_deduped, sparse_deduped = [], []
    for doc in all_dense:
        if doc["content"] not in seen:
            dense_deduped.append(doc)
            seen.add(doc["content"])
    for doc in all_sparse:
        if doc["content"] not in seen:
            sparse_deduped.append(doc)
            seen.add(doc["content"])

    # 4 — fuse with RRF
    fused = reciprocal_rank_fusion(dense_deduped, sparse_deduped)

    # 5 — rerank
    reranked = rerank(question, fused)

    # 6 — generate
    result = generate(question, reranked)
    result["chunks"] = reranked

    return result