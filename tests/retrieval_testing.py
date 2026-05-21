# run this as a quick script in your terminal
from ingestion.web import ingest_url
from retrieval.embedder import embed
from retrieval.vector_store import insert_chunks, dense_search
from retrieval.bm25 import build_index, sparse_search
from retrieval.hybrid import reciprocal_rank_fusion

# 1. ingest
print("Ingesting...")
chunks = ingest_url("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")

# 2. embed + store
print("Embedding and storing...")
embeddings = embed([c["content"] for c in chunks])
insert_chunks(chunks, embeddings)

# 3. build BM25 index
build_index(chunks)

# 4. query
query = "How does RAG reduce hallucinations?"
query_embedding = embed([query])[0]

dense = dense_search(query_embedding)
sparse = sparse_search(query)
fused = reciprocal_rank_fusion(dense, sparse)

print(f"\nTop {len(fused)} results after RRF fusion:")
for i, doc in enumerate(fused):
    print(f"\n[{i+1}] RRF score: {doc['rrf_score']}")
    print(f"    Source: {doc['source']}")
    print(f"    Preview: {doc['content'][:150]}...")