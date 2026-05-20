# retrieval/vector_store.py
import psycopg2
import psycopg2.extras
from config import DB_URL, TOP_K_DENSE

def get_conn():
    return psycopg2.connect(DB_URL)

def insert_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None:
    """Writes chunks + their embeddings to the documents table."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            for chunk, embedding in zip(chunks, embeddings):
                cur.execute(
                    """
                    INSERT INTO documents (source, content, embedding, metadata)
                    VALUES (%s, %s, %s::vector, %s)
                    """,
                    (
                        chunk["source"],
                        chunk["content"],
                        embedding,
                        psycopg2.extras.Json({
                            "chunk_index": chunk["chunk_index"]
                        }),
                    )
                )
        conn.commit()
    finally:
        conn.close()

def dense_search(query_embedding: list[float], top_k: int = TOP_K_DENSE) -> list[dict]:
    """Returns top_k chunks by cosine similarity to the query embedding."""
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, source, content, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, top_k)
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()