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
    
def get_ingested_sources() -> list[str]:
    """Returns a list of all unique sources in the knowledge base."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT source FROM documents")
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()
    

def get_source_chunk_count(source: str) -> int:
    """Returns how many chunks exist for a given source."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM documents WHERE source = %s",
                (source,)
            )
            return cur.fetchone()[0]
    finally:
        conn.close()

def source_exists(source: str) -> bool:
    """Returns True if this source has already been ingested."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM documents WHERE source = %s LIMIT 1",
                (source,)
            )
            return cur.fetchone() is not None
    finally:
        conn.close()

def cache_lookup(question: str) -> dict | None:
    """
    Searches for a cached answer to a similar question using
    PostgreSQL full-text search.

    Why full-text search instead of exact match:
    "How does RAG work?" and "How does RAG actually work?" should
    hit the same cache entry. Full-text search handles synonyms,
    stop words, and word order variation automatically.

    Why quality >= 0.7:
    We only cache answers we're confident in. Returning a cached
    low-quality answer is worse than computing a fresh one.
    """
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, question, answer, route, sources, quality, hits
                FROM query_cache
                WHERE to_tsvector('english', question) @@ plainto_tsquery('english', %s)
                AND quality >= 0.7
                ORDER BY quality DESC, created_at DESC
                LIMIT 1
                """,
                (question,)
            )
            row = cur.fetchone()
            if row:
                cur.execute(
                    "UPDATE query_cache SET hits = hits + 1 WHERE id = %s",
                    (row["id"],)
                )
                conn.commit()
                return dict(row)
            return None
    finally:
        conn.close()

def cache_store(question: str, answer: str, route: str,
                sources: list, quality: float) -> None:
    """
    Stores an answer in the cache.

    Only called when the answer is substantive — we check this
    in pipeline.py before calling here.

    ON CONFLICT DO NOTHING prevents duplicate cache entries
    if the same question is asked twice before the first answer
    is cached (race condition on slow machines).
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO query_cache
                    (question, answer, route, sources, quality)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (question, answer, route,
                 psycopg2.extras.Json(sources), quality)
            )
        conn.commit()
    finally:
        conn.close()

def delete_source(source: str) -> int:
    """
    Deletes all chunks for a given source.
    Used by refresh_source() to clear stale content before re-ingesting.
    Returns the number of rows deleted.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM documents WHERE source = %s",
                (source,)
            )
            deleted = cur.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()


def get_source_chunk_count(source: str) -> int:
    """
    Returns how many chunks exist for a given source.
    Used by the dynamic threshold to scale confidence requirements
    based on how much content we have on a topic.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM documents WHERE source = %s",
                (source,)
            )
            return cur.fetchone()[0]
    finally:
        conn.close()