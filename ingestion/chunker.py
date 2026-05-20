# ingestion/chunker.py
from config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(text: str, source: str) -> list[dict]:
    """
    Splits text into overlapping chunks.
    Returns list of dicts with 'content', 'source', 'chunk_index'.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + CHUNK_SIZE
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        if len(chunk_text.strip()) > 50:  # skip near-empty chunks
            chunks.append({
                "content": chunk_text,
                "source": source,
                "chunk_index": len(chunks),
            })

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks