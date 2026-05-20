# retrieval/embedder.py
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

_model = None

def get_model() -> SentenceTransformer:
    """Lazy-loads the embedding model once and reuses it."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    return _model

def embed(texts: list[str]) -> list[list[float]]:
    """Embeds a list of strings. Returns a list of vectors."""
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True  # required for cosine similarity
    )
    return embeddings.tolist()