# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- LM Studio ---
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_API_KEY = "lm-studio"
GENERATION_MODEL = os.getenv("GENERATION_MODEL")
THINKING_MODE = os.getenv("THINKING_MODE", "false").lower() == "true"

# --- Embeddings ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
EMBEDDING_DIMENSION = 768

# --- Database ---
DB_ENV = os.getenv("DB_ENV", "local")  # "local" or "remote"

if DB_ENV == "remote":
    DB_URL = os.getenv("REMOTE_DB_URL")
else:
    DB_URL = os.getenv("LOCAL_DB_URL")

# --- Retrieval ---
TOP_K_DENSE = 20
TOP_K_SPARSE = 20
TOP_K_FINAL = 5
RRF_K = 60
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3

# --- Chunking ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# --- Reranker ---
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")