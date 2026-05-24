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
HARDWARE_TIER = os.getenv("HARDWARE_TIER", "large").lower()

# Hardware tiers tune the pipeline to your available memory and model size.
# Each tier adjusts chunk size, retrieval depth, and query expansion count
# to balance quality against speed and memory usage.

TIER_CONFIGS = {
    "small": {
        # 8-16GB RAM, 7B model (Mistral 7B, Llama 3.1 8B, Qwen 2.5 7B)
        # Conservative settings to avoid OOM and keep latency under 30s.
        "chunk_size": 256,
        "chunk_overlap": 32,
        "top_k_dense": 10,
        "top_k_sparse": 10,
        "top_k_final": 3,
        "rrf_k": 60,
        "dense_weight": 0.7,
        "sparse_weight": 0.3,
        "max_query_variants": 2,   # query expansion: 2 variants instead of 3
        "description": "7B models, 8-16GB RAM"
    },
    "medium": {
        # 16-32GB RAM, 13B model (Llama 3.1 13B, Qwen 2.5 14B)
        # Balanced settings — good quality without maxing out memory.
        "chunk_size": 384,
        "chunk_overlap": 48,
        "top_k_dense": 15,
        "top_k_sparse": 15,
        "top_k_final": 4,
        "rrf_k": 60,
        "dense_weight": 0.7,
        "sparse_weight": 0.3,
        "max_query_variants": 3,
        "description": "13B models, 16-32GB RAM"
    },
    "large": {
        # 32GB+ RAM, 30B+ model (Qwen3 35B, Llama 3.1 70B Q4)
        # Full quality settings — your current setup.
        "chunk_size": 512,
        "chunk_overlap": 64,
        "top_k_dense": 20,
        "top_k_sparse": 20,
        "top_k_final": 5,
        "rrf_k": 60,
        "dense_weight": 0.7,
        "sparse_weight": 0.3,
        "max_query_variants": 4,
        "description": "30B+ models, 32GB+ RAM"
    }
}

# load the selected tier
_tier = TIER_CONFIGS.get(HARDWARE_TIER, TIER_CONFIGS["large"])

# expose as module-level constants so the rest of the codebase
# imports them exactly as before — no other files need to change
CHUNK_SIZE = _tier["chunk_size"]
CHUNK_OVERLAP = _tier["chunk_overlap"]
TOP_K_DENSE = _tier["top_k_dense"]
TOP_K_SPARSE = _tier["top_k_sparse"]
TOP_K_FINAL = _tier["top_k_final"]
RRF_K = _tier["rrf_k"]
DENSE_WEIGHT = _tier["dense_weight"]
SPARSE_WEIGHT = _tier["sparse_weight"]
MAX_QUERY_VARIANTS = _tier["max_query_variants"]

print(f"[config] hardware tier: {HARDWARE_TIER} ({_tier['description']})")
print(f"[config] chunk size: {CHUNK_SIZE} | top_k: {TOP_K_DENSE} dense / {TOP_K_SPARSE} sparse → {TOP_K_FINAL} final")

# --- Reranker ---
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")