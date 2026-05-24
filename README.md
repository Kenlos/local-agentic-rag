# Local Agentic RAG Pipeline + MCP Server

A production-quality local Retrieval-Augmented Generation (RAG) pipeline with hybrid retrieval, adaptive routing, a ReAct agent, and an MCP server — callable from Claude Desktop, Cursor, or any MCP-compatible client.

Built entirely on local infrastructure. No OpenAI. No LangChain. No cloud dependencies required.

---

## Features

- **Hybrid retrieval** — dense vector search (pgvector) + BM25 sparse retrieval fused with Reciprocal Rank Fusion (RRF)
- **Cross-encoder reranking** — re-scores retrieved chunks for precision before generation
- **Query expansion** — rewrites each query multiple ways to improve recall
- **Adaptive router** — decides per query whether to retrieve from the knowledge base, search the live web, or answer directly
- **Dynamic confidence threshold** — adjusts retrieval confidence requirements based on data quality per source
- **ReAct agent** — manually implemented reasoning loop with `vector_search`, `web_search`, `code_exec`, and `calculator` tools
- **MCP server** — exposes the full pipeline as a Model Context Protocol server callable from any MCP-compatible client
- **Web search** — Tavily integration for full article content retrieval with DuckDuckGo fallback
- **Dual database support** — local PostgreSQL for development, any remote PostgreSQL provider (Supabase, Neon, Railway) for production
- **Duplicate prevention** — source-level deduplication with force refresh for cron job use cases
- **RAGAs evaluation** — benchmarked against a naive dense-search baseline

---

## Benchmarks

Evaluated using RAGAs against a naive dense-search baseline:

| Metric | Naive Baseline | Enhanced Pipeline | Delta |
|---|---|---|---|
| Answer Relevancy | 63.5% | 97.0% | +34% |
| Context Recall | 0.0% | 50.0% | +50% |

> Context recall of 0% on the naive baseline means pure vector search retrieved chunks containing none of the information needed to answer correctly. Hybrid retrieval + reranking fixed this entirely.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Vector store | pgvector + PostgreSQL |
| Sparse retrieval | BM25s |
| Embeddings | nomic-embed-text-v1.5 (sentence-transformers) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Generation | Any OpenAI-compatible local model via LM Studio |
| Web search | Tavily (DuckDuckGo fallback) |
| MCP server | mcp (official Python SDK) |
| Evaluation | RAGAs |
| Web scraping | BeautifulSoup + requests |

---

## Prerequisites

- Python 3.11+
- PostgreSQL 16+ with pgvector extension
- [LM Studio](https://lmstudio.ai) with any OpenAI-compatible local model loaded
- Git

---

## Quick Setup with Docker Compose (Recommended)

git clone https://github.com/your-username/rag-pipeline.git
cd rag-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env — set GENERATION_MODEL and HARDWARE_TIER
python setup.py

---

## Setup

### 1 — Clone the repository

```bash
git clone https://github.com/your-username/rag-pipeline.git
cd rag-pipeline
```

### 2 — Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your values:

```bash
# --- environment ---
DB_ENV=local               # "local" or "remote"

# --- local postgres ---
LOCAL_DB_URL=postgresql://your-user@localhost:5432/rag_db

# --- remote postgres (Supabase, Neon, Railway, etc.) ---
REMOTE_DB_URL=postgresql://user:password@your-host:5432/dbname

# --- LM Studio ---
LM_STUDIO_BASE_URL=http://localhost:1234/v1
GENERATION_MODEL=your-exact-model-name-here
THINKING_MODE=false

# --- embeddings ---
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# --- web search (optional but recommended) ---
TAVILY_API_KEY=tvly-xxxxxxxxxx
```

### 5 — Set up PostgreSQL with pgvector

**Mac (Homebrew):**

```bash
brew install postgresql@16
brew services start postgresql@16

# build pgvector against your PostgreSQL installation
cd /tmp
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make PG_CONFIG=/opt/homebrew/opt/postgresql@16/bin/pg_config
make install PG_CONFIG=/opt/homebrew/opt/postgresql@16/bin/pg_config
```

**Create the database:**

```bash
psql -U your-user postgres
```

```sql
CREATE DATABASE rag_db;
\c rag_db
CREATE EXTENSION vector;

CREATE TABLE documents (
    id          SERIAL PRIMARY KEY,
    source      TEXT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(768),
    metadata    JSONB DEFAULT '{}'
);

CREATE INDEX ON documents
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
```

**Using a remote provider (Supabase, Neon, etc.):**

Run the same SQL above in your provider's SQL editor, then set `DB_ENV=remote` and `REMOTE_DB_URL` in your `.env`.

### 6 — Start LM Studio

1. Open LM Studio and load your model
2. Go to the **Developer** panel
3. Click **Start Server** — it runs at `http://localhost:1234` by default
4. Copy the exact model name from the loaded model and set it as `GENERATION_MODEL` in `.env`

> **Recommended:** Any instruction-tuned model with 7B+ parameters works. Qwen3, Llama 3, Mistral all work well. Models with thinking mode (Qwen3) should set `THINKING_MODE=false` for speed-sensitive steps.

---

## Usage

### Ingest a URL

```python
from pipeline import ingest

result = ingest("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
print(f"Ingested {result['ingested']} chunks from {result['source']}")
```

### Ingest a local file

```python
from pipeline import ingest

result = ingest("/path/to/document.pdf")
print(f"Ingested {result['ingested']} chunks")
```

Supported file types: `.txt`, `.md`, `.pdf`, `.docx`

### Query the pipeline

```python
from pipeline import query

result = query("How does RAG reduce hallucinations?")
print(result["answer"])
print(f"Route: {result['route']}")
print(f"Sources: {result['sources']}")
```

### Force refresh a source

```python
from pipeline import ingest

# re-ingests even if source already exists — useful for cron jobs
result = ingest("https://example.com/listings", force=True)
```

### Run the ReAct agent

```python
from agent import run

result = run("What is 3847 divided by 47 and what is the square root of that result?")
print(result["answer"])
print(f"Completed in {result['steps']} steps")

# full reasoning trace
for step in result["trace"]:
    print(f"Step {step['step']}: {step['action'] or 'ANSWER'}")
```

---

## MCP Server

The pipeline is exposed as an MCP server with four tools:

| Tool | Description |
|---|---|
| `query_knowledge_base(question)` | Runs the full RAG pipeline and returns a grounded answer |
| `ingest_url(url)` | Scrapes a URL and adds it to the knowledge base |
| `ingest_file(path)` | Ingests a local file into the knowledge base |
| `get_sources(question)` | Returns raw retrieved chunks without generating an answer |
| `evaluate_pipeline()` | Runs the RAGAs evaluation suite and returns metrics |

### Start the MCP server

```bash
python mcp_server.py
```

### Test with the MCP inspector

```bash
mcp dev mcp_server.py
```

Opens a browser UI where you can call each tool manually.

### Connect to Claude Desktop

Add this to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rag-pipeline": {
      "command": "/path/to/rag-pipeline/.venv/bin/python",
      "args": ["/path/to/rag-pipeline/mcp_server.py"],
      "env": {
        "DB_ENV": "local"
      }
    }
  }
}
```

Restart Claude Desktop. You'll see a hammer icon in the chat input — your pipeline tools are now available directly inside Claude.

---

## Switching Between Local and Remote Database

```bash
# development — local PostgreSQL, no network required
DB_ENV=local

# production — any PostgreSQL provider
DB_ENV=remote
```

No code changes required — just flip the `.env` variable.

---

## Running the Evaluation Suite

```bash
python -m eval.ragas_runner
```

Runs RAGAs evaluation across both the enhanced pipeline and naive baseline, printing a comparison table with deltas.

> **Note:** Evaluation makes many sequential LLM calls. On local hardware, expect 5-15 minutes for a full run depending on your model size and question set.

---

## Project Structure

```
rag_pipeline/
├── .env.example             # environment variable template
├── config.py                # all settings in one place
├── pipeline.py              # single entry point — wires everything together
├── router.py                # adaptive routing (direct / vector / web)
├── reranker.py              # cross-encoder re-ranking
├── query_expansion.py       # multi-query rewriting
├── generator.py             # LM Studio generation wrapper
├── agent.py                 # ReAct agent loop (manual, no framework)
├── mcp_server.py            # MCP server exposing pipeline as tools
├── ingestion/
│   ├── web.py               # URL scraping (BeautifulSoup)
│   ├── local.py             # local file ingestion (txt, md, pdf, docx)
│   └── chunker.py           # semantic chunking with dynamic chunk sizing
├── retrieval/
│   ├── embedder.py          # embedding model wrapper (lazy loaded)
│   ├── vector_store.py      # pgvector read/write, source management
│   ├── bm25.py              # BM25 sparse retrieval index
│   └── hybrid.py            # RRF fusion of dense + sparse results
└── eval/
    ├── ragas_runner.py      # RAGAs evaluation suite
    └── results.md           # benchmark results
```

---

## Architecture

### High-Level Flow

```
User query
    ↓
Adaptive Router
    ├── direct  → LLM answers from training knowledge
    ├── vector  → full RAG pipeline
    │       ↓
    │   Query Expansion (3-4 variants)
    │       ↓
    │   Hybrid Retrieval
    │   ├── Dense search  (pgvector cosine similarity)
    │   └── Sparse search (BM25)
    │       ↓
    │   RRF Fusion (weighted 70/30 dense/sparse)
    │       ↓
    │   Cross-Encoder Reranker
    │       ↓
    │   Dynamic Confidence Check
    │   ├── confident    → generate answer
    │   └── low confidence → fallback to direct
    │       ↓
    │   Generation (cited answer)
    │
    └── web  → Tavily search → chunk content → generate answer
```

### Layer 1 — Ingestion

Handles both web URLs and local files. BeautifulSoup strips navigation, scripts, and boilerplate before chunking. A dynamic chunk size kicks in for short documents (under 1,000 words) to ensure fine-grained retrieval on sparse sources. Duplicate ingestion is prevented at the source level — re-ingestion requires an explicit `force=True` flag.

### Layer 2 — Retrieval

**Dense retrieval** embeds the query using `nomic-embed-text-v1.5` and finds the top-K nearest chunks by cosine similarity in pgvector using an HNSW index.

**Sparse retrieval** runs the same query through a BM25 index built over all ingested chunks. BM25 captures exact keyword matches that semantic search misses — critical for technical queries with specific terminology.

**RRF fusion** combines both result sets using Reciprocal Rank Fusion with a 70/30 dense/sparse weighting. Documents appearing in both result sets accumulate higher scores.

### Layer 3 — Reranking

A cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores each retrieved chunk against the original query. Unlike the bi-encoder used for initial retrieval, the cross-encoder sees both the query and chunk simultaneously — producing more precise relevance scores. Only `TOP_K_FINAL` chunks (default 5) are passed to generation.

### Layer 4 — Adaptive Router

Routes each query to the optimal pipeline path using two checks:

1. **Web check** — binary LLM classification: does this need live data?
2. **KB relevance check** — cosine similarity between the query and the nearest KB chunk. If the score exceeds `VECTOR_RELEVANCE_THRESHOLD` (default 0.5), route to vector retrieval.

### Layer 5 — Dynamic Confidence

After retrieval, the top reranker score is compared against a threshold that scales with the amount of content available for that source:

| Source size | Threshold | Rationale |
|---|---|---|
| < 10 chunks (sparse) | -1.0 | Lenient — limited data, accept weak matches |
| 10–30 chunks (medium) | 0.0 | Standard — moderate confidence required |
| > 30 chunks (rich) | 1.0 | Strict — plenty of data, only strong matches |

Low-confidence retrievals fall back to direct generation rather than returning poorly grounded answers.

### Layer 6 — ReAct Agent

A manually implemented reasoning loop — no LangChain or framework abstractions. The agent follows a strict `Thought → Action → Observation → repeat → Answer` cycle with four tools:

- `vector_search` — queries the knowledge base
- `web_search` — queries live web via Tavily
- `code_exec` — executes Python in an isolated namespace
- `calculator` — evaluates math expressions safely

The loop runs for a maximum of 6 steps before terminating. Building this manually means every step is fully transparent and debuggable.

### Layer 7 — MCP Server

Wraps the entire pipeline as an MCP server using the official Python SDK. Any MCP-compatible client — Claude Desktop, Cursor, or a custom agent — can call `query_knowledge_base`, `ingest_url`, `get_sources`, or `evaluate_pipeline` through the standardized protocol with no custom integration code.

---

## Configuration Reference

All settings live in `config.py` and are controlled via `.env`:

| Variable | Default | Description |
|---|---|---|
| `DB_ENV` | `local` | `local` or `remote` |
| `LOCAL_DB_URL` | — | Local PostgreSQL connection string |
| `REMOTE_DB_URL` | — | Remote PostgreSQL connection string |
| `LM_STUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio server URL |
| `GENERATION_MODEL` | — | Exact model name as shown in LM Studio |
| `THINKING_MODE` | `false` | Enable extended reasoning (slower) |
| `EMBEDDING_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | Sentence-transformers embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| `TAVILY_API_KEY` | — | Tavily API key (optional, falls back to DuckDuckGo) |
| `TOP_K_DENSE` | `20` | Dense retrieval candidates before reranking |
| `TOP_K_SPARSE` | `20` | BM25 candidates before reranking |
| `TOP_K_FINAL` | `5` | Chunks passed to generation after reranking |
| `RRF_K` | `60` | RRF constant (60 is standard) |
| `DENSE_WEIGHT` | `0.7` | Dense retrieval weight in RRF |
| `SPARSE_WEIGHT` | `0.3` | Sparse retrieval weight in RRF |
| `CHUNK_SIZE` | `512` | Words per chunk (standard documents) |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |

---

## License

MIT