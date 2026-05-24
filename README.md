# Local Agentic RAG Pipeline + MCP Server

A production-quality local RAG (Retrieval-Augmented Generation) pipeline that supercharges any local AI model with hybrid retrieval, intelligent routing, and an MCP server — callable from Claude Desktop, Cursor, or any MCP-compatible AI client.

**No cloud APIs required. No LangChain. No framework lock-in. Just plug in your local model and go.**

---

## What does this do?

When you ask a question, most AI models answer purely from their training data. This pipeline gives your local model three superpowers:

1. **Knowledge base** — ingest any web page or document, and the model answers from that content instead of guessing
2. **Live web search** — automatically searches the web for current information when the question needs it
3. **Smart routing** — figures out the best source for each question without you telling it what to do

```
You ask a question
        ↓
Router decides: use my knowledge base, search the web, or answer directly?
        ↓
Retrieves the most relevant content
        ↓
Model generates a grounded, cited answer
        ↓
Good answers are cached — repeat questions return instantly
```

---

## Benchmarks

Two evaluation runs — a RAGAs metric evaluation on a focused 3-question set, and a comprehensive real-world evaluation across 19 questions spanning four topic areas (AI/ML, software engineering, business, and science).

### Comprehensive real-world evaluation (19 questions, 4 categories)

Measures keyword coverage against ground truth answers. Excludes one data point invalidated by a system sleep event during the run.

| Metric | Naive Baseline | Enhanced Pipeline | Improvement |
|---|---|---|---|
| Overall quality | 39.0% | 66.5% | +27.5pp |
| AI / ML questions | 45.8% | 69.8% | +24.0pp |
| Software Engineering | 38.9% | 66.0% | +27.1pp |
| Business questions | 33.4% | 54.6% | +21.2pp |
| Science questions | 37.5% | 75.8% | +38.3pp |

Routing accuracy across the 19 questions:

| Route type | Accuracy | Notes |
|---|---|---|
| Vector (KB retrieval) | 92% (11/12) | One question below KB relevance threshold |
| Web search | 100% (2/2) | Keyword detection working correctly |
| Direct (model knowledge) | 50% (3/6) | Known gap — KB relevance pulls general questions into retrieval |

Biggest individual improvements over naive:

- Q18 (speed of light): +77.5pp — naive failed with vector, enhanced correctly routed direct
- Q20 (Pythagorean theorem): +71.8pp — same pattern, pure knowledge question
- Q19 (quantum computing 2026): +40.0pp — naive had no live data, enhanced fetched current results
- Q9 (database indexes): +37.8pp — cache hit, returned in 0.025s vs 26s naive

### RAGAs metric evaluation (3 questions, focused)

| Metric | Naive Baseline | Enhanced Pipeline | Improvement |
|---|---|---|---|
| Answer Relevancy | 63.5% | 97.0% | +33.5pp |
| Context Recall | 0.0% | 50.0% | +50.0pp |

> Context recall of 0% on the naive baseline means pure vector search retrieved chunks containing none of the information needed to answer correctly. Hybrid retrieval and reranking fixed this entirely.

---

## Who is this for?

- **Developers** running local models (LM Studio, Ollama, llama.cpp) who want production-quality RAG without cloud dependencies
- **Researchers** who want to query their own documents with a local model
- **Builders** who want to expose their RAG pipeline as an MCP server for Claude Desktop or Cursor

---

## What you need before starting

Before cloning this repo, make sure you have:

| Requirement | Why | How to get it |
|---|---|---|
| Python 3.11+ | Runs the pipeline | [python.org](https://python.org) |
| Docker Desktop | Runs PostgreSQL with pgvector | [docker.com/products/docker-desktop](https://docker.com/products/docker-desktop) |
| LM Studio | Runs your local AI model | [lmstudio.ai](https://lmstudio.ai) |
| Git | Clones this repo | [git-scm.com](https://git-scm.com) |

**How much RAM do I need?**

| RAM | Recommended model size | Pipeline tier |
|---|---|---|
| 8–16 GB | 7B models (Mistral 7B, Llama 3.1 8B) | small |
| 16–32 GB | 13B models (Llama 3.1 13B, Qwen 2.5 14B) | medium |
| 32 GB+ | 30B+ models (Qwen3 35B, Llama 3.1 70B Q4) | large |

---

## Setup — step by step

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/rag-pipeline.git
cd rag-pipeline
```

### Step 2 — Create a Python virtual environment

A virtual environment keeps this project's dependencies isolated from the rest of your system. Think of it as a clean room for this project.

```bash
python -m venv .venv
source .venv/bin/activate
```

> **Windows:** use `.venv\Scripts\activate` instead

You will know it is active when you see `(.venv)` at the start of your terminal prompt.

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs everything the pipeline needs — embeddings, retrieval, the MCP server, web search, and more. It may take a few minutes the first time.

### Step 4 — Configure your environment

Copy the example config file and fill in your values:

```bash
cp .env.example .env
```

Open `.env` in any text editor. Here is what each setting means:

```bash
# ── Hardware ──────────────────────────────────────────────────
# How much RAM does your machine have?
# small = 8-16GB, medium = 16-32GB, large = 32GB+
HARDWARE_TIER=large

# ── Database ──────────────────────────────────────────────────
# Which database to use. Keep this as "local" for development.
DB_ENV=local

# Your Docker PostgreSQL connection string.
# The credentials match what is in docker-compose.yml.
# Do not change this unless you change docker-compose.yml too.
LOCAL_DB_URL=postgresql://rag_user:rag_password@localhost:5432/rag_db

# If you want to use a remote database (Supabase, Neon, Railway etc.)
# fill this in and set DB_ENV=remote
REMOTE_DB_URL=postgresql://user:password@your-host:5432/dbname

# ── Your Local Model ──────────────────────────────────────────
# The URL where LM Studio's server is running.
# This is the default — only change if you moved it.
LM_STUDIO_BASE_URL=http://localhost:1234/v1

# The exact model name as shown in LM Studio's Developer panel.
# Copy it exactly — even a small difference will cause errors.
GENERATION_MODEL=your-exact-model-name-here

# Whether to use the model's extended reasoning mode.
# Keep this false for speed. Set true only for complex tasks.
THINKING_MODE=false

# ── Embeddings ────────────────────────────────────────────────
# These are the models used for search — not generation.
# The defaults work well. Only change if you know what you are doing.
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# ── Web Search ────────────────────────────────────────────────
# Optional but strongly recommended. Free tier at app.tavily.com
# Without this, web search falls back to DuckDuckGo (less content)
TAVILY_API_KEY=tvly-xxxxxxxxxx
```

### Step 5 — Load your model in LM Studio

1. Open LM Studio
2. Download a model if you have not already (Qwen3, Llama 3, Mistral all work)
3. Go to the **Developer** tab (the `</>` icon in the sidebar)
4. Select your model from the dropdown
5. Click **Start Server**
6. Copy the exact model name shown — you will need it for `GENERATION_MODEL` in `.env`

> The server needs to stay running while you use the pipeline. LM Studio can run in the background.

### Step 6 — Run the setup script

This script checks everything is working and starts the database automatically:

```bash
python setup.py
```

It will:
- Check your Python version
- Start PostgreSQL in Docker (creates all tables automatically)
- Verify your `.env` values are set
- Test the database connection
- Test the LM Studio connection
- Run a quick end-to-end test

If anything is wrong it tells you exactly what to fix. A successful run looks like:

```
✅ Python 3.11.x
✅ Docker 27.x
✅ PostgreSQL container started
✅ GENERATION_MODEL is set
✅ Connected to database (0 documents in knowledge base)
✅ LM Studio is running with 1 model loaded
✅ Tier: large — 30B+ models, 32GB+ RAM
✅ Pipeline working — route: vector

══════════════════════════════════════════════════
  ✅ Setup complete — pipeline is ready
══════════════════════════════════════════════════
```

---

## Troubleshooting setup

### `role "rag_user" does not exist`

You have a local PostgreSQL installation already running on port 5432 that conflicts with Docker. Stop it first:

```bash
# Mac
brew services stop postgresql@16

# Linux
sudo systemctl stop postgresql
```

Then re-run `python setup.py`.

If you need both running at the same time, change the Docker port in `docker-compose.yml` from `5432:5432` to `5433:5432` and update `LOCAL_DB_URL` in `.env` to use port `5433`.

### `connection refused` on LM Studio check

LM Studio is not running or the server has not been started. Open LM Studio, go to the Developer tab, and click **Start Server**.

### `GENERATION_MODEL` errors

The model name in your `.env` does not exactly match what LM Studio shows. In LM Studio's Developer panel, look at the loaded model identifier and copy it character for character into `GENERATION_MODEL`.

---

## Using the pipeline

### Ingest a web page

```python
from pipeline import ingest

result = ingest("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
print(f"Ingested {result['ingested']} chunks")
```

### Ingest a local file

Supported formats: `.txt`, `.md`, `.pdf`, `.docx`

```python
from pipeline import ingest

result = ingest("/path/to/your/document.pdf")
print(f"Ingested {result['ingested']} chunks")
```

### Ask a question

```python
from pipeline import query

result = query("How does RAG reduce hallucinations?")
print(result["answer"])
print(f"Route used: {result['route']}")
print(f"Sources: {result['sources']}")
```

### Refresh a source (for scheduled updates)

```python
from pipeline import refresh_source

# deletes old chunks and re-ingests fresh content
result = refresh_source("https://example.com/page-that-changes")
```

### Force re-ingest (override duplicate check)

```python
from pipeline import ingest

result = ingest("https://example.com", force=True)
```

---

## MCP Server

The MCP (Model Context Protocol) server exposes your entire pipeline as tools that any MCP-compatible AI client can call — including Claude Desktop and Cursor.

### What is MCP?

MCP is an open standard created by Anthropic (now backed by OpenAI, Google, and Microsoft) for connecting AI models to external tools and data. Think of it as a USB-C standard for AI — any MCP-compatible client can plug into any MCP-compatible server without custom integration code.

### Available tools

| Tool | What it does |
|---|---|
| `query_knowledge_base(question)` | Runs the full pipeline and returns a grounded answer |
| `ingest_url(url)` | Scrapes a URL and adds it to the knowledge base |
| `ingest_file(path)` | Ingests a local file into the knowledge base |
| `get_sources(question)` | Returns raw retrieved chunks without generating an answer |
| `evaluate_pipeline()` | Runs the RAGAs evaluation suite |

### Start the server

```bash
python mcp_server.py
```

### Test with the MCP Inspector

```bash
mcp dev mcp_server.py
```

This opens a browser UI where you can call each tool manually and see the responses before connecting to a real client.

### Connect to Claude Desktop

Add this to your Claude Desktop config file:

**Mac:** `~/Library/Application Support/Claude/claude_desktop_config.json`

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

Replace `/path/to/rag-pipeline` with your actual project path. Fully quit and restart Claude Desktop. You will see a hammer icon in the chat input — your pipeline tools are now available.

---

## Switching between local and remote database

The pipeline supports any PostgreSQL provider — local Docker, Supabase, Neon, Railway, etc.

```bash
# development — local Docker, works offline, no network required
DB_ENV=local

# production — remote provider, accessible from anywhere
DB_ENV=remote
```

No code changes needed — just flip `DB_ENV` in your `.env`.

> **Tip:** develop locally, deploy remotely. Your data will not sync automatically between environments — re-ingest your sources when switching to remote for the first time.

---

## Run the evaluation suite

Measures your pipeline against a naive dense-search baseline:

```bash
python -m eval.ragas_runner
```

Produces a comparison table showing how much hybrid retrieval + reranking improves over plain vector search. Expect 5-15 minutes on local hardware.

---

## Docker commands reference

```bash
# start PostgreSQL in the background
docker compose up -d postgres

# stop PostgreSQL (data is preserved)
docker compose down

# stop PostgreSQL and delete all data (full reset)
docker compose down -v

# view PostgreSQL logs
docker compose logs postgres

# open a database shell
docker compose exec postgres psql -U rag_user -d rag_db

# check container status
docker compose ps
```

> `docker compose down -v` deletes all your ingested data. Use only when you want a clean slate.

---

## Architecture

### How routing works

Every query goes through a three-tier decision process before any retrieval happens:

```
Tier 1 — Keyword check (instant, no AI call)
    Does the question contain words like "today", "latest",
    "this week", "current", "news"?
    Yes → route to web search
    No  → continue to tier 2

Tier 2 — Knowledge base relevance check (fast, no AI call)
    How similar is this question to content already in the KB?
    Score >= 0.60 → route to vector retrieval
    Score <  0.60 → route to direct (model answers from training)

Tier 3 — Multi-part detection (one YES/NO AI call)
    Does the question have two distinct parts needing different sources?
    Example: "What does X say about Y AND what are the latest Z?"
    Yes → use both vector + web, combine results before answering
    No  → use single best route from tier 1 or 2
```

This means for most queries, zero or one AI call is made for routing. The model is only asked to make a decision when rule-based logic genuinely cannot — detecting multi-part questions.

### How retrieval works

When a question routes to the knowledge base, it goes through five stages:

**1 — Query expansion**

The model rewrites the question in multiple ways. "How does RAG work?" becomes three or four variants. Each variant is searched independently, increasing the chance of finding the right content.

**2 — Dense retrieval**

Each query variant is converted to a vector using `nomic-embed-text-v1.5` and compared against all stored chunks using cosine similarity in pgvector. This finds semantically similar content even when the exact words do not match.

**3 — Sparse retrieval (BM25)**

The same variants are run through a BM25 keyword index. This catches exact term matches that semantic search can miss — especially useful for technical terminology, proper nouns, and specific jargon.

**4 — RRF fusion**

Results from dense and sparse retrieval are combined using Reciprocal Rank Fusion (70% dense weight, 30% sparse weight). Chunks appearing in both result sets score higher. This produces a ranked list better than either method alone.

**5 — Cross-encoder reranking**

A cross-encoder model (`ms-marco-MiniLM-L-6-v2`) re-scores the top candidates by reading the query and each chunk together — much more precise than the bi-encoder used for initial retrieval. Only the top 5 chunks pass through to generation.

**Dynamic confidence threshold**

Before generating, the pipeline checks whether the reranker score exceeds a threshold that scales with data quality:

| Source size | Threshold | Meaning |
|---|---|---|
| Less than 10 chunks | -1.0 (lenient) | Sparse source — accept weak matches |
| 10 to 30 chunks | 0.0 (standard) | Moderate source — require some confidence |
| More than 30 chunks | 1.0 (strict) | Rich source — only strong matches pass |

If confidence is too low, the pipeline automatically retries with web search rather than generating a poorly-grounded answer.

### How caching works

Every substantive answer is stored in the `query_cache` table in PostgreSQL. On the next similar question, the pipeline does a full-text similarity search against cached questions — a sub-millisecond operation — and returns the cached answer immediately.

This reduces repeat query latency from 30-90 seconds to under 0.1 seconds.

### Full system diagram

```
                    ┌─────────────────────────────────┐
                    │           User Query             │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │         Cache Lookup             │
                    │   (PostgreSQL full-text search)  │
                    └──────┬───────────────┬───────────┘
                     Cache │               │ Miss
                     Hit   │               │
                           │    ┌──────────▼──────────┐
                           │    │    Agentic Router    │
                           │    │                      │
                           │    │  Tier 1: Keywords    │
                           │    │  Tier 2: KB score    │
                           │    │  Tier 3: Multi-part? │
                           │    └──┬──────┬──────┬─────┘
                           │  web  │vector│direct│
                    ┌──────▼──┐ ┌──▼───┐ ┌──▼────┐
                    │ Return  │ │Tavily│ │Query  │ │Direct│
                    │ Cached  │ │Search│ │Expand │ │ LLM  │
                    │ Answer  │ └──┬───┘ └──┬────┘ └──┬───┘
                    └─────────┘    │    ┌───┴────┐    │
                                   │    │Dense + │    │
                                   │    │Sparse  │    │
                                   │    │  RRF   │    │
                                   │    └───┬────┘    │
                                   │    ┌───┴────┐    │
                                   │    │Rerank  │    │
                                   │    └───┬────┘    │
                                   │    ┌───┴────┐    │
                                   │    │Confid- │    │
                                   │    │ence?   │    │
                                   │    └──┬──┬──┘    │
                              combine   yes│  │no      │
                           ┌────────────┘  │  └──►web  │
                    ┌──────▼──────┐ ┌──────▼──────────▼──┐
                    │   Combine   │ │      Generate       │
                    │  Contexts   │ │   (cited answer)    │
                    └──────┬──────┘ └──────────┬──────────┘
                           └──────────┬─────────┘
                    ┌─────────────────▼─────────────────┐
                    │           Cache Store              │
                    │    (if answer is substantive)      │
                    └─────────────────┬─────────────────┘
                    ┌─────────────────▼─────────────────┐
                    │           Final Answer             │
                    └───────────────────────────────────┘
```

---

## Project structure

```
rag_pipeline/
├── docker-compose.yml       # PostgreSQL service — one command setup
├── init.sql                 # Database schema — runs automatically on first start
├── setup.py                 # Setup validator — checks everything before you code
├── config.py                # All settings — controlled via .env
├── pipeline.py              # Main entry point — wires everything together
├── router.py                # Agentic router — tiered routing with multi-tool planning
├── reranker.py              # Cross-encoder reranking
├── query_expansion.py       # Multi-query rewriting for better recall
├── generator.py             # LM Studio generation wrapper
├── agent.py                 # ReAct agent loop (manual, no framework)
├── mcp_server.py            # MCP server — exposes pipeline as callable tools
├── .env.example             # Environment variable template
├── ingestion/
│   ├── web.py               # URL scraping (BeautifulSoup + Tavily)
│   ├── local.py             # Local file ingestion (txt, md, pdf, docx)
│   └── chunker.py           # Dynamic chunking with size adaptation
├── retrieval/
│   ├── embedder.py          # Embedding model wrapper (lazy loaded, cached)
│   ├── vector_store.py      # pgvector operations + cache + source management
│   ├── bm25.py              # BM25 sparse retrieval index
│   └── hybrid.py            # RRF fusion of dense + sparse results
└── eval/
    ├── ragas_runner.py      # RAGAs evaluation suite
    └── results.md           # Benchmark results
```

---

## Configuration reference

All settings live in `config.py` and are controlled via `.env`. You should never need to edit `config.py` directly.

| Variable | Default | Description |
|---|---|---|
| `HARDWARE_TIER` | `large` | `small` / `medium` / `large` — tunes chunk size and retrieval depth |
| `DB_ENV` | `local` | `local` (Docker) or `remote` (any PostgreSQL provider) |
| `LOCAL_DB_URL` | — | Local Docker connection string |
| `REMOTE_DB_URL` | — | Remote provider connection string |
| `LM_STUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio server URL |
| `GENERATION_MODEL` | — | Exact model name as shown in LM Studio |
| `THINKING_MODE` | `false` | Enable extended reasoning (slower but deeper) |
| `EMBEDDING_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | Embedding model for vector search |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| `TAVILY_API_KEY` | — | Optional — web search (falls back to DuckDuckGo without it) |

---

## Compatibility

Tested with the following local inference servers:

| Server | Status | Notes |
|---|---|---|
| LM Studio | ✅ Tested | Recommended. OpenAI-compatible endpoint at port 1234 |
| Ollama | ✅ Compatible | Set `LM_STUDIO_BASE_URL=http://localhost:11434/v1` |
| llama.cpp server | ✅ Compatible | Set `LM_STUDIO_BASE_URL` to your server address |
| Jan | ✅ Compatible | OpenAI-compatible endpoint |

Any server that exposes an OpenAI-compatible `/v1/chat/completions` endpoint will work.

**Tested models:**

| Model | Size | Tier | Notes |
|---|---|---|---|
| Qwen3 35B-A3B | 35B MoE | large | Recommended — strong reasoning, fast MoE inference |
| Qwen3 8B | 8B | small/medium | Good quality, fast |
| Llama 3.1 8B | 8B | small | Solid baseline |
| Mistral 7B | 7B | small | Fast, good for low-RAM setups |

---

## License

MIT — use it, fork it, build on it.
