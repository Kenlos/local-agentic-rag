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
from router import route
import requests
from bs4 import BeautifulSoup

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

def _rag_query(question: str) -> dict:
    """Internal — runs the full RAG retrieval + generation pipeline."""
    queries = expand_query(question)

    all_dense, all_sparse = [], []
    for q in queries:
        q_embedding = embed([q])[0]
        all_dense.extend(dense_search(q_embedding))
        all_sparse.extend(sparse_search(q))

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

    fused = reciprocal_rank_fusion(dense_deduped, sparse_deduped)
    reranked = rerank(question, fused)
    result = generate(question, reranked)
    result["chunks"] = reranked
    return result

def _direct_query(question: str) -> dict:
    """Internal — answers directly from model knowledge, no retrieval."""
    from openai import OpenAI
    from config import LM_STUDIO_BASE_URL, LM_STUDIO_API_KEY, GENERATION_MODEL, THINKING_MODE

    client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": question}],
        temperature=0.3,
        extra_body={"think": THINKING_MODE}
    )
    return {
        "answer": response.choices[0].message.content.strip(),
        "sources": [],
        "chunks": [],
        "route": "direct"
    }

def _web_query(question: str) -> dict:
    """
    Fetches live web results using DuckDuckGo lite and runs generation over them.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    try:
        # DuckDuckGo lite — more scraper friendly than html endpoint
        search_url = (
            f"https://lite.duckduckgo.com/lite/?q={requests.utils.quote(question)}"
        )
        response = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # extract result snippets directly from the page
        snippets = []
        sources = []

        for result in soup.select(".result-snippet"):
            text = result.get_text(strip=True)
            if text:
                snippets.append(text)

        for link in soup.select(".result-link"):
            href = link.get("href")
            if href and href.startswith("http"):
                sources.append(href)

        if not snippets:
            # fallback — grab all meaningful text from the page
            for p in soup.find_all(["p", "td"]):
                text = p.get_text(strip=True)
                if len(text) > 80:
                    snippets.append(text)

        if not snippets:
            return {
                "answer": "Could not retrieve live web results for this query.",
                "sources": [],
                "chunks": [],
                "route": "web"
            }

        # build fake chunks from snippets so generate() can consume them
        live_chunks = [
            {
                "content": snippet,
                "source": sources[i] if i < len(sources) else search_url,
                "chunk_index": i,
                "rrf_score": 0.0,
                "rerank_score": 0.0
            }
            for i, snippet in enumerate(snippets[:8])
        ]

        result = generate(question, live_chunks)
        result["chunks"] = live_chunks
        result["route"] = "web"
        return result

    except Exception as e:
        return {
            "answer": f"Web search failed: {str(e)}",
            "sources": [],
            "chunks": [],
            "route": "web"
        }

def query(question: str) -> dict:
    """
    Main entry point. Routes the query then executes the right pipeline.
    """
    decision = route(question)
    print(f"[router] → {decision}")

    if decision == "direct":
        result = _direct_query(question)
    elif decision == "web":
        result = _web_query(question)
    else:
        result = _rag_query(question)

    result["route"] = decision
    return result