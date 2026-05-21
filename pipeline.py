# pipeline.py
import os
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

def ingest(source: str, force: bool = False) -> dict:
    """
    Ingests a URL or local file path into the knowledge base.
    Skips if source already exists unless force=True.
    """
    from retrieval.vector_store import source_exists

    if not force and source_exists(source):
        print(f"[ingest] skipping — already ingested: {source}")
        return {"ingested": 0, "source": source, "skipped": True}

    if source.startswith("http://") or source.startswith("https://"):
        chunks = ingest_url(source)
    else:
        chunks = ingest_file(source)

    embeddings = embed([c["content"] for c in chunks])
    insert_chunks(chunks, embeddings)
    build_index(chunks)

    return {"ingested": len(chunks), "source": source, "skipped": False}

# This sets a dynamic threshold for data quality. Will use fallback routing if threshold not met
def _dynamic_threshold(chunk_count: int) -> float:
    """
    Returns a confidence threshold based on how much content
    exists for the most relevant source.

    Thresholds:
    - Sparse  (<10 chunks)  → -1.0  very lenient, accept weak matches
    - Medium  (10-30 chunks) → 0.0  standard threshold
    - Rich    (>30 chunks)  → 1.0  strict, only accept strong matches
    
    The more content we have on a topic, the more we expect
    retrieval to find a precise, high-scoring match.
    """
    if chunk_count < 10:
        threshold = -1.0
        quality = "sparse"
    elif chunk_count < 30:
        threshold = 0.0
        quality = "medium"
    else:
        threshold = 1.0
        quality = "rich"

    print(f"[pipeline] source has {chunk_count} chunks ({quality}) → threshold: {threshold}")
    return threshold

def _rag_query(question: str) -> dict:
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

    # get the top source and its chunk count
    first_embedding = embed([question])[0]

    if reranked:
        top_source = reranked[0]["source"]
        from retrieval.vector_store import get_source_chunk_count
        chunk_count = get_source_chunk_count(top_source)
    else:
        chunk_count = 0

    threshold = _dynamic_threshold(chunk_count)
    top_score = reranked[0].get("rerank_score", -999) if reranked else -999
    confident = top_score >= threshold

    print(f"[pipeline] top rerank score: {top_score:.4f} — {'confident' if confident else 'low confidence, will fallback'}")

    result = generate(question, reranked) if confident else {"answer": None, "sources": []}
    result["chunks"] = reranked
    result["confident"] = confident
    return result

def query(question: str) -> dict:
    """
    Main entry point with fallback routing.
    Routes → retrieves → checks quality → falls back if needed.
    """
    decision = route(question)
    print(f"[router] → {decision}")

    # web route — no fallback needed
    if decision == "web":
        result = _web_query(question)
        result["route"] = "web"
        return result
    
    # vector route — attempt retrieval, fall back to direct if low confidence
    if decision == "vector":
        result = _rag_query(question)

        if result["confident"]:
            result["route"] = "vector"
            return result

        # low confidence — fall back to direct
        print(f"[pipeline] falling back to direct due to low retrieval confidence")
        direct = _direct_query(question)
        direct["route"] = "vector→direct"  # shows the fallback happened
        direct["chunks"] = result["chunks"]  # keep chunks for transparency
        return direct

    # direct route
    result = _direct_query(question)
    result["route"] = "direct"
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

def _web_query_duckduckgo(question: str) -> dict:
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

# Tavily API web query Primary
def _web_query(question: str) -> dict:
    """
    Uses Tavily for web search — returns full article content.
    Falls back to DuckDuckGo if no API key is set.
    """
    tavily_key = os.getenv("TAVILY_API_KEY")

    if not tavily_key:
        print("[web] no Tavily key found, falling back to DuckDuckGo")
        return _web_query_duckduckgo(question)

    try:
        from tavily import TavilyClient
        from ingestion.chunker import chunk_text

        client = TavilyClient(api_key=tavily_key)

        response = client.search(
            query=question,
            search_depth="advanced",
            max_results=3,
            include_raw_content=True  # full article text not just snippets
        )

        live_chunks = []

        for result in response.get("results", []):
            # prefer raw content, fall back to snippet
            content = result.get("raw_content") or result.get("content", "")
            url = result.get("url", "")

            if content and len(content) > 100:
                chunks = chunk_text(content, source=url)
                live_chunks.extend(chunks)

        if not live_chunks:
            print("[web] Tavily returned no content, falling back to DuckDuckGo")
            return _web_query_duckduckgo(question)

        result = generate(question, live_chunks[:5])
        result["chunks"] = live_chunks[:5]
        result["route"] = "web"
        return result

    except Exception as e:
        print(f"[web] Tavily failed: {e}, falling back to DuckDuckGo")
        return _web_query_duckduckgo(question)