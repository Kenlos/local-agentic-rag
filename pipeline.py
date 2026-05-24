# pipeline.py 
import os
from ingestion.web import ingest_url
from ingestion.local import ingest_file
from retrieval.embedder import embed
from retrieval.vector_store import (
    insert_chunks, dense_search,
    source_exists, delete_source,
    get_source_chunk_count,
    cache_lookup, cache_store
)
from retrieval.bm25 import build_index, sparse_search
from retrieval.hybrid import reciprocal_rank_fusion
from reranker import rerank
from query_expansion import expand_query
from generator import generate
from router import plan


# ─── Ingestion ────────────────────────────────────────────────────────────────

def ingest(source: str, force: bool = False) -> dict:
    """
    Ingests a URL or local file into the knowledge base.
    Skips if already ingested unless force=True.
    """
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


def refresh_source(source: str) -> dict:
    """Deletes and re-ingests a source. Used for scheduled content refreshes."""
    delete_source(source)
    return ingest(source, force=True)


# ─── Internal pipeline stages ─────────────────────────────────────────────────

def _dynamic_threshold(chunk_count: int) -> float:
    """
    Scales the reranker confidence threshold to data quality.

    The more content we have on a topic, the more we expect
    retrieval to find a precise match. A 2-chunk source gets
    a lenient threshold. A 100-chunk source gets a strict one.
    """
    if chunk_count < 10:
        label, threshold = "sparse", -1.0
    elif chunk_count < 30:
        label, threshold = "medium", 0.0
    else:
        label, threshold = "rich", 1.0

    print(f"[pipeline] {chunk_count} chunks ({label}) → threshold {threshold}")
    return threshold


def _rag_query(question: str) -> dict:
    """
    Full RAG retrieval pipeline.
    Returns result with 'confident' flag so the caller knows
    whether to trust this result or fall back.
    """
    # query expansion — multiple variants improve recall
    queries = expand_query(question)

    all_dense, all_sparse = [], []
    for q in queries:
        q_embedding = embed([q])[0]
        all_dense.extend(dense_search(q_embedding))
        all_sparse.extend(sparse_search(q))

    # deduplicate before fusion
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

    # dynamic confidence threshold
    if reranked:
        top_source = reranked[0]["source"]
        chunk_count = get_source_chunk_count(top_source)
        threshold = _dynamic_threshold(chunk_count)
        top_score = reranked[0].get("rerank_score", -999)
    else:
        threshold, top_score = 0.0, -999

    confident = top_score >= threshold
    print(f"[pipeline] rerank score: {top_score:.4f} — {'confident ✓' if confident else 'low confidence ✗'}")

    if confident:
        result = generate(question, reranked)
    else:
        result = {"answer": "", "sources": []}

    result["chunks"] = reranked
    result["confident"] = confident
    return result


def _web_query(question: str) -> dict:
    """
    Live web search via Tavily with DuckDuckGo fallback.
    """
    tavily_key = os.getenv("TAVILY_API_KEY")

    if tavily_key:
        try:
            from tavily import TavilyClient
            from ingestion.chunker import chunk_text

            client = TavilyClient(api_key=tavily_key)
            response = client.search(
                query=question,
                search_depth="advanced",
                max_results=3,
                include_raw_content=True
            )

            live_chunks = []
            for r in response.get("results", []):
                content = r.get("raw_content") or r.get("content", "")
                url = r.get("url", "")
                if content and len(content) > 100:
                    from ingestion.chunker import chunk_text
                    chunks = chunk_text(content, source=url)
                    live_chunks.extend(chunks)

            if live_chunks:
                result = generate(question, live_chunks[:5])
                result["chunks"] = live_chunks[:5]
                return result

        except Exception as e:
            print(f"[web] Tavily failed: {e}, falling back to DuckDuckGo")

    # DuckDuckGo fallback
    return _web_query_duckduckgo(question)


def _web_query_duckduckgo(question: str) -> dict:
    """DuckDuckGo fallback when Tavily is unavailable."""
    import requests
    from bs4 import BeautifulSoup
    from ingestion.chunker import chunk_text

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    try:
        url = f"https://lite.duckduckgo.com/lite/?q={requests.utils.quote(question)}"
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        snippets = [
            r.get_text(strip=True)
            for r in soup.select(".result-snippet")
            if r.get_text(strip=True)
        ]
        if not snippets:
            for p in soup.find_all(["p", "td"]):
                text = p.get_text(strip=True)
                if len(text) > 80:
                    snippets.append(text)

        live_chunks = [
            {
                "content": s, "source": url,
                "chunk_index": i,
                "rrf_score": 0.0, "rerank_score": 0.0
            }
            for i, s in enumerate(snippets[:8])
        ]

        result = generate(question, live_chunks)
        result["chunks"] = live_chunks
        return result

    except Exception as e:
        return {
            "answer": f"Web search failed: {e}",
            "sources": [], "chunks": []
        }


def _direct_query(question: str) -> dict:
    """Answers directly from the model's training knowledge."""
    from openai import OpenAI
    from config import (
        LM_STUDIO_BASE_URL, LM_STUDIO_API_KEY,
        GENERATION_MODEL, THINKING_MODE
    )
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
        "confident": True
    }


def _is_substantive(answer: str) -> bool:
    """
    Checks if an answer is worth caching.

    We don't evaluate quality with an LLM call — that's the
    overhead we're avoiding. Instead we use simple heuristics:
    - Long enough to be a real answer (not just "I don't know")
    - Doesn't contain failure phrases

    This is intentionally simple. The model's thinking mode
    already handles output quality. We just need to catch
    clear failures before caching them.
    """
    if not answer or len(answer) < 100:
        return False
    failure_phrases = [
        "i cannot answer",
        "i don't have",
        "no information",
        "web search failed",
        "could not retrieve"
    ]
    answer_lower = answer.lower()
    return not any(phrase in answer_lower for phrase in failure_phrases)


# ─── Main entry point ─────────────────────────────────────────────────────────

def query(question: str) -> dict:
    """
    Main pipeline entry point.

    Three things beyond basic RAG:
    1. Cache — return immediately if we've seen this before
    2. Multi-tool planning — use the right source(s) for the question
    3. Web retry — fall back to live search if vector retrieval fails
    """

    # ── 1. Cache check ────────────────────────────────────────────
    # Sub-millisecond full-text lookup.
    # Solves the 57s latency problem for repeated queries.
    cached = cache_lookup(question)
    if cached:
        print(f"[pipeline] cache hit (quality: {cached['quality']:.2f}, hits: {cached['hits']})")
        return {
            "answer": cached["answer"],
            "sources": cached.get("sources", []),
            "chunks": [],
            "route": f"cache:{cached['route']}",
            "cache_hit": True
        }

    # ── 2. Plan ───────────────────────────────────────────────────
    # Router analyzes the question and returns a structured plan.
    # One LLM call. No evaluation call. No memory lookup.
    retrieval_plan = plan(question)
    tools = retrieval_plan.get("tools", ["direct"])
    combine = retrieval_plan.get("combine", False)
    sub_queries = retrieval_plan.get("sub_queries", {})

    results_by_tool = {}

    # ── 3. Execute ────────────────────────────────────────────────
    for tool in tools:
        print(f"\n[pipeline] executing: {tool}")

        if tool == "vector":
            q = sub_queries.get("vector", question)
            result = _rag_query(q)
            results_by_tool["vector"] = result

        elif tool == "web":
            q = sub_queries.get("web", question)
            result = _web_query(q)
            results_by_tool["web"] = result

        elif tool == "direct":
            result = _direct_query(question)
            results_by_tool["direct"] = result

    # ── 4. Combine ────────────────────────────────────────────────
    # Merge results from multiple tools into a single context
    # and re-generate. This produces more comprehensive answers
    # for multi-part questions.
    if combine and len(results_by_tool) > 1:
        print(f"\n[pipeline] combining: {list(results_by_tool.keys())}")

        combined_chunks = []
        all_sources = set()
        for tool_result in results_by_tool.values():
            combined_chunks.extend(tool_result.get("chunks", []))
            for s in tool_result.get("sources", []):
                all_sources.add(s)

        combined_chunks = combined_chunks[:8]
        final_result = generate(question, combined_chunks)
        final_result["chunks"] = combined_chunks
        final_result["sources"] = list(all_sources)
        final_result["route"] = "+".join(tools)

    else:
        # single tool — priority: confident vector > web > direct
        if "vector" in results_by_tool and results_by_tool["vector"].get("confident"):
            final_result = results_by_tool["vector"]
            final_result["route"] = "vector"
        elif "web" in results_by_tool:
            final_result = results_by_tool["web"]
            final_result["route"] = "web"
        elif "direct" in results_by_tool:
            final_result = results_by_tool["direct"]
            final_result["route"] = "direct"
        else:
            final_result = list(results_by_tool.values())[0]
            final_result["route"] = tools[0]

    # ── 5. Web retry on vector failure ────────────────────────────
    # If vector retrieval wasn't confident AND we haven't already
    # tried web, fall back to live search.
    #
    # Why not re-evaluate with an LLM? The model's thinking mode
    # already handles output quality. We only need to catch the
    # specific case where retrieval structurally failed — which
    # the 'confident' flag tells us directly without an extra call.
    vector_failed = (
        "vector" in results_by_tool and
        not results_by_tool["vector"].get("confident") and
        "web" not in results_by_tool
    )

    if vector_failed:
        print(f"\n[pipeline] vector not confident, retrying with web")
        web_result = _web_query(question)
        web_answer = web_result.get("answer", "")

        # only use web result if it's actually better than empty
        if _is_substantive(web_answer):
            web_result["route"] = f"vector→web"
            final_result = web_result
        else:
            # web also failed — fall back to direct
            print(f"[pipeline] web also failed, falling back to direct")
            final_result = _direct_query(question)
            final_result["route"] = "vector→web→direct"

    # ── 6. Cache ──────────────────────────────────────────────────
    # Store good answers. No LLM quality scoring — just heuristics.
    # The model's thinking mode already ensured output quality.
    answer = final_result.get("answer", "")
    if _is_substantive(answer):
        cache_store(
            question,
            answer,
            final_result.get("route", "unknown"),
            final_result.get("sources", []),
            0.8  # fixed score — we trust the model's own quality control
        )
        print(f"[pipeline] answer cached")

    return final_result