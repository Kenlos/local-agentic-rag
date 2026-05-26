# router.py — full replacement
import json
from openai import OpenAI
from config import LM_STUDIO_BASE_URL, LM_STUDIO_API_KEY, GENERATION_MODEL

client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)

# ─── Thresholds ───────────────────────────────────────────────────────────────

KB_RELEVANCE_THRESHOLD = 0.60
WEB_KEYWORDS = [
    "today", "this week", "this month", "latest", "current", "recent",
    "news", "now", "right now", "live", "2026", "2025", "just announced",
    "breaking", "update", "release"
]

# ─── Multi-part detection prompt ──────────────────────────────────────────────
# Only used when we need to decide between single vs multi-tool.
# Much simpler than asking the model to produce a full plan.

MULTI_PART_PROMPT = """Does this question have TWO OR MORE completely distinct parts 
that EACH require different information sources to answer properly?

Only answer YES if both parts are substantial and genuinely need different sources.
Simple questions with context do NOT count as multi-part.

Examples of YES:
- "What does Paul Graham say about work AND what are the latest AI tools in 2026?"
- "Explain Docker and also what new Docker features were released this month?"

Examples of NO:
- "How does RAG reduce hallucinations?" (single topic)
- "What is PostgreSQL and how does it work?" (single topic with elaboration)
- "What are the latest AI developments?" (single topic needing web)

Reply with only YES or NO.

Question: {question}"""

_multipart_cache: dict[str, bool] = {}

def _is_multi_part(query: str) -> bool:
    if query in _multipart_cache:
        return _multipart_cache[query]
    raw = _call_llm(MULTI_PART_PROMPT.format(question=query))
    result = "yes" in raw.lower()
    _multipart_cache[query] = result
    return result


def _kb_relevance_score(question: str) -> float:
    """
    Fast cosine similarity check — no LLM, just an embedding + index lookup.
    Returns similarity score of the single best matching chunk.
    """
    try:
        from retrieval.embedder import embed
        from retrieval.vector_store import dense_search
        q_embedding = embed([question])[0]
        results = dense_search(q_embedding, top_k=1)
        score = results[0].get("score", 0.0) if results else 0.0
        print(f"[router] KB relevance: {score:.4f}")
        return score
    except Exception as e:
        print(f"[router] KB check failed: {e}")
        return 0.0


def _needs_web(question: str) -> bool:
    """
    Keyword-based web detection — no LLM needed.
    If the question contains time-sensitive keywords, route to web.
    Fast and reliable for the clear-cut cases.
    """
    q_lower = question.lower()
    return any(kw in q_lower for kw in WEB_KEYWORDS)


def _call_llm(prompt: str, max_tokens: int = 256) -> str:
    """LLM call with aggressive JSON extraction from any response field."""
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
        extra_body={"think": False}
    )
    message = response.choices[0].message

    # try content first
    raw = (message.content or "").strip()

    # fall back to reasoning_content
    if not raw and hasattr(message, "reasoning_content"):
        reasoning = message.reasoning_content or ""
        # for YES/NO responses
        if "YES" in reasoning.upper():
            return "YES"
        if "NO" in reasoning.upper():
            return "NO"
        # for JSON responses — extract the object
        start = reasoning.find("{")
        end = reasoning.rfind("}") + 1
        if start != -1 and end > start:
            raw = reasoning[start:end]

    return raw.strip()


def _is_multi_part(question: str) -> bool:
    """
    Asks the LLM a simple YES/NO question — much more reliable
    than asking it to produce a full JSON plan.
    """
    raw = _call_llm(MULTI_PART_PROMPT.format(question=question))
    result = "yes" in raw.lower()
    print(f"[router] multi-part: {result}")
    return result


def _get_sub_queries(question: str, kb_score: float) -> dict:
    """
    Gets tool-specific sub-queries for multi-part questions.
    Separate prompt from the multi-part check — simpler, more reliable.
    """
    raw = _call_llm(
        SUBQUERY_PROMPT.format(question=question, kb_score=kb_score),
        max_tokens=256
    )
    try:
        clean = raw.strip()
        if "```" in clean:
            parts = clean.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    clean = part
                    break
        return json.loads(clean)
    except Exception:
        return {"vector": question, "web": question}


def plan(question: str) -> dict:
    """
    Creates a retrieval plan using a tiered decision process:

    Tier 1 — keyword check (no LLM, instant):
        Does the question contain time-sensitive keywords?
        If yes → web

    Tier 2 — KB relevance check (no LLM, fast):
        Does the KB have relevant content?
        If yes → vector
        If no  → direct

    Tier 3 — multi-part check (one LLM YES/NO call):
        Does the question need multiple sources?
        If yes → vector + web with specific sub-queries

    This approach uses the LLM only where rule-based logic
    genuinely can't make the decision — multi-part detection.
    Everything else is deterministic.
    """
    kb_score = _kb_relevance_score(question)

    # tier 1 — keyword-based web detection
    # handles "latest news", "this week", "today" etc.
    # no LLM needed — these are unambiguous signals
    if _needs_web(question):
        print(f"[router] tier 1: web keywords detected")

        # still check if KB is also relevant — might need both
        if kb_score >= KB_RELEVANCE_THRESHOLD and _is_multi_part(question):
            sub_queries = _get_sub_queries(question, kb_score)
            print(f"[router] tools: ['vector', 'web'] | combine: True")
            return {
                "tools": ["vector", "web"],
                "combine": True,
                "sub_queries": sub_queries,
                "reasoning": "web keywords + KB relevant — combining sources",
                "kb_score": kb_score
            }

        print(f"[router] tools: ['web'] | combine: False")
        return {
            "tools": ["web"],
            "combine": False,
            "sub_queries": {"web": question},
            "reasoning": "web keywords detected",
            "kb_score": kb_score
        }

    # tier 2 — KB relevance check
    # no LLM needed — cosine similarity tells us directly
    if kb_score >= KB_RELEVANCE_THRESHOLD:
        # check if it's also a multi-part question needing web
        if _is_multi_part(question):
            sub_queries = _get_sub_queries(question, kb_score)
            print(f"[router] tools: ['vector', 'web'] | combine: True")
            return {
                "tools": ["vector", "web"],
                "combine": True,
                "sub_queries": sub_queries,
                "reasoning": "KB relevant + multi-part — combining sources",
                "kb_score": kb_score
            }

        print(f"[router] tools: ['vector'] | combine: False")
        return {
            "tools": ["vector"],
            "combine": False,
            "sub_queries": {"vector": question},
            "reasoning": "KB has relevant content",
            "kb_score": kb_score
        }

    # tier 3 — nothing in KB, not a web query
    # answer directly from model knowledge
    print(f"[router] tools: ['direct'] | combine: False")
    return {
        "tools": ["direct"],
        "combine": False,
        "sub_queries": {},
        "reasoning": "no KB content, no web keywords — direct answer",
        "kb_score": kb_score
    }