# router.py — full replacement
from openai import OpenAI
from config import LM_STUDIO_BASE_URL, LM_STUDIO_API_KEY, GENERATION_MODEL

client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)

# threshold for what counts as "relevant" in the knowledge base
# cosine similarity ranges 0-1, 0.5 is a reasonable starting point
VECTOR_RELEVANCE_THRESHOLD = 0.5

def _is_in_knowledge_base(query: str) -> bool:
    """
    Does a quick dense search to check if the knowledge base
    contains anything relevant to this query.
    Returns True if the top result exceeds the relevance threshold.
    """
    try:
        from retrieval.embedder import embed
        from retrieval.vector_store import dense_search

        q_embedding = embed([query])[0]
        results = dense_search(q_embedding, top_k=1)

        if not results:
            return False

        top_score = results[0].get("score", 0)
        print(f"[router] KB relevance score: {top_score:.4f} (threshold: {VECTOR_RELEVANCE_THRESHOLD})")
        return top_score >= VECTOR_RELEVANCE_THRESHOLD

    except Exception as e:
        print(f"[router] KB check failed: {e}")
        return False

def _is_web_query(query: str) -> bool:
    """
    Uses the LLM to check only whether this needs live web data.
    Binary decision — much more reliable than a three-way classification.
    """
    prompt = """Does this question require real-time or very recent information 
(current news, live prices, today's events, recent releases)?

Reply with YES or NO only.

Question: {query}""".format(query=query)

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1024,
        extra_body={"think": False}
    )

    message = response.choices[0].message
    raw = message.content or ""
    if not raw and hasattr(message, "reasoning_content"):
        raw = message.reasoning_content or ""

    return "yes" in raw.strip().lower()

def route(query: str) -> str:
    """
    Three-step routing:
    1. Check if query needs live web data (LLM binary decision)
    2. Check if knowledge base has relevant content (vector similarity)
    3. Default to direct if neither matches
    """
    # step 1 — web check first, it's the most clear-cut decision
    if _is_web_query(query):
        print(f"[router] → web")
        return "web"

    # step 2 — check knowledge base relevance via vector similarity
    if _is_in_knowledge_base(query):
        print(f"[router] → vector")
        return "vector"

    # step 3 — nothing relevant in KB, answer directly
    print(f"[router] → direct")
    return "direct"