# router.py
from openai import OpenAI
from config import LM_STUDIO_BASE_URL, LM_STUDIO_API_KEY, GENERATION_MODEL

client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)

ROUTER_PROMPT = """You are a query routing classifier.

Classify the question into exactly one of these categories:
- direct: general knowledge, definitions, concepts, math, coding syntax
- vector: specific documents or topics ingested into a knowledge base  
- web: real-time info, current events, news, live data

Reply with a single word only. No explanation. No punctuation.
One of: direct, vector, web

Question: {query}
Category:"""

def _extract_route(text: str) -> str:
    """Extracts a valid route keyword from any text."""
    if not text:
        return None
    cleaned = text.strip().lower()
    # check first word first
    first_word = cleaned.split()[0].strip(".,:\n") if cleaned else ""
    if first_word in ("direct", "vector", "web"):
        return first_word
    # scan full text for any valid route keyword
    for route_key in ("direct", "web", "vector"):
        if route_key in cleaned:
            return route_key
    return None

def route(query: str) -> str:
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "user",
                "content": ROUTER_PROMPT.format(query=query)
            }
        ],
        temperature=0.0,
        max_tokens=1024,  # enough for think block + the single word answer
        extra_body={"think": False}
    )

    message = response.choices[0].message

    # try content first
    decision = _extract_route(message.content)

    # if empty, fall back to reasoning_content
    if not decision and hasattr(message, "reasoning_content"):
        print(f"[router] content empty, parsing reasoning_content")
        decision = _extract_route(message.reasoning_content)

    print(f"[router] → {decision or 'vector (default)'}")
    return decision or "vector"