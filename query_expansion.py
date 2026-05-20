# query_expansion.py
from openai import OpenAI
from config import LM_STUDIO_BASE_URL, LM_STUDIO_API_KEY, GENERATION_MODEL

client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)

def expand_query(query: str) -> list[str]:
    """
    Rewrites the query in multiple ways to improve retrieval recall.
    Returns a list of query variants including the original.
    """
    prompt = f"""Generate 3 different ways to ask the following question. 
Return only the questions, one per line, no numbering or explanation.

Question: {query}"""

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        extra_body={"think": False}
    )

    raw = response.choices[0].message.content.strip()
    variants = [q.strip() for q in raw.split("\n") if q.strip()]

    # always include the original
    all_queries = [query] + variants
    return all_queries[:4]  # cap at 4 total