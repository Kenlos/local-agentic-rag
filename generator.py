# generator.py
from openai import OpenAI
from config import LM_STUDIO_BASE_URL, LM_STUDIO_API_KEY, GENERATION_MODEL, THINKING_MODE

client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)

SYSTEM_PROMPT = """You are a precise question-answering assistant.
Answer the user's question using only the provided context chunks.
If the context does not contain enough information to answer, say so clearly.
Always cite which source your answer is drawn from."""

def generate(query: str, chunks: list[dict]) -> dict:
    """
    Generates an answer from retrieved chunks.
    Returns a dict with 'answer' and 'sources'.
    """
    if not chunks:
        return {
            "answer": "No relevant context found to answer this question.",
            "sources": []
        }

    # build context block from chunks
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['content']}"
        for c in chunks
    )

    user_message = f"""Context:
{context}

Question: {query}"""

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0.3,
        extra_body={"think": THINKING_MODE}
    )

    sources = list({c["source"] for c in chunks})

    return {
        "answer": response.choices[0].message.content.strip(),
        "sources": sources
    }