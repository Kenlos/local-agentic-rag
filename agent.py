# agent.py
import json
import re
from openai import OpenAI
from config import LM_STUDIO_BASE_URL, LM_STUDIO_API_KEY, GENERATION_MODEL, THINKING_MODE

client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)

# --- tool definitions ---
TOOLS = {
    "vector_search": "Search the local knowledge base for relevant documents. Input: a search query string.",
    "web_search":    "Search the live web for current information. Input: a search query string.",
    "code_exec":     "Execute Python code and return the result. Input: valid Python code as a string.",
    "calculator":    "Evaluate a math expression and return the result. Input: a math expression string.",
}

SYSTEM_PROMPT = """You are a reasoning agent that solves problems step by step using tools.

Available tools:
{tools}

You must follow this EXACT format for every response until you have a final answer:

Thought: <your reasoning about what to do next>
Action: <tool_name>
Input: <input to the tool>

When you have enough information to answer, use this format:
Thought: I now have enough information to answer.
Answer: <your final answer>

Rules:
- Always start with a Thought
- Only use one tool per step
- Wait for the observation before taking the next action
- Never make up tool results
- Only use tools from the available list
- Always use vector_search before answering questions about specific topics
- Always use web_search for anything time-sensitive or recent
- Never answer directly without using at least one tool first 
""".format(tools="\n".join(f"- {k}: {v}" for k, v in TOOLS.items()))


# --- tool implementations ---

def _tool_vector_search(query: str) -> str:
    from retrieval.embedder import embed
    from retrieval.vector_store import dense_search
    from retrieval.bm25 import sparse_search
    from retrieval.hybrid import reciprocal_rank_fusion
    from reranker import rerank

    q_embedding = embed([query])[0]
    dense = dense_search(q_embedding)
    sparse = sparse_search(query)
    fused = reciprocal_rank_fusion(dense, sparse)
    reranked = rerank(query, fused)

    if not reranked:
        return "No relevant documents found in the knowledge base."

    return "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['content']}"
        for c in reranked
    )

def _tool_web_search(query: str) -> str:
    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    try:
        url = f"https://lite.duckduckgo.com/lite/?q={requests.utils.quote(query)}"
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

        return "\n\n".join(snippets[:6]) or "No web results found."

    except Exception as e:
        return f"Web search failed: {str(e)}"

def _tool_code_exec(code: str) -> str:
    import io
    import contextlib

    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, {})  # isolated namespace
        result = output.getvalue()
        return result if result else "Code executed successfully with no output."
    except Exception as e:
        return f"Error: {str(e)}"

def _tool_calculator(expression: str) -> str:
    try:
        # safe eval — only math expressions
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return "Error: only basic math expressions are allowed."
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

TOOL_MAP = {
    "vector_search": _tool_vector_search,
    "web_search":    _tool_web_search,
    "code_exec":     _tool_code_exec,
    "calculator":    _tool_calculator,
}


# --- response parser ---

def _parse_response(text: str) -> dict:
    """
    Parses the model's response into thought, action, input, or answer.
    Returns a dict with keys: thought, action, input, answer
    """
    result = {
        "thought": None,
        "action": None,
        "input": None,
        "answer": None
    }

    # extract thought
    thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|Answer:|$)", text, re.DOTALL)
    if thought_match:
        result["thought"] = thought_match.group(1).strip()

    # extract answer if present
    answer_match = re.search(r"Answer:\s*(.+?)$", text, re.DOTALL)
    if answer_match:
        result["answer"] = answer_match.group(1).strip()
        return result

    # extract action + input
    action_match = re.search(r"Action:\s*(\w+)", text)
    input_match = re.search(r"Input:\s*(.+?)(?=Thought:|Action:|Answer:|$)", text, re.DOTALL)

    if action_match:
        result["action"] = action_match.group(1).strip()
    if input_match:
        result["input"] = input_match.group(1).strip()

    return result


# --- main agent loop ---

def run(question: str, max_steps: int = 6) -> dict:
    """
    Runs the ReAct agent loop.
    Returns a dict with the final answer and the full reasoning trace.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    trace = []  # full reasoning trace for transparency

    for step in range(max_steps):
        print(f"\n[agent step {step + 1}]")

        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
            extra_body={"think": THINKING_MODE}
        )

        message = response.choices[0].message

        # handle thinking mode — use content if available, else reasoning_content
        raw_text = message.content
        if not raw_text and hasattr(message, "reasoning_content"):
            raw_text = message.reasoning_content or ""

        print(f"[agent raw]\n{raw_text[:300]}...")

        parsed = _parse_response(raw_text)
        trace.append({
            "step": step + 1,
            "thought": parsed["thought"],
            "action": parsed["action"],
            "input": parsed["input"],
            "answer": parsed["answer"],
        })

        # if the model produced a final answer, we're done
        if parsed["answer"]:
            print(f"[agent] finished in {step + 1} steps")
            return {
                "answer": parsed["answer"],
                "trace": trace,
                "steps": step + 1
            }

        # if no valid action, stop to avoid infinite loop
        if not parsed["action"] or parsed["action"] not in TOOL_MAP:
            print(f"[agent] no valid action found, stopping")
            return {
                "answer": raw_text,  # return whatever the model said
                "trace": trace,
                "steps": step + 1
            }

        # execute the tool
        tool_fn = TOOL_MAP[parsed["action"]]
        print(f"[agent] calling {parsed['action']}({parsed['input'][:80]}...)")
        observation = tool_fn(parsed["input"])
        print(f"[agent] observation: {observation[:150]}...")

        # append assistant response + observation to message history
        messages.append({"role": "assistant", "content": raw_text})
        messages.append({
            "role": "user",
            "content": f"Observation: {observation}\n\nContinue."
        })

    # hit max steps without an answer
    return {
        "answer": "Agent reached maximum steps without producing a final answer.",
        "trace": trace,
        "steps": max_steps
    }