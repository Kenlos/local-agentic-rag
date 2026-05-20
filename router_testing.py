# add to your test script or run fresh
from pipeline import query

test_queries = [
    "What is a binary search tree?",           # expect: direct
    "How does RAG reduce hallucinations?",      # expect: direct or vector
    "What are the top AI news stories today?",  # expect: web
]

for q in test_queries:
    print(f"\nQ: {q}")
    result = query(q)
    print(f"Route: {result['route']}")
    print(f"Answer preview: {result['answer'][:200]}...")