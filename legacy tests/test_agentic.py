# test_agentic.py
import time
from pipeline import ingest, query

print("Ingesting content...")
ingest("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
ingest("https://paulgraham.com/greatwork.html")

tests = [
    {
        "q": "How does RAG reduce hallucinations?",
        "note": "vector — KB has this"
    },
    {
        "q": "What is the latest news in AI this week?",
        "note": "web — needs live data"
    },
    {
        "q": "What is a binary search tree?",
        "note": "direct — general knowledge"
    },
    {
        "q": "What does Paul Graham say about great work and what are the latest AI productivity tools?",
        "note": "vector+web combined"
    },
]

for t in tests:
    print(f"\n{'='*60}")
    print(f"Q: {t['q']}")
    print(f"Expected: {t['note']}")
    print('='*60)
    start = time.time()
    result = query(t["q"])
    elapsed = time.time() - start
    print(f"Route:  {result['route']}")
    print(f"Time:   {elapsed:.1f}s")
    print(f"Answer: {result['answer'][:250]}...")

# cache test
print(f"\n{'='*60}")
print("Cache test — repeating first question")
print('='*60)
start = time.time()
result = query("How does RAG reduce hallucinations?")
elapsed = time.time() - start
print(f"Route:     {result['route']}")
print(f"Time:      {elapsed:.1f}s  ← should be near instant")
print(f"Cache hit: {result.get('cache_hit', False)}")