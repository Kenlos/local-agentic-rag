# test_pipeline_capabilities.py
import time
from pipeline import ingest, query

# ─── Test Sites ───────────────────────────────────────────────
INGEST_SOURCES = [
    {
        "url": "https://en.wikipedia.org/wiki/Large_language_model",
        "label": "Wikipedia — LLMs"
    },
    {
        "url": "https://en.wikipedia.org/wiki/Vector_database",
        "label": "Wikipedia — Vector DBs"
    },
    {
        "url": "https://news.ycombinator.com",
        "label": "Hacker News front page"
    },
    {
        "url": "https://paulgraham.com/greatwork.html",
        "label": "Paul Graham — Great Work essay"
    },
    {
        "url": "https://arxiv.org/abs/2005.11401",
        "label": "Original RAG paper (arxiv)"
    },
]

# ─── Test Queries ─────────────────────────────────────────────
# These test all three routes: direct, vector, web
TEST_QUERIES = [
    # should route → vector (ingested content)
    {
        "question": "What are the main components of a large language model?",
        "expected_route": "vector"
    },
    {
        "question": "What is a vector database used for?",
        "expected_route": "vector"
    },
    {
        "question": "What does Paul Graham say about great work?",
        "expected_route": "vector"
    },
    {
        "question": "What is the original RAG paper about?",
        "expected_route": "vector"
    },
    # should route → direct (general knowledge)
    {
        "question": "What is the difference between supervised and unsupervised learning?",
        "expected_route": "direct"
    },
    {
        "question": "What is Python's GIL?",
        "expected_route": "direct"
    },
    # should route → web (live/current data)
    {
        "question": "What are the top stories on Hacker News today?",
        "expected_route": "web"
    },
    {
        "question": "What is the latest news in AI this week?",
        "expected_route": "web"
    },
]

def print_divider(label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print('='*60)

def test_ingestion():
    print_divider("INGESTION TESTS")
    results = []

    for source in INGEST_SOURCES:
        print(f"\n→ Ingesting: {source['label']}")
        print(f"  URL: {source['url']}")
        try:
            start = time.time()
            result = ingest(source["url"])
            elapsed = time.time() - start
            print(f"  ✅ {result['ingested']} chunks in {elapsed:.1f}s")
            results.append({
                "label": source["label"],
                "chunks": result["ingested"],
                "time": elapsed,
                "status": "ok"
            })
        except Exception as e:
            print(f"  ❌ Failed: {str(e)[:80]}")
            results.append({
                "label": source["label"],
                "chunks": 0,
                "time": 0,
                "status": f"error: {str(e)[:60]}"
            })
        time.sleep(2)  # be polite between requests

    return results

def test_queries():
    print_divider("QUERY TESTS")
    results = []

    for test in TEST_QUERIES:
        question = test["question"]
        expected = test["expected_route"]
        print(f"\n→ Q: {question}")
        print(f"  Expected route: {expected}")

        try:
            start = time.time()
            result = query(question)
            elapsed = time.time() - start
            actual_route = result.get("route", "unknown")
            routed_correctly = actual_route == expected

            print(f"  Actual route:   {actual_route} {'✅' if routed_correctly else '⚠️ MISMATCH'}")
            print(f"  Time: {elapsed:.1f}s")
            print(f"  Answer preview: {result['answer'][:200]}...")
            if result.get("sources"):
                print(f"  Sources: {result['sources'][:2]}")

            results.append({
                "question": question,
                "expected_route": expected,
                "actual_route": actual_route,
                "correct": routed_correctly,
                "time": elapsed,
                "answer_length": len(result["answer"])
            })
        except Exception as e:
            print(f"  ❌ Error: {str(e)[:80]}")
            results.append({
                "question": question,
                "expected_route": expected,
                "actual_route": "error",
                "correct": False,
                "time": 0,
                "answer_length": 0
            })

        time.sleep(3)

    return results

def print_summary(ingest_results, query_results):
    print_divider("SUMMARY")

    print("\n── Ingestion ──")
    for r in ingest_results:
        status = "✅" if r["status"] == "ok" else "❌"
        print(f"  {status} {r['label']}: {r['chunks']} chunks ({r['time']:.1f}s)")

    total_chunks = sum(r["chunks"] for r in ingest_results)
    print(f"\n  Total chunks ingested: {total_chunks}")

    print("\n── Routing accuracy ──")
    correct = sum(1 for r in query_results if r["correct"])
    total = len(query_results)
    print(f"  {correct}/{total} queries routed correctly ({correct/total*100:.0f}%)")

    mismatches = [r for r in query_results if not r["correct"]]
    if mismatches:
        print("\n  Mismatches:")
        for r in mismatches:
            print(f"    - '{r['question'][:50]}...'")
            print(f"      Expected: {r['expected_route']} | Got: {r['actual_route']}")

    avg_time = sum(r["time"] for r in query_results if r["time"] > 0)
    avg_time /= max(len([r for r in query_results if r["time"] > 0]), 1)
    print(f"\n  Avg query time: {avg_time:.1f}s")

if __name__ == "__main__":
    print("\n🔍 RAG Pipeline Capability Test")
    print("This will ingest 5 sources and run 8 queries.\n")

    ingest_results = test_ingestion()
    query_results = test_queries()
    print_summary(ingest_results, query_results)