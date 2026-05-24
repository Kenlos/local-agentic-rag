# test_comprehensive.py
import time
import json
from datetime import datetime
from pipeline import query
from retrieval.embedder import embed
from retrieval.vector_store import dense_search
from generator import generate

# ─── Test Questions ────────────────────────────────────────────────────────────
# 20 questions across 4 categories.
# Each has an expected_route so we can measure routing accuracy.
# ground_truth is used to score answer quality without a second LLM call.

TEST_CASES = [
    # ── Category 1: AI / ML (vector retrieval expected) ──────────────────────
    {
        "id": 1,
        "category": "AI / ML",
        "question": "What are the main components of a RAG pipeline?",
        "expected_route": "vector",
        "ground_truth": "A RAG pipeline consists of an ingestion layer that processes documents into chunks, an embedding model that converts text to vectors, a vector store for retrieval, a retrieval stage combining dense and sparse search, a reranker, and a generation model.",
        "route_type": "vector"
    },
    {
        "id": 2,
        "category": "AI / ML",
        "question": "How do large language models handle context windows?",
        "expected_route": "vector",
        "ground_truth": "Large language models process text within a fixed context window measured in tokens. Content beyond this limit is truncated. Longer context windows allow more information but increase computational cost.",
        "route_type": "vector"
    },
    {
        "id": 3,
        "category": "AI / ML",
        "question": "What is the difference between dense and sparse retrieval in vector databases?",
        "expected_route": "vector",
        "ground_truth": "Dense retrieval uses neural embeddings and cosine similarity to find semantically similar content. Sparse retrieval uses keyword matching methods like BM25. Hybrid approaches combine both for better coverage.",
        "route_type": "vector"
    },
    {
        "id": 4,
        "category": "AI / ML",
        "question": "What are hallucinations in AI models and why do they occur?",
        "expected_route": "vector",
        "ground_truth": "Hallucinations occur when AI models generate plausible-sounding but incorrect information. They arise because models predict probable token sequences rather than retrieving verified facts, especially when training data is insufficient or ambiguous.",
        "route_type": "vector"
    },
    {
        "id": 5,
        "category": "AI / ML",
        "question": "What is HNSW indexing and why is it used in vector databases?",
        "expected_route": "vector",
        "ground_truth": "HNSW (Hierarchical Navigable Small World) is a graph-based approximate nearest neighbor index. It builds a layered graph for fast traversal during similarity search. It is preferred for vector databases because it does not require knowing dataset size upfront.",
        "route_type": "vector"
    },

    # ── Category 2: Software Engineering (vector retrieval expected) ──────────
    {
        "id": 6,
        "category": "Software Engineering",
        "question": "What is PostgreSQL and what makes it different from other databases?",
        "expected_route": "vector",
        "ground_truth": "PostgreSQL is an open-source relational database system known for ACID compliance, extensibility, and support for advanced data types including JSON and vectors via extensions like pgvector.",
        "route_type": "vector"
    },
    {
        "id": 7,
        "category": "Software Engineering",
        "question": "How does Docker containerization work?",
        "expected_route": "vector",
        "ground_truth": "Docker uses OS-level virtualization to package applications and their dependencies into containers. Containers share the host OS kernel but are isolated from each other, making them lightweight compared to virtual machines.",
        "route_type": "vector"
    },
    {
        "id": 8,
        "category": "Software Engineering",
        "question": "What is the difference between a Docker container and a Docker image?",
        "expected_route": "vector",
        "ground_truth": "A Docker image is a read-only template containing the application and its dependencies. A container is a running instance of an image. Multiple containers can run from the same image simultaneously.",
        "route_type": "vector"
    },
    {
        "id": 9,
        "category": "Software Engineering",
        "question": "What are database indexes and when should you use them?",
        "expected_route": "vector",
        "ground_truth": "Database indexes are data structures that speed up query performance by allowing the database to find rows without scanning the entire table. They should be used on columns frequently used in WHERE clauses, joins, and ORDER BY statements.",
        "route_type": "vector"
    },
    {
        "id": 10,
        "category": "Software Engineering",
        "question": "What is the purpose of a Docker Compose file?",
        "expected_route": "vector",
        "ground_truth": "Docker Compose defines and manages multi-container applications using a YAML file. It specifies services, networks, and volumes, allowing complex applications to be started with a single command.",
        "route_type": "vector"
    },

    # ── Category 3: Business / Startups (mixed vector + direct) ──────────────
    {
        "id": 11,
        "category": "Business",
        "question": "What does Paul Graham say about how to find good startup ideas?",
        "expected_route": "vector",
        "ground_truth": "Paul Graham argues that the best startup ideas come from noticing problems in your own life, especially those that seem too obvious or niche. Ideas that seem embarrassingly simple or serve a small market often become large companies.",
        "route_type": "vector"
    },
    {
        "id": 12,
        "category": "Business",
        "question": "What does Paul Graham say about the qualities needed for great work?",
        "expected_route": "vector",
        "ground_truth": "Paul Graham argues that great work requires choosing the right problem, working with deep curiosity and interest, and being willing to follow your natural inclinations even when they seem unconventional.",
        "route_type": "vector"
    },
    {
        "id": 13,
        "category": "Business",
        "question": "What is product-market fit and how do you know when you have it?",
        "expected_route": "direct",
        "ground_truth": "Product-market fit means your product satisfies a strong market demand. Signs include organic growth, high retention, users recommending it without prompting, and struggling to keep up with demand.",
        "route_type": "direct"
    },
    {
        "id": 14,
        "category": "Business",
        "question": "What is the difference between a startup and a small business?",
        "expected_route": "direct",
        "ground_truth": "A startup is designed for rapid growth and scalability, typically seeking venture funding and aiming for a large market. A small business is designed for profitability and sustainability within a defined local or niche market.",
        "route_type": "direct"
    },
    {
        "id": 15,
        "category": "Business",
        "question": "What are the latest AI tools being used by startups in 2026?",
        "expected_route": "web",
        "ground_truth": "Current AI tools for startups include LLM APIs, RAG pipelines, code generation tools, and agent frameworks. The specific tools change rapidly so current sources should be consulted.",
        "route_type": "web"
    },

    # ── Category 4: Science / General (direct + web) ──────────────────────────
    {
        "id": 16,
        "category": "Science",
        "question": "What are the main causes of climate change?",
        "expected_route": "vector",
        "ground_truth": "Climate change is primarily caused by greenhouse gas emissions from burning fossil fuels, deforestation, and industrial processes. These gases trap heat in the atmosphere, raising global temperatures.",
        "route_type": "vector"
    },
    {
        "id": 17,
        "category": "Science",
        "question": "How does quantum computing differ from classical computing?",
        "expected_route": "vector",
        "ground_truth": "Quantum computing uses qubits that can exist in superposition of states simultaneously, unlike classical bits which are either 0 or 1. This enables quantum computers to solve certain problems exponentially faster.",
        "route_type": "vector"
    },
    {
        "id": 18,
        "category": "Science",
        "question": "What is the speed of light and why is it important in physics?",
        "expected_route": "direct",
        "ground_truth": "The speed of light is approximately 299,792,458 meters per second in a vacuum. It is a fundamental constant in physics, central to Einstein's theory of relativity and the definition of causality.",
        "route_type": "direct"
    },
    {
        "id": 19,
        "category": "Science",
        "question": "What are the latest developments in quantum computing in 2026?",
        "expected_route": "web",
        "ground_truth": "Quantum computing is advancing rapidly with improvements in qubit stability and error correction. Current sources should be consulted for the latest specific developments.",
        "route_type": "web"
    },
    {
        "id": 20,
        "category": "Science",
        "question": "What is the Pythagorean theorem?",
        "expected_route": "direct",
        "ground_truth": "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides: a² + b² = c².",
        "route_type": "direct"
    },
]


# ─── Naive Pipeline ────────────────────────────────────────────────────────────

def naive_query(question: str) -> dict:
    """
    Naive baseline — pure dense vector search, no expansion,
    no reranking, no routing, no hybrid retrieval.
    Answers directly from top-5 vector results.
    """
    try:
        q_embedding = embed([question])[0]
        results = dense_search(q_embedding, top_k=5)

        if not results:
            from config import LM_STUDIO_BASE_URL, LM_STUDIO_API_KEY, GENERATION_MODEL
            from openai import OpenAI
            client = OpenAI(
                base_url=LM_STUDIO_BASE_URL,
                api_key=LM_STUDIO_API_KEY
            )
            response = client.chat.completions.create(
                model=GENERATION_MODEL,
                messages=[{"role": "user", "content": question}],
                temperature=0.3
            )
            return {
                "answer": response.choices[0].message.content.strip(),
                "route": "naive-direct",
                "sources": []
            }

        result = generate(question, results)
        result["route"] = "naive-vector"
        return result

    except Exception as e:
        return {
            "answer": f"Error: {e}",
            "route": "naive-error",
            "sources": []
        }


# ─── Quality Scoring ──────────────────────────────────────────────────────────

def score_answer(answer: str, ground_truth: str) -> float:
    """
    Scores answer quality using keyword overlap against ground truth.

    Why not use an LLM for scoring?
    We want to avoid adding more LLM calls to an already slow test.
    Keyword overlap is a fast, deterministic proxy for coverage.
    It is not perfect but it is consistent and costs nothing.

    Score = proportion of ground truth key terms found in the answer.
    """
    if not answer or len(answer) < 20:
        return 0.0

    # failure phrases — clear signal the pipeline couldn't answer
    failure_phrases = [
        "i cannot", "i don't have", "no information",
        "web search failed", "could not retrieve",
        "i don't know", "unable to answer"
    ]
    if any(p in answer.lower() for p in failure_phrases):
        return 0.1

    # extract meaningful words from ground truth (ignore stopwords)
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "can",
        "to", "of", "in", "on", "at", "by", "for", "with", "about",
        "as", "into", "through", "during", "before", "after", "it",
        "its", "this", "that", "these", "those", "and", "or", "but",
        "if", "because", "so", "yet", "both", "either", "not", "no"
    }

    gt_words = set(
        w.lower().strip(".,!?;:")
        for w in ground_truth.split()
        if w.lower() not in stopwords and len(w) > 3
    )

    answer_lower = answer.lower()
    matched = sum(1 for w in gt_words if w in answer_lower)

    return round(matched / len(gt_words), 3) if gt_words else 0.0


# ─── Test Runner ──────────────────────────────────────────────────────────────

def run_tests():
    print(f"\n{'='*70}")
    print(f"  Comprehensive Pipeline Test — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  {len(TEST_CASES)} questions across 4 categories")
    print(f"{'='*70}\n")

    enhanced_results = []
    naive_results = []

    for i, case in enumerate(TEST_CASES):
        print(f"[{i+1:02d}/{len(TEST_CASES)}] {case['category']} — {case['question'][:55]}...")

        # ── Enhanced pipeline ──
        start = time.time()
        enhanced = query(case["question"])
        enhanced_time = time.time() - start
        enhanced_score = score_answer(enhanced.get("answer", ""), case["ground_truth"])
        route_correct = case["expected_route"] in enhanced.get("route", "")

        enhanced_results.append({
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "expected_route": case["expected_route"],
            "actual_route": enhanced.get("route", "unknown"),
            "route_correct": route_correct,
            "quality_score": enhanced_score,
            "time": enhanced_time,
            "answer_length": len(enhanced.get("answer", "")),
            "cache_hit": enhanced.get("cache_hit", False)
        })

        # ── Naive pipeline ──
        start = time.time()
        naive = naive_query(case["question"])
        naive_time = time.time() - start
        naive_score = score_answer(naive.get("answer", ""), case["ground_truth"])

        naive_results.append({
            "id": case["id"],
            "quality_score": naive_score,
            "time": naive_time,
        })

        # progress indicator
        route_symbol = "✅" if route_correct else "⚠️"
        print(f"       Route: {route_symbol} {enhanced.get('route', '?'):<20} "
              f"Quality: enhanced={enhanced_score:.2f} naive={naive_score:.2f} "
              f"Time: {enhanced_time:.0f}s")
        print()

        time.sleep(2)

    return enhanced_results, naive_results


def print_report(enhanced_results, naive_results):
    naive_by_id = {r["id"]: r for r in naive_results}

    print(f"\n{'='*70}")
    print("  RESULTS BY CATEGORY")
    print(f"{'='*70}")

    categories = {}
    for r in enhanced_results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    for cat, results in categories.items():
        enhanced_scores = [r["quality_score"] for r in results]
        naive_scores = [naive_by_id[r["id"]]["quality_score"] for r in results]
        route_acc = sum(1 for r in results if r["route_correct"]) / len(results)

        avg_enhanced = sum(enhanced_scores) / len(enhanced_scores)
        avg_naive = sum(naive_scores) / len(naive_scores)
        delta = avg_enhanced - avg_naive

        print(f"\n  {cat}")
        print(f"  {'─'*40}")
        print(f"  Enhanced quality:  {avg_enhanced:.2f}")
        print(f"  Naive quality:     {avg_naive:.2f}")
        print(f"  Delta:             {'+' if delta >= 0 else ''}{delta:.2f}")
        print(f"  Routing accuracy:  {route_acc*100:.0f}%")

    print(f"\n{'='*70}")
    print("  OVERALL SUMMARY")
    print(f"{'='*70}")

    all_enhanced = [r["quality_score"] for r in enhanced_results]
    all_naive = [naive_by_id[r["id"]]["quality_score"] for r in naive_results]
    all_times = [r["time"] for r in enhanced_results]
    routing_correct = sum(1 for r in enhanced_results if r["route_correct"])
    cache_hits = sum(1 for r in enhanced_results if r["cache_hit"])

    avg_e = sum(all_enhanced) / len(all_enhanced)
    avg_n = sum(all_naive) / len(all_naive)
    delta = avg_e - avg_n
    avg_time = sum(all_times) / len(all_times)

    print(f"\n  Quality scores (keyword coverage vs ground truth):")
    print(f"  Enhanced pipeline:   {avg_e:.3f}  ({avg_e*100:.1f}%)")
    print(f"  Naive baseline:      {avg_n:.3f}  ({avg_n*100:.1f}%)")
    print(f"  Overall delta:       {'+' if delta >= 0 else ''}{delta:.3f}  ({delta*100:+.1f}%)")

    print(f"\n  Routing:")
    print(f"  Correct routes:      {routing_correct}/{len(enhanced_results)} ({routing_correct/len(enhanced_results)*100:.0f}%)")

    print(f"\n  Routing breakdown:")
    route_types = {}
    for r in enhanced_results:
        expected = r["expected_route"]
        correct = r["route_correct"]
        if expected not in route_types:
            route_types[expected] = {"correct": 0, "total": 0}
        route_types[expected]["total"] += 1
        if correct:
            route_types[expected]["correct"] += 1

    for route, stats in sorted(route_types.items()):
        pct = stats["correct"] / stats["total"] * 100
        print(f"    {route:<10} {stats['correct']}/{stats['total']} ({pct:.0f}%)")

    print(f"\n  Performance:")
    print(f"  Avg query time:      {avg_time:.1f}s")
    print(f"  Cache hits:          {cache_hits}/{len(enhanced_results)}")

    print(f"\n  Per-question detail:")
    print(f"  {'ID':<4} {'Category':<22} {'Expected':<10} {'Got':<22} {'E':>5} {'N':>5} {'OK'}")
    print(f"  {'─'*85}")
    for r in enhanced_results:
        n = naive_by_id[r["id"]]
        ok = "✅" if r["route_correct"] else "⚠️ "
        route_display = r["actual_route"][:20]
        print(f"  {r['id']:<4} {r['category']:<22} {r['expected_route']:<10} "
              f"{route_display:<22} {r['quality_score']:>5.2f} {n['quality_score']:>5.2f} {ok}")

    print(f"\n{'='*70}\n")

    # save results to file
    output = {
        "timestamp": datetime.now().isoformat(),
        "enhanced_results": enhanced_results,
        "naive_results": naive_results,
        "summary": {
            "enhanced_avg_quality": avg_e,
            "naive_avg_quality": avg_n,
            "quality_delta": delta,
            "routing_accuracy": routing_correct / len(enhanced_results),
            "avg_query_time": avg_time,
            "cache_hits": cache_hits
        }
    }

    with open("eval/test_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Full results saved to eval/test_results.json")


if __name__ == "__main__":
    enhanced_results, naive_results = run_tests()
    print_report(enhanced_results, naive_results)