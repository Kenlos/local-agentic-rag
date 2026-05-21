# eval/ragas_runner.py
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- question set ---
# These are the questions RAGAs will use to evaluate both pipelines.
# Add more as you ingest more content into your knowledge base.
EVAL_QUESTIONS = [
    "How does RAG reduce hallucinations?",
    "What is the difference between RAG and fine-tuning?",
    "What are the limitations of RAG systems?",
]

GROUND_TRUTHS = [
    "RAG reduces hallucinations by grounding LLM responses in retrieved external documents rather than relying solely on parametric memory.",
    "RAG retrieves external documents at inference time without modifying model weights, while fine-tuning updates the model's parameters on domain-specific data.",
    "RAG limitations include poor retrieval quality, context window constraints, inability to fully eliminate hallucinations, and difficulty handling conflicting sources.",
]

def _safe_llm_call(fn, retries: int = 3, delay: float = 5.0):
    """Retries an LLM call with a delay between attempts."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            print(f"  [retry {attempt + 1}/{retries}] {str(e)[:80]}")
            time.sleep(delay)
    return None

def _get_ragas_llm():
    """Returns a LangchainLLMWrapper pointing at LM Studio."""
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        base_url=os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
        api_key="lm-studio",
        model=os.getenv("GENERATION_MODEL", "qwen/qwen3.6-35b-a3b"),
        temperature=0.0,
    )
    return LangchainLLMWrapper(llm)


def _get_ragas_embeddings():
    """Returns embeddings wrapper pointing at LM Studio."""
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(
        openai_api_base=os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
        openai_api_key="lm-studio",
        model=os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5"),
        check_embedding_ctx_length=False,  # skip context length validation
    )
    return LangchainEmbeddingsWrapper(embeddings)


def _run_enhanced_pipeline(questions: list[str]) -> tuple[list[str], list[list[str]]]:
    from retrieval.embedder import embed
    from retrieval.vector_store import dense_search
    from retrieval.bm25 import sparse_search
    from retrieval.hybrid import reciprocal_rank_fusion
    from reranker import rerank
    from query_expansion import expand_query
    from generator import generate

    answers, contexts = [], []

    for i, question in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {question[:60]}...")
        time.sleep(2)  # give LM Studio breathing room between calls

        try:
            queries = expand_query(question)
            all_dense, all_sparse = [], []
            for q in queries:
                q_emb = embed([q])[0]
                all_dense.extend(dense_search(q_emb))
                all_sparse.extend(sparse_search(q))

            seen = set()
            dense_deduped, sparse_deduped = [], []
            for doc in all_dense:
                if doc["content"] not in seen:
                    dense_deduped.append(doc)
                    seen.add(doc["content"])
            for doc in all_sparse:
                if doc["content"] not in seen:
                    sparse_deduped.append(doc)
                    seen.add(doc["content"])

            fused = reciprocal_rank_fusion(dense_deduped, sparse_deduped)
            reranked = rerank(question, fused)
            result = generate(question, reranked)

            answers.append(result["answer"])
            contexts.append([c["content"] for c in reranked])

        except Exception as e:
            print(f"  Error on question {i+1}: {e}")
            answers.append("")
            contexts.append([""])

    return answers, contexts


def _run_naive_pipeline(questions: list[str]) -> tuple[list[str], list[list[str]]]:
    from retrieval.embedder import embed
    from retrieval.vector_store import dense_search
    from generator import generate
    from config import TOP_K_FINAL

    answers, contexts = [], []

    for i, question in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {question[:60]}...")
        time.sleep(2)

        try:
            q_emb = embed([question])[0]
            results = dense_search(q_emb, top_k=TOP_K_FINAL)
            result = generate(question, results)

            answers.append(result["answer"])
            contexts.append([c["content"] for c in results])

        except Exception as e:
            print(f"  Error on question {i+1}: {e}")
            answers.append("")
            contexts.append([""])

    return answers, contexts

def _build_dataset(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> Dataset:
    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

def run_evaluation() -> dict:
    print("\n=== RAGAs Evaluation ===\n")

    ragas_llm = _get_ragas_llm()
    ragas_embeddings = _get_ragas_embeddings()

    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    for m in metrics:
        m.llm = ragas_llm
        if hasattr(m, "embeddings"):
            m.embeddings = ragas_embeddings

    def score_dataset(dataset: Dataset, label: str) -> dict:
        """Scores a dataset one sample at a time to avoid concurrent timeout."""
        all_scores = {m.name: [] for m in metrics}

        for i in range(len(dataset)):
            print(f"  [{label}] scoring sample {i+1}/{len(dataset)}...")
            single = dataset.select([i])

            for metric in metrics:
                time.sleep(3)  # breathing room between each metric call
                try:
                    result = evaluate(single, metrics=[metric])
                    val = result[metric.name]
                    score = val[0] if isinstance(val, list) else float(val)
                    all_scores[metric.name].append(score if score is not None else 0.0)
                except Exception as e:
                    print(f"    [{metric.name}] failed: {str(e)[:60]}")
                    all_scores[metric.name].append(0.0)

        # average across samples
        return {
            k: sum(v) / len(v) if v else 0.0
            for k, v in all_scores.items()
        }

    # --- enhanced pipeline ---
    print("Running enhanced pipeline...")
    enh_answers, enh_contexts = _run_enhanced_pipeline(EVAL_QUESTIONS)
    enh_dataset = _build_dataset(EVAL_QUESTIONS, enh_answers, enh_contexts, GROUND_TRUTHS)

    # --- naive baseline ---
    print("\nRunning naive baseline...")
    naive_answers, naive_contexts = _run_naive_pipeline(EVAL_QUESTIONS)
    naive_dataset = _build_dataset(EVAL_QUESTIONS, naive_answers, naive_contexts, GROUND_TRUTHS)

    # --- score both ---
    print("\nScoring enhanced pipeline...")
    enh_scores = score_dataset(enh_dataset, "enhanced")

    print("\nScoring naive baseline...")
    naive_scores = score_dataset(naive_dataset, "naive")

    # --- print comparison ---
    print("\n--- Results ---")
    print(f"{'Metric':<25} {'Naive':>10} {'Enhanced':>10} {'Delta':>10}")
    print("-" * 58)

    output = {}
    metric_names = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

    for metric in metric_names:
        naive_val = naive_scores.get(metric, 0.0)
        enh_val = enh_scores.get(metric, 0.0)
        delta = enh_val - naive_val
        direction = "↑" if delta > 0 else "↓"
        print(f"{metric:<25} {naive_val:>10.4f} {enh_val:>10.4f} {direction}{abs(delta):>8.4f}")
        output[f"enhanced_{metric}"] = enh_val
        output[f"naive_{metric}"] = naive_val
        output[f"delta_{metric}"] = delta

    return output

if __name__ == "__main__":
    run_evaluation()