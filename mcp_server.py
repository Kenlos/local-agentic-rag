# mcp_server.py
from mcp.server.fastmcp import FastMCP  # instead of from fastmcp import FastMCP
import pipeline

mcp = FastMCP(
    name="Local Agentic RAG Pipeline",
    instructions="""
    A local RAG pipeline with hybrid retrieval, reranking, and query expansion.
    Use query_knowledge_base for questions about ingested content.
    Use ingest_url to add web pages to the knowledge base.
    Use ingest_file to add local files to the knowledge base.
    Use get_sources to retrieve raw source chunks without generating an answer.
    Use evaluate_pipeline to run the RAGAs evaluation suite.
    """
)

@mcp.tool()
def query_knowledge_base(question: str) -> str:
    """
    Run the full RAG pipeline for a question.
    Includes query expansion, hybrid retrieval, reranking, and generation.
    Returns a grounded answer with cited sources.
    """
    try:
        result = pipeline.query(question)
        answer = result["answer"]
        sources = result.get("sources", [])
        route = result.get("route", "unknown")

        return f"""Answer: {answer}

Route used: {route}
Sources: {', '.join(sources) if sources else 'None'}"""

    except Exception as e:
        return f"Error running pipeline: {str(e)}"


@mcp.tool()
def ingest_url(url: str) -> str:
    """
    Scrape a URL and add its content to the knowledge base.
    Accepts any publicly accessible web page.
    """
    try:
        result = pipeline.ingest(url)
        return f"Successfully ingested {result['ingested']} chunks from {result['source']}"
    except Exception as e:
        return f"Error ingesting URL: {str(e)}"


@mcp.tool()
def ingest_file(path: str) -> str:
    """
    Ingest a local file into the knowledge base.
    Supports .txt, .md, .pdf, and .docx files.
    """
    try:
        result = pipeline.ingest(path)
        return f"Successfully ingested {result['ingested']} chunks from {result['source']}"
    except Exception as e:
        return f"Error ingesting file: {str(e)}"


@mcp.tool()
def get_sources(question: str) -> str:
    """
    Retrieve the raw source chunks for a question without generating an answer.
    Useful for inspecting what the knowledge base contains on a topic.
    """
    try:
        from retrieval.embedder import embed
        from retrieval.vector_store import dense_search
        from retrieval.bm25 import sparse_search
        from retrieval.hybrid import reciprocal_rank_fusion
        from reranker import rerank

        q_embedding = embed([question])[0]
        dense = dense_search(q_embedding)
        sparse = sparse_search(question)
        fused = reciprocal_rank_fusion(dense, sparse)
        reranked = rerank(question, fused)

        if not reranked:
            return "No relevant sources found in the knowledge base."

        output = []
        for i, chunk in enumerate(reranked):
            output.append(
                f"[{i+1}] Source: {chunk['source']}\n"
                f"    Score: {chunk.get('rerank_score', 0):.4f}\n"
                f"    Preview: {chunk['content'][:200]}..."
            )

        return "\n\n".join(output)

    except Exception as e:
        return f"Error retrieving sources: {str(e)}"


@mcp.tool()
def evaluate_pipeline() -> str:
    """
    Run the RAGAs evaluation suite against the pipeline.
    Returns retrieval recall, answer faithfulness, and other metrics.
    """
    try:
        from eval.ragas_runner import run_evaluation
        metrics = run_evaluation()
        return "\n".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    except Exception as e:
        return f"Evaluation failed: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")