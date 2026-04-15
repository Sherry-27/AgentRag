from langchain_core.tools import tool
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import JinaReranker
retriever = HybridRetriever(top_k=10)
reranker = JinaReranker(top_n=5)
@tool
def hybrid_retrieve(query: str) -> list[dict]:
    """
    Retrieve relevant documents using hybrid search (BM25 + dense embeddings).
    Use this to find relevant context for answering questions.
    """
    results = retriever.hybrid_search(query)
    return results
@tool
def rerank_documents(query: str, documents: list[dict]) -> list[dict]:
    """
    Rerank retrieved documents by relevance using Jina reranker.
    Always use this after hybrid_retrieve for better precision.
    """
    return reranker.rerank(query, documents)
@tool
def decompose_query(query: str) -> list[str]:
    """
    Decompose a complex multi-hop query into simpler sub-queries.
    Use this when the question requires information from multiple sources.
    Example: 'Compare X from paper A with Y from paper B' → [query for X, query for Y]
    """
    # This is handled by the LLM itself in the graph
    # Tool signals to agent that decomposition happened
    return [query]