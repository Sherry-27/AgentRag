import httpx
import os


class JinaReranker:
    def __init__(self, top_n=5):
        self.api_key = os.getenv("JINA_API_KEY")
        self.top_n = top_n
        self.url = "https://api.jina.ai/v1/rerank"

    def rerank(self, query: str, documents: list[dict]) -> list[dict]:
        """Rerank documents using Jina AI — multilingual model"""
        if not documents:
            return []

        texts = [doc["text"] for doc in documents]

        response = httpx.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": texts,
                "top_n": self.top_n
            },
            timeout=30
        )

        results = response.json()
        reranked = []
        for item in results["results"]:
            doc = documents[item["index"]].copy()
            doc["rerank_score"] = item["relevance_score"]
            reranked.append(doc)

        return reranked