from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi
import os
import numpy as np


class HybridRetriever:
    def __init__(self, top_k=10):
        from dotenv import load_dotenv
        load_dotenv()
        self.client = QdrantClient(url=os.getenv("QDRANT_URL"), timeout=100)
        self.collection = os.getenv("QDRANT_COLLECTION") or "agentrag_docs"
        self.dense_model = TextEmbedding("BAAI/bge-small-en-v1.5")
        self.top_k = top_k
        self._all_texts = None
        self._bm25 = None

    def _build_bm25(self):
        """Build BM25 index from all Qdrant documents"""
        if self._bm25 is not None:
            return
        results, _ = self.client.scroll(
            collection_name=self.collection,
            limit=10000,
            with_payload=True
        )
        self._all_texts = [r.payload["text"] for r in results]
        self._all_ids = [r.id for r in results]
        tokenized = [text.lower().split() for text in self._all_texts]
        self._bm25 = BM25Okapi(tokenized)
        print(f"BM25 index built on {len(self._all_texts)} chunks")

    def dense_search(self, query: str) -> list[dict]:
        query_vector = list(self.dense_model.embed([query]))[0]
    
    
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist(), # Direct vector pass karein
            using="dense",               # Vector ka naam yahan batayein
            limit=self.top_k,
            with_payload=True
        ).points # .points lagana zaroori hai response nikalne ke liye

        return [{
            "text": r.payload["text"],
            "score": r.score if hasattr(r, 'score') else 0,
            "source": r.payload.get("source", ""),
            "page": r.payload.get("page", 0)
        } for r in results]


    def sparse_search(self, query: str) -> list[dict]:
        """BM25 sparse keyword search"""
        self._build_bm25()
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:self.top_k]
        return [{
            "text": self._all_texts[i],
            "score": float(scores[i]),
            "source": "",
            "page": 0
        } for i in top_indices if scores[i] > 0]

    def hybrid_search(self, query: str, alpha=0.7) -> list[dict]:
        """
        Combine dense + sparse using RRF (Reciprocal Rank Fusion)
        alpha=0.7 means 70% dense, 30% sparse weight
        Higher alpha = more semantic, lower = more keyword match
        """
        dense_results = self.dense_search(query)
        sparse_results = self.sparse_search(query)

        rrf_scores = {}
        k = 60  # RRF constant — standard value

        for rank, result in enumerate(dense_results):
            key = result["text"][:100]
            if key not in rrf_scores:
                rrf_scores[key] = {"result": result, "score": 0}
            rrf_scores[key]["score"] += alpha * (1 / (k + rank + 1))

        for rank, result in enumerate(sparse_results):
            key = result["text"][:100]
            if key not in rrf_scores:
                rrf_scores[key] = {"result": result, "score": 0}
            rrf_scores[key]["score"] += (1 - alpha) * (1 / (k + rank + 1))

        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        return [r["result"] for r in sorted_results[:self.top_k]]