from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    SparseVectorParams, SparseIndexParams
)
from fastembed import TextEmbedding
from langchain_core.documents import Document
import uuid
import os


class QdrantEmbedder:
    def __init__(self):
 #       self.client = QdrantClient(url=os.getenv("QDRANT_URL"))
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"), 
            timeout=100  # Yeh timeout error khatam kar dega
        )
        self.collection = os.getenv("QDRANT_COLLECTION")
        # Free local dense embedding — no API key needed
        self.dense_model = TextEmbedding("BAAI/bge-small-en-v1.5")
        self.vector_size = 384

    def create_collection(self):
        """Create Qdrant collection with dense + sparse vectors"""
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config={
                "dense": VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            }
        )
        print(f"Collection '{self.collection}' created")

    def embed_and_upload(self, chunks: list[Document]):
        """Embed chunks and upload to Qdrant in batches"""
        texts = [chunk.page_content for chunk in chunks]
        dense_vectors = list(self.dense_model.embed(texts))

        points = []
        for i, (chunk, dense_vec) in enumerate(zip(chunks, dense_vectors)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": dense_vec.tolist()},
                payload={
                    "text": chunk.page_content,
                    "source": chunk.metadata.get("source", ""),
                    "page": chunk.metadata.get("page", 0),
                    "chunk_id": chunk.metadata.get("chunk_id", i),
                }
            )
            points.append(point)

        # Upload in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(collection_name=self.collection, points=batch)
            print(f"  Uploaded batch {i // batch_size + 1}/{(len(points) - 1) // batch_size + 1}")

        print(f"Uploaded {len(points)} chunks to Qdrant")