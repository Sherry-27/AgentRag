# ingest.py
from dotenv import load_dotenv
load_dotenv()
from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import SemanticChunker
from src.ingestion.embedder import QdrantEmbedder
def main():
    # 1. Load documents
    loader = DocumentLoader("data/documents")
    docs = loader.load_pdfs()
    # 2. Chunk semantically
    chunker = SemanticChunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk(docs)
    # 3. Embed and upload
    embedder = QdrantEmbedder()
    embedder.create_collection()
    embedder.embed_and_upload(chunks)
    print(" Ingestion complete!")
if __name__ == "__main__":
    main()