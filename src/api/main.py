from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.agent.graph import run_agent
from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import SemanticChunker
from src.ingestion.embedder import QdrantEmbedder
from dotenv import load_dotenv
import tempfile, shutil, os

load_dotenv()

app = FastAPI(title="AgentRAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    reasoning_trace: list[str]
    was_multihop: bool
    docs_retrieved: int
    confidence: float
    sub_queries_used: list[str]


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Main query endpoint — runs full agentic pipeline"""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        result = run_agent(request.query)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a new PDF document"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        loader = DocumentLoader(tmpdir)
        docs = loader.load_pdfs()
        chunker = SemanticChunker()
        chunks = chunker.chunk(docs)
        embedder = QdrantEmbedder()
        embedder.embed_and_upload(chunks)

    return {
        "message": f"Uploaded and ingested {file.filename}",
        "chunks_created": len(chunks)
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "2.0.0"}