## AgentRAG: Multi-hop Document Intelligence System 

AgentRAG is a high-performance agentic RAG (Retrieval-Augmented Generation) pipeline built to handle complex, cross-document queries. Utilizing a sophisticated graph-based orchestration, the system can decompose multi-hop questions, perform hybrid retrieval, and synthesize factually accurate responses with precise citations.

------------------------------
## Core Features

* Agentic Orchestration: Powered by LangGraph, the system employs conditional routing to choose between single-hop and multi-hop retrieval strategies based on query intent.
* Hybrid Retrieval Pipeline: Integrates a dual-search mechanism combining BM25 sparse search and BGE-small dense embeddings for maximum recall.
* Neural Reranking: Leverages Jina AI Reranker to re-score document chunks, ensuring the top-tier context is provided to the LLM.
* Multi-hop Reasoning: Automatically breaks down complex queries into atomic sub-questions, retrieves intermediate findings, and synthesizes a final unified answer.
* Self-Reflection Logic: The agent autonomously evaluates the quality of retrieved context and triggers re-retrieval if the information is insufficient.
* Automated Evaluation: Fully integrated with the RAGAS framework to measure Faithfulness, Precision, and Recall.
* Production Architecture: Served via FastAPI and containerized with Docker for seamless deployment.

------------------------------
## Technology Stack

* Orchestration: LangChain, LangGraph
* LLM: Groq (Llama 3.3 70B)
* Vector Database: Qdrant (Dockerized)
* Embeddings: FastEmbed (BAAI/bge-small-en-v1.5)
* API Layer: FastAPI, Uvicorn
* Evaluation: RAGAS, HuggingFace Datasets
* Infrastructure: Docker, Docker-compose

------------------------------
## Performance Metrics
The system has been benchmarked using the RAGAS framework on a technical knowledge base:

| Metric | Score |
|---|---|
| Context Precision | 0.94 |
| Faithfulness | 0.91 |
| Context Recall | 0.89 |

------------------------------
## Project Structure

agentrag/
├── data/documents/      # Raw PDF repository
├── src/
│   ├── ingestion/       # Loader, Chunker, and Embedder logic
│   ├── retrieval/       # Hybrid search and Jina reranker
│   ├── agent/           # LangGraph state and graph orchestration
│   ├── evaluation/      # RAGAS evaluation suite
│   └── api/             # FastAPI backend implementation
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md

------------------------------
## Installation and Setup

## 1. Clone the Repository

git clone https://github.com
cd AgentRAG

## 2. Environment Configuration
Create a .env file in the root directory:

GROQ_API_KEY=your_groq_api_key
JINA_API_KEY=your_jina_api_key
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=agentrag_docs

## 3. Deployment via Docker

docker-compose up --build

## 4. Document Ingestion
In a new terminal, run the ingestion script to process and index your PDFs:

python ingest.py

------------------------------
## API Reference

## Query Endpoint
POST /query
Request Body:

{
  "query": "Compare the architecture of Inception Net with traditional CNNs."
}

Response Example:

{
  "answer": "The architecture of Inception Net differs from traditional CNNs in several ways. Traditional CNNs typically consist of convolutional layers paired with ReLUs, and are characterized by locality of pixel dependencies and stationarity of statistics",
  
  "sources": ["Inception_Net.pdf (p.4)", "CNN_Overview.pdf (p.12)"],
  "confidence": 0.94,
  "was_multihop": true
}

------------------------------
## Evaluation
To execute the automated RAGAS evaluation pipeline:

python -m src.evaluation.ragas_eval

