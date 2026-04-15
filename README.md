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
Project Structure
text
agentrag/
├── data/
│   └── documents/          # Raw PDF repository
├── src/
│   ├── ingestion/          # Loader, Chunker, and Embedder logic
│   ├── retrieval/          # Hybrid search and Jina reranker
│   ├── agent/              # LangGraph state and graph orchestration
│   ├── evaluation/         # RAGAS evaluation suite
│   └── api/                # FastAPI backend implementation
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
Use code with caution.
Installation and Setup
1. Clone the Repository
bash
git clone https://github.com
cd AgentRAG
Use code with caution.
2. Environment Configuration
Create a .env file in the root directory and add the following:
text
GROQ_API_KEY=your_groq_api_key
JINA_API_KEY=your_jina_api_key
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=agentrag_docs
Use code with caution.
3. Deployment via Docker
bash
docker-compose up --build
Use code with caution.
4. Document Ingestion
In a new terminal, run the ingestion script to process and index your documents:
bash
python ingest.py
Use code with caution.
API Reference
Query Endpoint
POST /query
Request Body:
json
{
  "query": "Compare the architecture of Inception Net with traditional CNNs."
}
Use code with caution.
Response Example:
json
{
  "answer": "The architecture of Inception Net differs from traditional CNNs in several ways. While traditional CNNs typically use stacked convolutional layers, Inception Net utilizes parallel 'Inception modules' that capture features at multiple scales simultaneously.",
  "sources": [
    "Inception_Net.pdf (p.4)",
    "CNN_Overview.pdf (p.12)"
  ],
  "confidence": 0.94,
  "was_multihop": true
}
Use code with caution.
Evaluation
To execute the automated RAGAS evaluation pipeline and measure system performance:
bash
python -m src.evaluation.ragas_eval
Use code with caution.
