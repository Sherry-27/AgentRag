# AgentRAG: Multi-hop Document Intelligence System

AgentRAG is a high-performance agentic RAG (Retrieval-Augmented Generation) pipeline designed to handle complex, cross-document queries. It leverages graph-based orchestration to decompose multi-hop questions, perform hybrid retrieval, and generate accurate, citation-backed responses.

---

## Core Features

* **Agentic Orchestration**: Powered by LangGraph with conditional routing for single-hop vs multi-hop query handling
* **Hybrid Retrieval Pipeline**: Combines BM25 sparse search with dense embeddings (BGE-small)
* **Neural Reranking**: Uses Jina AI Reranker to refine retrieved context
* **Multi-hop Reasoning**: Decomposes complex queries into sub-questions and synthesizes final answers
* **Self-Reflection Logic**: Automatically re-retrieves context when information is insufficient
* **Automated Evaluation**: Integrated with RAGAS for Faithfulness, Precision, and Recall
* **Production Ready**: FastAPI backend with Docker-based deployment

---

## Technology Stack

* **Orchestration**: LangChain, LangGraph
* **LLM**: Groq (Llama 3.3 70B)
* **Vector Database**: Qdrant
* **Embeddings**: FastEmbed (BAAI/bge-small-en-v1.5)
* **API Layer**: FastAPI, Uvicorn
* **Evaluation**: RAGAS, HuggingFace Datasets
* **Infrastructure**: Docker, Docker Compose

---

## Performance Metrics

Benchmarked using the RAGAS framework on a technical knowledge base:

| Metric            | Score |
| ----------------- | ----- |
| Context Precision | 0.94  |
| Faithfulness      | 0.91  |
| Context Recall    | 0.89  |

---

## Project Structure

```
agentrag/
├── data/
│   └── documents/
├── src/
│   ├── ingestion/
│   ├── retrieval/
│   ├── agent/
│   ├── evaluation/
│   └── api/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Installation and Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-username/AgentRAG.git
cd AgentRAG
```

### 2. Configure Environment

Create `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
JINA_API_KEY=your_jina_api_key
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=agentrag_docs
```

### 3. Run with Docker

```bash
docker-compose up --build
```

### 4. Ingest Documents

```bash
python ingest.py
```

---

## API Reference

### POST `/query`

#### Request

```json
{
  "query": "Compare the architecture of Inception Net with traditional CNNs."
}
```

#### Response

```json
{
  "answer": "...",
  "sources": ["doc.pdf (p.4)"],
  "confidence": 0.94,
  "was_multihop": true
}
```

---

## Evaluation

```bash
python -m src.evaluation.ragas_eval
```

---

## Notes

* Ensure Docker is running before deployment
* Configure `.env` correctly
* Add documents to `data/documents/` before ingestion

---

## License

MIT License
