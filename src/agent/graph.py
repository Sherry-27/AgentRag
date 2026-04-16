from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from src.agent.state import AgentState
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import JinaReranker
import os
import json

# ── Initialize Models ────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

retriever = HybridRetriever(top_k=10)
reranker = JinaReranker(top_n=5)


    # ── NODE 1: Query Analysis ————
def analyze_query(state: AgentState) -> AgentState:
    """
    Agent decides:
    1. Simple or multi-hop query?
    2. Rewrites query for better retrieval
    3. Generates sub-queries if multi-hop needed
    4. Estimates confidence needed
    """
    prompt = f"""Analyze this query carefully and determine:

1. Is it simple (single document) or multi-hop (needs multiple sources)?
2. Rewrite the query to be more specific for semantic search.
3. If multi-hop, list 2-3 focused sub-queries.
4. Rate how complex this query is (0.0 to 1.0).

Query: {state['query']}

Respond ONLY in valid JSON — no extra text:
{{
    "needs_multihop": true or false,
    "refined_query": "improved query here",
    "sub_queries": ["sub-query 1", "sub-query 2"] or [],
    "complexity": 0.0 to 1.0,
    "reasoning": "why you made this decision"
}}"""

    response = llm.invoke([HumanMessage(content=prompt)])

    # Clean response — remove markdown fences if present
    content = response.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    content = content.strip()

    result = json.loads(content)

    return {
        **state,
        "needs_multihop": result["needs_multihop"],
        "refined_query": result["refined_query"],
        "sub_queries": result.get("sub_queries", []),
        "confidence_score": result.get("complexity", 0.5),
        "reasoning_steps": [
            f"ANALYSIS: {'Multi-hop' if result['needs_multihop'] else 'Single-hop'} query detected",
            f"REFINED: {result['refined_query']}",
            f"REASONING: {result.get('reasoning', 'N/A')}"
        ]
    }


# ── NODE 2: Single-hop Retrieval ─────────────────────────────────
def single_hop_retrieve(state: AgentState) -> AgentState:
    """Fast retrieval for simple queries"""
    query = state["refined_query"]

    docs = retriever.hybrid_search(query)
    reranked = reranker.rerank(query, docs)

    return {
        **state,
        "retrieved_docs": docs,
        "reranked_docs": reranked,
        "reasoning_steps": [
            f"RETRIEVAL: Got {len(docs)} docs via hybrid search",
            f"RERANKING: Top {len(reranked)} after Jina rerank"
        ]
    }


# ── NODE 3: Multi-hop Retrieval ──────────────────────────────────
def multi_hop_retrieve(state: AgentState) -> AgentState:
    """
    Multi-hop: retrieve for each sub-query separately,
    get intermediate answers, then synthesize.
    Hammad's system cannot do this.
    """
    sub_queries = state["sub_queries"]
    all_docs = []
    intermediate_answers = []

    for i, sub_query in enumerate(sub_queries):
        # Retrieve for this specific sub-query
        docs = retriever.hybrid_search(sub_query)
        reranked = reranker.rerank(sub_query, docs)
        all_docs.extend(reranked)

        # Get intermediate answer for this sub-query
        context = "\n\n".join([d["text"] for d in reranked[:3]])
        prompt = f"""Answer this specific sub-question based ONLY on the context.
Be brief and factual.

Sub-question: {sub_query}
Context: {context}

Answer:"""
        response = llm.invoke([HumanMessage(content=prompt)])
        intermediate_answers.append(f"[Sub-query {i+1}] {response.content}")

    # Deduplicate docs by text content
    seen = set()
    unique_docs = []
    for doc in all_docs:
        key = doc["text"][:100]
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    return {
        **state,
        "retrieved_docs": unique_docs,
        "reranked_docs": unique_docs[:5],
        "intermediate_answers": intermediate_answers,
        "reasoning_steps": [
            f"MULTI-HOP: Executed {len(sub_queries)} sub-queries",
            f"INTERMEDIATE: {intermediate_answers}",
            f"DEDUP: {len(unique_docs)} unique docs after deduplication"
        ]
    }


# ── NODE 4: Self-Reflection ──────────────────────────────────────
def self_reflect(state: AgentState) -> AgentState:
    """
    Agent checks if retrieved docs actually answer the query.
    If not enough context, it refines and retrieves again.
    This is what makes AgentRAG genuinely agentic.
    """
    context_docs = state["reranked_docs"]
    if not context_docs:
        return {
            **state,
            "reasoning_steps": ["REFLECTION: No docs found, cannot answer"]
        }

    context_preview = "\n".join([d["text"][:200] for d in context_docs[:3]])

    prompt = f"""You retrieved these documents for this query.
Assess if they contain enough information to answer well.

Query: {state['query']}
Retrieved context preview:
{context_preview}

Respond ONLY in valid JSON:
{{
    "sufficient": true or false,
    "missing": "what information is missing if any",
    "confidence": 0.0 to 1.0
}}"""

    response = llm.invoke([HumanMessage(content=prompt)])

    content = response.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]

    result = json.loads(content.strip())

    # If not sufficient, try one more retrieval with different query
    if not result["sufficient"] and result.get("missing"):
        extra_docs = retriever.hybrid_search(result["missing"])
        extra_reranked = reranker.rerank(result["missing"], extra_docs)
        combined = state["reranked_docs"] + extra_reranked
        # Deduplicate
        seen = set()
        unique = []
        for doc in combined:
            key = doc["text"][:100]
            if key not in seen:
                seen.add(key)
                unique.append(doc)

        return {
            **state,
            "reranked_docs": unique[:6],
            "confidence_score": result["confidence"],
            "reasoning_steps": [
                f"REFLECTION: Insufficient context detected",
                f"MISSING: {result['missing']}",
                f"RE-RETRIEVAL: Added {len(extra_reranked)} more docs",
                f"CONFIDENCE: {result['confidence']}"
            ]
        }

    return {
        **state,
        "confidence_score": result["confidence"],
        "reasoning_steps": [
            f"REFLECTION: Context sufficient",
            f"CONFIDENCE: {result['confidence']}"
        ]
    }


# ── NODE 5: Generate Final Answer ────────────────────────────────
def generate_answer(state: AgentState) -> AgentState:
    """Generate final answer with source citations and confidence"""
    context_docs = state["reranked_docs"]

    context = "\n\n---\n\n".join([
        f"[Source: {doc.get('source', 'Unknown')}, Page {doc.get('page', '?')}]\n{doc['text']}"
        for doc in context_docs
    ])

    # Include intermediate answers for multi-hop
    intermediate = ""
    if state.get("intermediate_answers"):
        intermediate = "\n\nIntermediate findings from sub-queries:\n" + "\n".join(
            state["intermediate_answers"]
        )

    prompt = f"""You are an intelligent document assistant.
Answer based ONLY on the provided context. Be thorough, accurate, and cite sources.

Original Question: {state['query']}
{intermediate}

Context:
{context}

Instructions:
- Answer completely using the context
- Cite sources as [Source: filename, Page X]
- If context is insufficient, say so clearly
- Be concise but complete"""

    response = llm.invoke([HumanMessage(content=prompt)])

    sources = list(set([
        f"{doc.get('source', 'Unknown')} (p.{doc.get('page', '?')})"
        for doc in context_docs
    ]))

    return {
        **state,
        "final_answer": response.content,
        "sources": sources,
        "reasoning_steps": [
            f"ANSWER: Generated from {len(context_docs)} sources",
            f"CONFIDENCE: {state.get('confidence_score', 0.0):.2f}"
        ]
    }


# ── ROUTING FUNCTIONS ────────────────────────────────────────────
def route_query(state: AgentState) -> str:
    """Router after analysis: simple or multi-hop?"""
    if state["needs_multihop"]:
        return "multi_hop"
    return "single_hop"


# ── BUILD THE GRAPH ──────────────────────────────────────────────
def build_agent_graph():
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("single_hop_retrieve", single_hop_retrieve)
    graph.add_node("multi_hop_retrieve", multi_hop_retrieve)
    graph.add_node("self_reflect", self_reflect)        # Hammad ke paas nahi hai
    graph.add_node("generate_answer", generate_answer)

    # Entry point
    graph.set_entry_point("analyze_query")

    # Conditional routing after analysis
    graph.add_conditional_edges(
        "analyze_query",
        route_query,
        {
            "single_hop": "single_hop_retrieve",
            "multi_hop": "multi_hop_retrieve"
        }
    )

    # Both paths go through self-reflection before answering
    graph.add_edge("single_hop_retrieve", "self_reflect")
    graph.add_edge("multi_hop_retrieve", "self_reflect")
    graph.add_edge("self_reflect", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


# ── GLOBAL AGENT INSTANCE ────────────────────────────────────────
agent = build_agent_graph()


def run_agent(query: str) -> dict:
    """Run the full agent pipeline"""
    initial_state = AgentState(
        query=query,
        refined_query="",
        retrieved_docs=[],
        reranked_docs=[],
        reasoning_steps=[],
        sub_queries=[],
        intermediate_answers=[],
        final_answer="",
        sources=[],
        needs_multihop=False,
        confidence_score=0.0
    )

    result = agent.invoke(initial_state)

    return {
        "answer": result["final_answer"],
        "sources": result["sources"],
        "reasoning_trace": result["reasoning_steps"],
        "was_multihop": result["needs_multihop"],
        "docs_retrieved": len(result["retrieved_docs"]),
        "confidence": result["confidence_score"],
        "sub_queries_used": result.get("sub_queries", [])
    }