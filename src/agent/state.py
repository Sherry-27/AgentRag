from typing import TypedDict, Annotated, List
import operator


class AgentState(TypedDict):
    """
    Complete state that flows through the LangGraph agent.
    Every node reads from and writes to this state.
    """
    query: str                                              # Original user question
    refined_query: str                                      # Rewritten for better retrieval
    retrieved_docs: List[dict]                              # Raw retrieved docs
    reranked_docs: List[dict]                               # After Jina reranking
    reasoning_steps: Annotated[List[str], operator.add]    # Agent thought log (appended)
    sub_queries: List[str]                                  # Multi-hop sub-questions
    intermediate_answers: List[str]                         # Answers per sub-query
    final_answer: str                                       # Final response to user
    sources: List[str]                                      # Source citations
    needs_multihop: bool                                    # Routing decision
    confidence_score: float                                 # Answer confidence 0-1