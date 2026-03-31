from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Optional
from typing import TypedDict

DEFAULT_SYSTEM_PROMPT = (
    "You answer the user's question concisely. "
    "When context passages are provided below, base your answer only on them; "
    "if the context is empty or insufficient, say so briefly and answer from general knowledge."
)


class EvalAgentState(TypedDict, total=False):
    question: str
    retrieval_context: str
    answer: str


def build_eval_graph(
    llm: Any,
    retriever: Optional[Callable[[str], str]],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Any:
    """
    Build a LangGraph with retrieve -> generate.

    Parameters
    ----------
    llm
        LangChain BaseChatModel (e.g. ChatAnthropic).
    retriever
        If None, retrieval_context stays empty. Otherwise maps question -> context string.
    """
    from langchain_core.messages import HumanMessage
    from langchain_core.messages import SystemMessage
    from langgraph.graph import END
    from langgraph.graph import START
    from langgraph.graph import StateGraph

    def retrieve_node(state: EvalAgentState) -> dict[str, str]:
        q = state["question"]
        if retriever is None:
            return {"retrieval_context": ""}
        ctx = retriever(q)
        return {"retrieval_context": ctx}

    def generate_node(state: EvalAgentState) -> dict[str, str]:
        ctx = (state.get("retrieval_context") or "").strip()
        q = state["question"]
        if ctx:
            user = f"Context:\n{ctx}\n\nQuestion: {q}"
        else:
            user = f"Question: {q}"
        resp = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user),
            ]
        )
        content = getattr(resp, "content", str(resp))
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return {"answer": str(content)}

    graph = StateGraph(EvalAgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def run_eval_graph(graph: Any, question: str) -> EvalAgentState:
    """Invoke the compiled graph and return the final state."""
    return graph.invoke({"question": question})
