from langchain_core.language_models.fake_chat_models import FakeListChatModel

from tiledb.vector_search.evals.agent_graph import build_eval_graph
from tiledb.vector_search.evals.agent_graph import run_eval_graph


def test_build_eval_graph_no_retriever() -> None:
    llm = FakeListChatModel(responses=["answer without context"])
    graph = build_eval_graph(llm, retriever=None)
    out = run_eval_graph(graph, "What is 2+2?")
    assert out.get("retrieval_context") == ""
    assert out.get("answer") == "answer without context"


def test_build_eval_graph_with_retriever() -> None:
    llm = FakeListChatModel(responses=["answer with context"])

    def retriever(q: str) -> str:
        assert "climate" in q.lower()
        return "ctx: polar bears"

    graph = build_eval_graph(llm, retriever)
    out = run_eval_graph(graph, "Tell me about climate")
    assert "polar" in out.get("retrieval_context", "")
    assert out.get("answer") == "answer with context"
