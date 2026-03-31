from __future__ import annotations

import json
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional

import tiledb

from tiledb.vector_search.cli.progress import progress_bar
from tiledb.vector_search.cli.progress import progress_done
from tiledb.vector_search.evals.agent_graph import build_eval_graph
from tiledb.vector_search.evals.agent_graph import run_eval_graph
from tiledb.vector_search.evals.dataset import EvalExample
from tiledb.vector_search.evals.dataset import load_jsonl_dataset
from tiledb.vector_search.evals.metrics import exact_match
from tiledb.vector_search.evals.metrics import llm_judge_answer
from tiledb.vector_search.evals.metrics import token_f1
from tiledb.vector_search.evals.retrieval import make_index_retriever
from tiledb.vector_search.evals.retrieval import query_topk_file_paths
from tiledb.vector_search.evals.retrieval import recall_at_k


def _meta_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def read_index_embedding_model_name(
    index_uri: str,
    config: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Resolve embedding model id/path from vector index group metadata.

    Uses ``sentence_transformer_model`` when present, otherwise
    ``model_name_or_path`` inside ``embedding_kwargs`` JSON.
    """
    ctx = (
        tiledb.Ctx(tiledb.Config(config)) if config is not None else None
    )
    with tiledb.Group(index_uri, "r", ctx=ctx) as group:
        st = group.meta.get("sentence_transformer_model")
        if st is not None:
            s = _meta_str(st).strip()
            if s:
                return s
        raw_kw = group.meta.get("embedding_kwargs")
        if raw_kw is None:
            return None
        try:
            kw = json.loads(_meta_str(raw_kw))
        except (json.JSONDecodeError, TypeError):
            return None
        path = kw.get("model_name_or_path")
        if path:
            return str(path).strip() or None
    return None


@dataclass
class ExampleResult:
    example_id: Optional[str]
    question: str
    prediction: str
    gold_answer: str
    exact: bool
    f1: float
    llm_judge_score: Optional[float] = None
    llm_judge_raw: Optional[str] = None
    recall_at_k: Optional[float] = None


@dataclass
class AggregateStats:
    n: int = 0
    exact_rate: float = 0.0
    mean_f1: float = 0.0
    mean_llm_judge: Optional[float] = None
    mean_recall_at_k: Optional[float] = None


def _aggregate(results: List[ExampleResult]) -> AggregateStats:
    if not results:
        return AggregateStats()
    n = len(results)
    exact_rate = sum(1 for r in results if r.exact) / n
    mean_f1 = sum(r.f1 for r in results) / n
    judge_vals = [r.llm_judge_score for r in results if r.llm_judge_score is not None]
    mean_judge = sum(judge_vals) / len(judge_vals) if judge_vals else None
    recall_vals = [r.recall_at_k for r in results if r.recall_at_k is not None]
    mean_recall = (
        sum(recall_vals) / len(recall_vals)
        if recall_vals
        else None
    )
    return AggregateStats(
        n=n,
        exact_rate=exact_rate,
        mean_f1=mean_f1,
        mean_llm_judge=mean_judge,
        mean_recall_at_k=mean_recall,
    )


def _score_example(
    ex: EvalExample,
    prediction: str,
    judge_llm: Optional[Any],
) -> ExampleResult:
    ex_id = ex.example_id
    em = exact_match(prediction, ex.gold_answer)
    f1 = token_f1(prediction, ex.gold_answer)
    j_score: Optional[float] = None
    j_raw: Optional[str] = None
    if judge_llm is not None:
        j_score, j_raw = llm_judge_answer(
            judge_llm, ex.question, prediction, ex.gold_answer
        )
    return ExampleResult(
        example_id=ex_id,
        question=ex.question,
        prediction=prediction,
        gold_answer=ex.gold_answer,
        exact=em,
        f1=f1,
        llm_judge_score=j_score,
        llm_judge_raw=j_raw,
    )


def run_agent_on_dataset(
    examples: List[EvalExample],
    llm: Any,
    index_uri: Optional[str],
    top_k: int,
    use_retrieval: bool,
    llm_judge: bool,
    judge_llm: Optional[Any] = None,
    progress_label: Optional[str] = None,
) -> List[ExampleResult]:
    from tiledb.vector_search.object_api.object_index import ObjectIndex

    retriever = None
    obj_index = None
    if use_retrieval and index_uri:
        obj_index = ObjectIndex(uri=index_uri, load_metadata_in_memory=True)
        retriever = make_index_retriever(obj_index, top_k)
    graph = build_eval_graph(llm, retriever)
    judge = judge_llm if llm_judge else None
    out: List[ExampleResult] = []
    n = len(examples)
    t0 = time.monotonic()
    for i, ex in enumerate(examples):
        if progress_label and n:
            progress_bar(i, n, progress_label, t0)
        state = run_eval_graph(graph, ex.question)
        pred = state.get("answer", "")
        r = _score_example(ex, pred, judge)
        if ex.relevant_file_paths and obj_index is not None:
            paths = query_topk_file_paths(obj_index, ex.question, top_k)
            r.recall_at_k = recall_at_k(ex.relevant_file_paths, paths)
        out.append(r)
    if progress_label and n:
        progress_done(n, n, progress_label, t0)
    return out


def run_agent_eval_report(
    dataset_path: Path,
    index_uri: Optional[str],
    top_k: int,
    anthropic_model: str,
    llm_judge: bool,
    run_without_index: bool,
    run_with_index: bool,
    index_config: Optional[Mapping[str, Any]] = None,
    show_progress: bool = False,
) -> Dict[str, Any]:
    from langchain_anthropic import ChatAnthropic

    examples = load_jsonl_dataset(dataset_path)
    llm = ChatAnthropic(model=anthropic_model)
    judge_llm = ChatAnthropic(model=anthropic_model) if llm_judge else None

    report: Dict[str, Any] = {
        "dataset": dataset_path.resolve().name,
        "top_k": top_k,
        "model": anthropic_model,
        "llm_judge": llm_judge,
    }

    if run_with_index:
        if not index_uri:
            raise ValueError("index_uri required when run_with_index is True")
        with_results = run_agent_on_dataset(
            examples,
            llm,
            index_uri,
            top_k,
            use_retrieval=True,
            llm_judge=llm_judge,
            judge_llm=judge_llm,
            progress_label=(
                "Eval examples (with index)" if show_progress else None
            ),
        )
        emb_model = read_index_embedding_model_name(
            index_uri, config=index_config
        )
        report["with_index"] = {
            "index_uri": index_uri,
            "embedding_model": emb_model,
            "per_example": [asdict(r) for r in with_results],
            "aggregate": asdict(_aggregate(with_results)),
        }

    if run_without_index:
        no_results = run_agent_on_dataset(
            examples,
            llm,
            None,
            top_k,
            use_retrieval=False,
            llm_judge=llm_judge,
            judge_llm=judge_llm,
            progress_label=(
                "Eval examples (without index)" if show_progress else None
            ),
        )
        report["without_index"] = {
            "per_example": [asdict(r) for r in no_results],
            "aggregate": asdict(_aggregate(no_results)),
        }

    return report


@dataclass
class IndexCompareRetrievalResult:
    example_id: Optional[str]
    question: str
    recall_index_a: float
    recall_index_b: float


def compare_indexes_retrieval(
    examples: List[EvalExample],
    index_uri_a: str,
    index_uri_b: str,
    top_k: int,
    progress_label: Optional[str] = None,
) -> List[IndexCompareRetrievalResult]:
    from tiledb.vector_search.object_api.object_index import ObjectIndex

    labeled = [e for e in examples if e.relevant_file_paths]
    if not labeled:
        return []
    idx_a = ObjectIndex(uri=index_uri_a, load_metadata_in_memory=True)
    idx_b = ObjectIndex(uri=index_uri_b, load_metadata_in_memory=True)
    out: List[IndexCompareRetrievalResult] = []
    n = len(labeled)
    t0 = time.monotonic()
    for i, ex in enumerate(labeled):
        if progress_label and n:
            progress_bar(i, n, progress_label, t0)
        pa = query_topk_file_paths(idx_a, ex.question, top_k)
        pb = query_topk_file_paths(idx_b, ex.question, top_k)
        ra = recall_at_k(ex.relevant_file_paths, pa)
        rb = recall_at_k(ex.relevant_file_paths, pb)
        out.append(
            IndexCompareRetrievalResult(
                example_id=ex.example_id,
                question=ex.question,
                recall_index_a=ra,
                recall_index_b=rb,
            )
        )
    if progress_label and n:
        progress_done(n, n, progress_label, t0)
    return out


def run_compare_indexes_report(
    dataset_path: Path,
    index_uri_a: str,
    index_uri_b: str,
    top_k: int,
    anthropic_model: str,
    answer_metrics: bool,
    llm_judge: bool,
    index_config: Optional[Mapping[str, Any]] = None,
    show_progress: bool = False,
) -> Dict[str, Any]:
    examples = load_jsonl_dataset(dataset_path)
    retrieval_rows = compare_indexes_retrieval(
        examples,
        index_uri_a,
        index_uri_b,
        top_k,
        progress_label="Retrieval compare" if show_progress else None,
    )
    emb_a = read_index_embedding_model_name(index_uri_a, config=index_config)
    emb_b = read_index_embedding_model_name(index_uri_b, config=index_config)
    report: Dict[str, Any] = {
        "dataset": dataset_path.resolve().name,
        "index_a": index_uri_a,
        "index_b": index_uri_b,
        "index_a_embedding_model": emb_a,
        "index_b_embedding_model": emb_b,
        "top_k": top_k,
        "retrieval": {
            "per_example": [asdict(r) for r in retrieval_rows],
            "mean_recall_a": (
                sum(r.recall_index_a for r in retrieval_rows) / len(retrieval_rows)
                if retrieval_rows
                else None
            ),
            "mean_recall_b": (
                sum(r.recall_index_b for r in retrieval_rows) / len(retrieval_rows)
                if retrieval_rows
                else None
            ),
        },
    }

    if answer_metrics:
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model=anthropic_model)
        judge_llm = ChatAnthropic(model=anthropic_model) if llm_judge else None
        a_results = run_agent_on_dataset(
            examples,
            llm,
            index_uri_a,
            top_k,
            use_retrieval=True,
            llm_judge=llm_judge,
            judge_llm=judge_llm,
            progress_label=(
                "Answer metrics (index A)" if show_progress else None
            ),
        )
        b_results = run_agent_on_dataset(
            examples,
            llm,
            index_uri_b,
            top_k,
            use_retrieval=True,
            llm_judge=llm_judge,
            judge_llm=judge_llm,
            progress_label=(
                "Answer metrics (index B)" if show_progress else None
            ),
        )
        report["answers_index_a"] = {
            "index_uri": index_uri_a,
            "embedding_model": emb_a,
            "per_example": [asdict(r) for r in a_results],
            "aggregate": asdict(_aggregate(a_results)),
        }
        report["answers_index_b"] = {
            "index_uri": index_uri_b,
            "embedding_model": emb_b,
            "per_example": [asdict(r) for r in b_results],
            "aggregate": asdict(_aggregate(b_results)),
        }

    return report
