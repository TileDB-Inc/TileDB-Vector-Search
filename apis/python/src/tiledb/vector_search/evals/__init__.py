"""Evaluation helpers for RAG agents and vector-index comparisons."""

from tiledb.vector_search.evals.dataset import EvalExample
from tiledb.vector_search.evals.dataset import load_jsonl_dataset

__all__ = [
    "EvalExample",
    "load_jsonl_dataset",
]
