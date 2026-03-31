from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterator
from typing import List
from typing import Optional


@dataclass(frozen=True)
class EvalExample:
    """One row from an eval JSONL file."""

    question: str
    gold_answer: str
    example_id: Optional[str] = None
    relevant_file_paths: Optional[List[str]] = None

    @staticmethod
    def from_dict(row: dict[str, Any]) -> EvalExample:
        if "question" not in row or "gold_answer" not in row:
            raise KeyError("Each JSONL row must include 'question' and 'gold_answer'")
        rel = row.get("relevant_file_paths")
        if rel is not None and not isinstance(rel, list):
            raise TypeError("relevant_file_paths must be a list of strings when present")
        return EvalExample(
            question=str(row["question"]),
            gold_answer=str(row["gold_answer"]),
            example_id=row.get("id"),
            relevant_file_paths=[str(x) for x in rel] if rel else None,
        )


def load_jsonl_dataset(path: Path) -> list[EvalExample]:
    """Load eval examples from a JSONL file (one JSON object per line)."""
    examples: list[EvalExample] = []
    text = path.read_text(encoding="utf-8")
    for line_num, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON on line {line_num} of {path}") from e
        if not isinstance(row, dict):
            raise TypeError(f"Line {line_num} of {path} must be a JSON object")
        examples.append(EvalExample.from_dict(row))
    return examples


def iter_jsonl_dataset(path: Path) -> Iterator[EvalExample]:
    yield from load_jsonl_dataset(path)
