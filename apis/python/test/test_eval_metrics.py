import json

import pytest

from tiledb.vector_search.evals.dataset import EvalExample
from tiledb.vector_search.evals.dataset import load_jsonl_dataset
from tiledb.vector_search.evals.metrics import exact_match
from tiledb.vector_search.evals.metrics import normalize_answer
from tiledb.vector_search.evals.metrics import parse_llm_judge_score
from tiledb.vector_search.evals.metrics import token_f1


def test_normalize_answer() -> None:
    assert normalize_answer("  Hello   World  ") == "hello world"


def test_exact_match() -> None:
    assert exact_match("Hello", "hello")
    assert not exact_match("no", "yes")


def test_token_f1() -> None:
    assert token_f1("the cat", "the cat") == 1.0
    assert token_f1("", "") == 1.0
    assert token_f1("a b", "a b c") == pytest.approx(4 / 5)


def test_parse_llm_judge_score() -> None:
    assert parse_llm_judge_score('{"score": 0.75, "rationale": "ok"}') == 0.75
    assert parse_llm_judge_score("no score field in this text") is None
    assert parse_llm_judge_score('text "score": 0.3') == pytest.approx(0.3)


def test_load_jsonl_dataset(tmp_path) -> None:
    p = tmp_path / "d.jsonl"
    p.write_text(
        '{"question":"q1","gold_answer":"a1"}\n\n{"question":"q2","gold_answer":"a2","id":"x"}\n',
        encoding="utf-8",
    )
    rows = load_jsonl_dataset(p)
    assert len(rows) == 2
    assert rows[0] == EvalExample(question="q1", gold_answer="a1")
    assert rows[1].example_id == "x"


def test_eval_example_from_dict_errors() -> None:
    with pytest.raises(KeyError):
        EvalExample.from_dict({"question": "q"})
    with pytest.raises(TypeError):
        EvalExample.from_dict(
            {"question": "q", "gold_answer": "a", "relevant_file_paths": "bad"}
        )


def test_load_jsonl_invalid(tmp_path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_jsonl_dataset(p)
    p2 = tmp_path / "bad2.jsonl"
    p2.write_text(json.dumps(["list"]) + "\n", encoding="utf-8")
    with pytest.raises(TypeError):
        load_jsonl_dataset(p2)
