from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any
from typing import Optional


def normalize_answer(text: str) -> str:
    """Lowercase and collapse whitespace for robust string comparison."""
    return " ".join(text.lower().split())


def exact_match(prediction: str, gold: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(gold)


def token_f1(prediction: str, gold: str) -> float:
    """Token-level F1 between prediction and gold (whitespace tokenization)."""
    pred_toks = normalize_answer(prediction).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    pred_set = pred_toks
    gold_set = gold_toks
    pc = Counter(pred_set)
    gc = Counter(gold_set)
    overlap = sum((pc & gc).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_set)
    recall = overlap / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


_SCORE_RE = re.compile(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', re.IGNORECASE)


def parse_llm_judge_score(raw: str) -> Optional[float]:
    """Extract a numeric score from judge output (JSON preferred)."""
    raw = raw.strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "score" in data:
            s = float(data["score"])
            return max(0.0, min(1.0, s))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    m = _SCORE_RE.search(raw)
    if m:
        try:
            s = float(m.group(1))
            return max(0.0, min(1.0, s))
        except ValueError:
            pass
    return None


def llm_judge_answer(
    llm: Any,
    question: str,
    prediction: str,
    gold: str,
) -> tuple[Optional[float], str]:
    """
    Ask an LLM to score whether the prediction matches the gold answer (0.0–1.0).

    Parameters
    ----------
    llm
        A LangChain chat model with .invoke(messages).
    """
    from langchain_core.messages import HumanMessage
    from langchain_core.messages import SystemMessage

    sys = (
        "You compare a model answer to a reference answer for the same question. "
        "Reply with a single JSON object only, no markdown: "
        '{"score": <number from 0.0 to 1.0>, "rationale": "<short>"}. '
        "1.0 means the prediction is fully correct or equivalent; 0.0 means wrong or unrelated."
    )
    user = (
        f"Question: {question}\n\n"
        f"Reference answer: {gold}\n\n"
        f"Model answer: {prediction}\n"
    )
    msg = llm.invoke([SystemMessage(content=sys), HumanMessage(content=user)])
    content = getattr(msg, "content", str(msg))
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    score = parse_llm_judge_score(str(content))
    return score, str(content)
