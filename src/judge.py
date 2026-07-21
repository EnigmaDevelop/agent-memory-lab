"""Deterministic answer checking + false-memory-rate classification.

No LLM-as-judge anywhere — a deliberate scoring choice this lab defends in
the article. Every task has an exact machine-checkable answer; this module
just normalizes the agent's raw string and compares.

For trap tasks specifically, `classify` implements the citation-trace
attribution this benchmark commits to: a wrong answer is only labeled
"interference_confirmed" when the agent's answer behaviorally matches the
stale (pre-revision) value *and* it cited the stale source session. If the
behavioral match is there but the citation isn't (a real, measured gap for
weak local models — see Step 3's smoke test), it's "interference_suspected"
instead of being silently counted either way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.memory.base import SessionRecord

_NUMERIC_STRIP_RE = re.compile(r"[^0-9.\-]")


def normalize_numeric(raw: str) -> float | None:
    if raw is None:
        return None
    cleaned = _NUMERIC_STRIP_RE.sub("", raw.strip())
    if cleaned in ("", "-", "."):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def normalize_string(raw: str) -> str:
    return raw.strip().strip("'\"").strip().lower()


def _values_match(task: dict, agent_answer: str | None, expected) -> bool:
    if agent_answer is None:
        return False
    if task["answer_type"] == "int":
        parsed = normalize_numeric(agent_answer)
        if parsed is None:
            return False
        tolerance = task.get("tolerance", 0)
        return abs(parsed - float(expected)) <= tolerance
    return normalize_string(agent_answer) == normalize_string(str(expected))


def is_correct(task: dict, agent_answer: str | None) -> bool:
    return _values_match(task, agent_answer, task["answer"])


@dataclass
class JudgeResult:
    correct: bool
    category: str  # "correct" | "wrong" | "interference_confirmed" | "interference_suspected" | "wrong_other"
    final_answer: str | None
    expected: object
    cited_source_session: int | None


def judge_session(task: dict, record: SessionRecord) -> JudgeResult:
    correct = is_correct(task, record.final_answer)

    if task["task_type"] != "trap":
        category = "correct" if correct else "wrong"
        return JudgeResult(correct, category, record.final_answer, task["answer"], record.cited_source_session)

    if correct:
        return JudgeResult(True, "correct", record.final_answer, task["answer"], record.cited_source_session)

    stale_match = _values_match(task, record.final_answer, task["stale_answer"])
    if not stale_match:
        category = "wrong_other"
    elif record.cited_source_session == task["stale_source_session"]:
        category = "interference_confirmed"
    else:
        category = "interference_suspected"

    return JudgeResult(False, category, record.final_answer, task["answer"], record.cited_source_session)
