from pathlib import Path

import pytest

from environment.build_env import build as build_env
from tasks.generate import N_SESSIONS, generate


@pytest.fixture(scope="module")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("env")
    build_env(42, d / "company.db", d / "docs", d / "manifest.json")
    return d / "company.db"


def test_same_seed_produces_identical_tasks(db_path: Path):
    tasks_a = generate(0, db_path)
    tasks_b = generate(0, db_path)
    assert tasks_a == tasks_b


def test_different_seed_reorders_but_keeps_same_task_content(db_path: Path):
    """--seed controls session order only; task content (facts, questions,
    answers) must stay identical across seeds — otherwise a difference
    between two seeds could come from content difficulty, not pure order,
    which would defeat the point of the order-shuffle control."""
    tasks_a = generate(0, db_path)
    tasks_b = generate(1, db_path)

    sessions_a = [t["session"] for t in tasks_a]
    sessions_b = [t["session"] for t in tasks_b]
    by_id_a = {t["task_id"]: t["session"] for t in tasks_a}
    by_id_b = {t["task_id"]: t["session"] for t in tasks_b}
    assert sessions_a == sorted(sessions_a) == sessions_b == sorted(sessions_b)
    assert by_id_a != by_id_b, "different seeds should shuffle session order"

    # Same underlying tasks (ids, prompts, answers) regardless of order.
    content_a = {t["task_id"]: (t["prompt"], t["answer"], t.get("stale_answer")) for t in tasks_a}
    content_b = {t["task_id"]: (t["prompt"], t["answer"], t.get("stale_answer")) for t in tasks_b}
    assert content_a == content_b


def test_task_type_counts(db_path: Path):
    tasks = generate(0, db_path)
    counts = {"independent": 0, "dependent": 0, "trap": 0}
    for t in tasks:
        counts[t["task_type"]] += 1
    assert counts == {"independent": 10, "dependent": 12, "trap": 8}
    assert len(tasks) == N_SESSIONS == 30


def test_every_trap_stale_answer_differs_from_correct(db_path: Path):
    tasks = generate(0, db_path)
    trap_tasks = [t for t in tasks if t["task_type"] == "trap"]
    assert len(trap_tasks) == 8
    for t in trap_tasks:
        assert t["answer"] != t["stale_answer"], t["task_id"]


def test_trap_gaps_are_not_trivially_small(db_path: Path):
    """A 0-vs-1 gap is barely distinguishable from noise. Every trap task
    should clear a minimum effect size — this is the automated half of
    verifying that traps are genuinely traps; the other half is manual
    review of tasks.yaml. Task content is fixed regardless of --seed
    (see test_different_seed_reorders_...), so checking seed=0 covers
    every seed.
    """
    tasks = generate(0, db_path)
    for t in tasks:
        if t["task_type"] != "trap":
            continue
        gap = abs(t["answer"] - t["stale_answer"])
        assert gap >= 2, f"{t['task_id']} gap={gap} is too small to be a meaningful trap"


def test_dependent_and_trap_tasks_cite_a_strictly_earlier_session(db_path: Path):
    tasks = generate(0, db_path)
    for t in tasks:
        if t["task_type"] in ("dependent", "trap"):
            assert t["expected_source_session"] < t["session"], t["task_id"]
        if t["task_type"] == "trap":
            assert t["stale_source_session"] < t["expected_source_session"], t["task_id"]


def test_dependent_and_trap_prompts_do_not_leak_the_decided_value(db_path: Path):
    """Dependent/trap prompts must reference a fact by label only — if the
    numeric threshold or category list leaks into the prompt text, the task
    is solvable without memory at all, which defeats the point.
    """
    tasks = generate(0, db_path)
    leak_strings = ["$2000", "$5000", "45 days", "30 days", "billing, bug_report", "bug_report, data_export"]
    for t in tasks:
        if t["task_type"] in ("dependent", "trap"):
            for leak in leak_strings:
                assert leak not in t["prompt"], f"{t['task_id']} leaks decided value: {leak!r}"


def test_independent_tasks_need_no_history(db_path: Path):
    tasks = generate(0, db_path)
    for t in tasks:
        if t["task_type"] == "independent":
            assert "expected_source_session" not in t
            assert "stale_answer" not in t


def test_missing_environment_raises_clear_error(tmp_path: Path):
    with pytest.raises(sqlite3_or_system_exit()):
        generate(0, tmp_path / "does-not-exist.db")


def sqlite3_or_system_exit():
    import sqlite3

    return sqlite3.OperationalError
