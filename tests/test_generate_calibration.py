from pathlib import Path

import pytest

from environment.build_env import build as build_env
from tasks.generate_calibration import generate


@pytest.fixture(scope="module")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("env")
    build_env(42, d / "company.db", d / "docs", d / "manifest.json")
    return d / "company.db"


def test_same_seed_produces_identical_tasks(db_path: Path):
    tasks_a = generate(24, 0, db_path)
    tasks_b = generate(24, 0, db_path)
    assert tasks_a == tasks_b


def test_n_controls_count(db_path: Path):
    assert len(generate(10, 0, db_path)) == 10
    assert len(generate(24, 0, db_path)) == 24


def test_all_tasks_are_independent_with_no_source_session(db_path: Path):
    tasks = generate(24, 0, db_path)
    for t in tasks:
        assert t["task_type"] == "independent"
        assert "expected_source_session" not in t
        assert "stale_answer" not in t


def test_task_ids_are_unique(db_path: Path):
    tasks = generate(24, 0, db_path)
    assert len(tasks) == len({t["task_id"] for t in tasks})


def test_sessions_are_1_to_n(db_path: Path):
    tasks = generate(24, 0, db_path)
    assert sorted(t["session"] for t in tasks) == list(range(1, 25))


def test_small_n_still_samples_every_sql_shape(db_path: Path):
    """Round-robin sampling (not exhausting one family before the next)
    must give even a small calibration set at least one join-based task
    (e.g. campaign attribution) and one date-range task — those shapes map
    to the actual failure modes observed live (bad joins, date handling),
    so a family-sequential draw that only ever samples the first few
    template families would quietly make the set easier than intended."""
    tasks = generate(10, 0, db_path)
    prompts = " ".join(t["prompt"] for t in tasks)
    assert "attributed to" in prompts  # join-based family
    assert "signed up before" in prompts or "signed up before" in prompts  # date-range family


def test_different_seed_reorders_but_keeps_same_content(db_path: Path):
    tasks_a = generate(24, 0, db_path)
    tasks_b = generate(24, 1, db_path)
    content_a = {t["task_id"]: t["prompt"] for t in tasks_a}
    content_b = {t["task_id"]: t["prompt"] for t in tasks_b}
    assert content_a == content_b
    sessions_a = {t["task_id"]: t["session"] for t in tasks_a}
    sessions_b = {t["task_id"]: t["session"] for t in tasks_b}
    assert sessions_a != sessions_b


def test_requesting_more_than_available_raises(db_path: Path):
    with pytest.raises(RuntimeError):
        generate(10_000, 0, db_path)
