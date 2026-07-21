from pathlib import Path

import pytest

from environment.build_env import build as build_env
from src.calibrate import make_run_id, run_calibration
from src.llm import LLMResponse, ScriptedLLM, ToolCall


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory):
    d = tmp_path_factory.mktemp("env")
    build_env(42, d / "company.db", d / "docs", d / "manifest.json")
    return {"db_path": d / "company.db", "docs_dir": d / "docs"}


def _answer(value: str) -> LLMResponse:
    return LLMResponse(content="", tool_calls=[ToolCall(id="c1", name="answer", arguments={"value": value, "source_session": None})])


def test_run_calibration_scores_against_real_sql_answers(env, monkeypatch):
    """Every calibration task's answer is SQL-computed at generation time;
    a ScriptedLLM that always answers '999999' should score 0 correct, and
    swapping in a monkeypatched client isn't needed to prove the scoring
    plumbing works end-to-end (agent -> judge -> rows)."""
    import src.calibrate as calibrate_module

    monkeypatch.setattr(calibrate_module, "build_llm_client", lambda provider, model: ScriptedLLM([_answer("999999")] * 5))
    rows, transcripts = run_calibration("ollama", "fake-model", n=5, seed=0, db_path=env["db_path"], docs_dir=env["docs_dir"], verbose=False)
    assert len(rows) == 5
    assert len(transcripts) == 5
    assert all(r["correct"] is False for r in rows)  # 999999 is not a real answer for any of these


def test_make_run_id_is_filesystem_safe():
    run_id = make_run_id("ollama", "mistral:latest", 0, 24)
    assert ":" not in run_id
    assert run_id == "calibration_ollama_mistral-latest_seed0_n24"
