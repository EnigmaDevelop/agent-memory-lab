from pathlib import Path

import pytest

from environment.build_env import build as build_env
from src.agent import MAX_STEPS, run_session
from src.llm import LLMResponse, ScriptedLLM, ToolCall
from src.memory.base import MemoryContext

TASK = {"session": 5, "task_id": "t_test", "prompt": "How many employees are there?"}


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory):
    d = tmp_path_factory.mktemp("env")
    build_env(42, d / "company.db", d / "docs", d / "manifest.json")
    return {"db_path": d / "company.db", "docs_dir": d / "docs"}


def _answer_response(value: str, source_session=None) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id="c1", name="answer", arguments={"value": value, "source_session": source_session})],
    )


def test_answers_directly(env):
    llm = ScriptedLLM([_answer_response("12", None)])
    record = run_session(TASK, MemoryContext(text=""), llm, env["db_path"], env["docs_dir"])
    assert record.final_answer == "12"
    assert record.cited_source_session is None
    assert record.session == 5
    assert record.task_id == "t_test"
    assert len(record.transcript) == 1


def test_cites_a_source_session(env):
    llm = ScriptedLLM([_answer_response("38", source_session=10)])
    record = run_session(TASK, MemoryContext(text="some history"), llm, env["db_path"], env["docs_dir"])
    assert record.cited_source_session == 10


def test_executes_sql_tool_before_answering(env):
    sql_response = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="c1", name="sql_query", arguments={"query": "SELECT COUNT(*) FROM employees"})],
    )
    llm = ScriptedLLM([sql_response, _answer_response("12")])
    record = run_session(TASK, MemoryContext(text=""), llm, env["db_path"], env["docs_dir"])
    assert record.final_answer == "12"
    tool_results = [m for m in record.transcript if m["role"] == "tool"]
    assert len(tool_results) == 1
    assert "12" in tool_results[0]["content"]  # 12 employees in the generated environment


def test_sql_query_rejects_non_select(env):
    destructive = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="c1", name="sql_query", arguments={"query": "DELETE FROM employees"})],
    )
    llm = ScriptedLLM([destructive, _answer_response("done")])
    record = run_session(TASK, MemoryContext(text=""), llm, env["db_path"], env["docs_dir"])
    tool_result = next(m for m in record.transcript if m["role"] == "tool")
    assert "Error" in tool_result["content"]

    import sqlite3

    conn = sqlite3.connect(env["db_path"])
    count = conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0]
    conn.close()
    assert count == 12  # untouched


def test_sql_query_blocks_sqlite_live_clock_functions(env):
    """The system prompt already warns against SQLite's now()/CURRENT_DATE,
    but a weak local model sometimes ignores that instruction anyway
    (observed live with qwen2.5:3b even after the prompt fix — see
    project memory). The tool layer must enforce this deterministically
    instead of relying solely on instruction-following.
    """
    live_clock_queries = [
        "SELECT COUNT(*) FROM events WHERE event_date >= DATE('now', '-45 days')",
        "SELECT COUNT(*) FROM events WHERE event_date >= CURRENT_DATE",
        'SELECT COUNT(*) FROM events WHERE event_date >= DATE("now", "-30 days")',
        "SELECT datetime('now')",
        "SELECT CURRENT_TIMESTAMP",
    ]
    for query in live_clock_queries:
        response = LLMResponse(content="", tool_calls=[ToolCall(id="c1", name="sql_query", arguments={"query": query})])
        llm = ScriptedLLM([response, _answer_response("0")])
        record = run_session(TASK, MemoryContext(text=""), llm, env["db_path"], env["docs_dir"])
        tool_result = next(m for m in record.transcript if m["role"] == "tool")
        assert "Error" in tool_result["content"], f"expected block for: {query}"
        assert "2026-01-01" in tool_result["content"]


def test_sql_query_allows_literal_dates_including_names_containing_now(env):
    """Regression guard: the live-clock block must not false-positive on
    ordinary literal dates or on substrings like the surname 'Nowak' that
    merely contain the letters 'now'."""
    response = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="c1", name="sql_query", arguments={
            "query": "SELECT COUNT(*) FROM events WHERE event_date >= date('2026-01-01', '-45 day') AND event_date < '2026-01-01'"
        })],
    )
    llm = ScriptedLLM([response, _answer_response("0")])
    record = run_session(TASK, MemoryContext(text=""), llm, env["db_path"], env["docs_dir"])
    tool_result = next(m for m in record.transcript if m["role"] == "tool")
    assert "Error" not in tool_result["content"]


def test_read_doc_returns_real_content(env):
    read_call = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="c1", name="read_doc", arguments={"path": "policies/active_customer_definition.md"})],
    )
    llm = ScriptedLLM([read_call, _answer_response("90")])
    record = run_session(TASK, MemoryContext(text=""), llm, env["db_path"], env["docs_dir"])
    tool_result = next(m for m in record.transcript if m["role"] == "tool")
    assert "90 days" in tool_result["content"]


def test_read_doc_blocks_path_traversal(env):
    read_call = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="c1", name="read_doc", arguments={"path": "../../../../windows/win.ini"})],
    )
    llm = ScriptedLLM([read_call, _answer_response("n/a")])
    record = run_session(TASK, MemoryContext(text=""), llm, env["db_path"], env["docs_dir"])
    tool_result = next(m for m in record.transcript if m["role"] == "tool")
    assert "Error" in tool_result["content"]


def test_no_tool_call_gets_nudged_then_recovers(env):
    empty_response = LLMResponse(content="Let me think about this.", tool_calls=[])
    llm = ScriptedLLM([empty_response, _answer_response("42")])
    record = run_session(TASK, MemoryContext(text=""), llm, env["db_path"], env["docs_dir"])
    assert record.final_answer == "42"
    nudges = [m for m in record.transcript if m["role"] == "user"]
    assert len(nudges) == 1


def test_gives_up_after_max_steps_without_answer(env):
    stalling = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="c1", name="sql_query", arguments={"query": "SELECT 1"})],
    )
    llm = ScriptedLLM([stalling] * MAX_STEPS)
    record = run_session(TASK, MemoryContext(text=""), llm, env["db_path"], env["docs_dir"])
    assert record.final_answer is None
    assert len(llm.calls) == MAX_STEPS


def test_system_prompt_contains_real_schema_and_docs(env):
    llm = ScriptedLLM([_answer_response("12")])
    run_session(TASK, MemoryContext(text=""), llm, env["db_path"], env["docs_dir"])
    system_content = llm.calls[0][0][0].content
    assert system_content.startswith("You are a data analyst")
    # Real table/column names, not left for the model to guess (this was a
    # real failure mode with qwen2.5:3b: it invented tables like "support"
    # and "problems" instead of the real support_tickets table).
    assert "CREATE TABLE customers" in system_content
    assert "CREATE TABLE support_tickets" in system_content
    assert "policies/active_customer_definition.md" in system_content


def test_system_prompt_pins_todays_date_and_warns_off_sqlite_now(env):
    """The environment's data is dated relative to a fixed synthetic
    REFERENCE_DATE, not the real calendar. A model that uses SQLite's
    now()/CURRENT_DATE for a "trailing N days" question silently gets zero
    rows back — observed for real with qwen2.5:3b (it wrote
    `event_date >= DATE('now','-45 days')` and got 0 instead of the right
    answer). The prompt must pin an explicit date and warn against now().
    """
    llm = ScriptedLLM([_answer_response("12")])
    run_session(TASK, MemoryContext(text=""), llm, env["db_path"], env["docs_dir"])
    system_content = llm.calls[0][0][0].content
    assert "2026-01-01" in system_content
    assert "now()" in system_content


def test_answer_alongside_another_tool_call_is_rejected_and_tool_still_runs(env):
    """A model calling answer() in the same turn as sql_query() is guessing
    at a result it hasn't seen yet. The harness must run the sql_query for
    real and refuse the premature answer, rather than accepting whatever
    placeholder value the model guessed (observed in the wild: it passed the
    SQL column alias itself, e.g. "total_invoice", as the answer value).
    """
    premature = LLMResponse(
        content="",
        tool_calls=[
            ToolCall(id="c1", name="sql_query", arguments={"query": "SELECT COUNT(*) FROM employees"}),
            ToolCall(id="c2", name="answer", arguments={"value": "employee_count", "source_session": None}),
        ],
    )
    llm = ScriptedLLM([premature, _answer_response("12")])
    record = run_session(TASK, MemoryContext(text=""), llm, env["db_path"], env["docs_dir"])

    assert record.final_answer == "12"  # the second, real answer -- not "employee_count"
    tool_results = [m for m in record.transcript if m["role"] == "tool"]
    assert len(tool_results) == 1
    assert "12" in tool_results[0]["content"]  # the sql_query call was actually executed
    rejections = [m for m in record.transcript if m["role"] == "user" and "ignored" in m["content"]]
    assert len(rejections) == 1
