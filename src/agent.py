"""Minimal tool-use agent: sql_query, read_doc, answer.

Runs a single session's tool loop against whatever `MemoryContext` a memory
strategy handed it, and returns a `SessionRecord` for the caller (later:
`run.py`) to feed back into that strategy's `on_session_end`. The agent
itself is memory-strategy-agnostic — it only ever sees a block of text to
prepend to the task prompt.

Citation trace: the `answer` tool requires the model to name the session it
recalled a fact from (or null). This is what lets a later judge tell "wrong
answer" apart from "wrong answer because it used a stale source" — the
interference-attribution design this whole benchmark hinges on.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from environment.build_env import REFERENCE_DATE
from src.llm import LLMClient, Message, ToolCall, ToolSpec
from src.memory.base import MemoryContext, SessionRecord

MAX_STEPS = 8

SYSTEM_PROMPT_TEMPLATE = """You are a data analyst assistant for Solace Metrics, a B2B SaaS company.

Today's date is {today}. For any question about "current", "right now", or a
trailing lookback window (e.g. "the last 90 days"), compute relative to
{today} — do NOT use SQLite's now() or CURRENT_DATE, they will not match
this database (its data is dated relative to {today}, not the real calendar).

Database schema (SQLite) — use these exact table and column names, do not guess others:
{schema}

Reference documents available via read_doc:
{doc_list}

You have three tools:
- sql_query(query): run a read-only SQL SELECT (or WITH...SELECT) against the company database.
- read_doc(path): read a markdown reference document, e.g. "policies/active_customer_definition.md".
- answer(value, source_session): submit your final answer and end the session. Only call this
  once you actually have the result from a tool call — never call it in the same turn as
  another tool call, and never pass a column alias or placeholder as `value`.

Ground every answer in sql_query or read_doc output — never guess a number.
If you relied on a decision from an EARLIER SESSION in this conversation's
history to answer, cite that session's number in source_session. If you
relied only on this session's own tool calls, pass source_session: null.
Give `value` as a bare number or short exact string (e.g. "yes", "no", a
name) — no units, no explanation, no extra formatting.
"""

TOOL_SPECS = [
    ToolSpec(
        name="sql_query",
        description="Run a read-only SQL SELECT statement against the company database and get the result rows.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "A SELECT (or WITH...SELECT) statement."}},
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="read_doc",
        description="Read a markdown reference document by its path relative to the docs folder.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "e.g. 'policies/active_customer_definition.md'"}
            },
            "required": ["path"],
        },
    ),
    ToolSpec(
        name="answer",
        description="Submit your final answer and end the session.",
        parameters={
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "description": "The final answer, as a bare number or short exact string.",
                },
                "source_session": {
                    "type": ["integer", "null"],
                    "description": (
                        "The session number this answer's key fact was decided or revised in, "
                        "if you relied on prior-session history. Null if you didn't need history."
                    ),
                },
            },
            "required": ["value"],
        },
    ),
]


def message_to_dict(m: Message) -> dict:
    d: dict = {"role": m.role, "content": m.content}
    if m.tool_calls:
        d["tool_calls"] = [{"name": tc.name, "arguments": tc.arguments} for tc in m.tool_calls]
    if m.tool_call_id:
        d["tool_call_id"] = m.tool_call_id
    if m.tool_name:
        d["tool_name"] = m.tool_name
    return d


def _describe_schema(db_path: Path) -> str:
    uri = f"file:{Path(db_path).resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name"
        ).fetchall()
        return "\n\n".join(r[0] for r in rows)
    finally:
        conn.close()


def _list_docs(docs_dir: Path) -> str:
    docs_dir = Path(docs_dir)
    paths = sorted(str(p.relative_to(docs_dir)).replace("\\", "/") for p in docs_dir.rglob("*.md"))
    return "\n".join(f"- {p}" for p in paths)


def run_session(
    task: dict,
    memory_context: MemoryContext,
    llm_client: LLMClient,
    db_path: Path,
    docs_dir: Path,
    max_steps: int = MAX_STEPS,
) -> SessionRecord:
    user_content = task["prompt"]
    if memory_context.text:
        user_content = (
            "Prior session history (for context; may or may not be relevant to this task):\n"
            f"{memory_context.text}\n\n---\n\nCurrent task:\n{task['prompt']}"
        )

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        today=REFERENCE_DATE.isoformat(), schema=_describe_schema(db_path), doc_list=_list_docs(docs_dir)
    )
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_content),
    ]
    transcript: list[dict] = []
    final_answer: str | None = None
    cited_source_session: int | None = None

    for _ in range(max_steps):
        response = llm_client.complete(messages, TOOL_SPECS)
        assistant_msg = Message(role="assistant", content=response.content, tool_calls=response.tool_calls)
        messages.append(assistant_msg)
        transcript.append(message_to_dict(assistant_msg))

        if not response.tool_calls:
            nudge = Message(role="user", content="Please use a tool, and finish by calling answer(...).")
            messages.append(nudge)
            transcript.append(message_to_dict(nudge))
            continue

        # A model calling `answer` alongside another tool in the same turn is
        # guessing at a result it hasn't seen yet (observed in the wild: it
        # passes the SQL column alias as `value` instead of the real number).
        # Run the other tool call(s) for real and make it try again next turn.
        if len(response.tool_calls) > 1 and any(tc.name == "answer" for tc in response.tool_calls):
            for tc in response.tool_calls:
                if tc.name == "answer":
                    continue
                result_text = _execute_tool(tc, db_path, docs_dir)
                tool_msg = Message(role="tool", content=result_text, tool_call_id=tc.id, tool_name=tc.name)
                messages.append(tool_msg)
                transcript.append(message_to_dict(tool_msg))
            reject = Message(
                role="user",
                content=(
                    "Your answer(...) call was ignored because you called it in the same turn as "
                    "another tool, before seeing that tool's result. Review the result above, then "
                    "call answer(...) with the actual value."
                ),
            )
            messages.append(reject)
            transcript.append(message_to_dict(reject))
            continue

        answer_call = next((tc for tc in response.tool_calls if tc.name == "answer"), None)
        if answer_call:
            final_answer = str(answer_call.arguments.get("value"))
            cited_source_session = answer_call.arguments.get("source_session")
            break

        for tc in response.tool_calls:
            result_text = _execute_tool(tc, db_path, docs_dir)
            tool_msg = Message(role="tool", content=result_text, tool_call_id=tc.id, tool_name=tc.name)
            messages.append(tool_msg)
            transcript.append(message_to_dict(tool_msg))

    return SessionRecord(
        session=task["session"],
        task_id=task["task_id"],
        task_prompt=task["prompt"],
        transcript=transcript,
        final_answer=final_answer,
        cited_source_session=cited_source_session,
    )


def _execute_tool(tc: ToolCall, db_path: Path, docs_dir: Path) -> str:
    try:
        if tc.name == "sql_query":
            return _run_sql_query(tc.arguments.get("query", ""), db_path)
        if tc.name == "read_doc":
            return _read_doc(tc.arguments.get("path", ""), docs_dir)
        return f"Error: unknown tool '{tc.name}'."
    except Exception as exc:  # noqa: BLE001 - tool errors are returned to the model, not raised
        return f"Error: {exc}"


_LIVE_CLOCK_RE = re.compile(r"(?i)current_date\b|current_timestamp\b|current_time\b|['\"]now['\"]")


def _run_sql_query(query: str, db_path: Path) -> str:
    stripped = query.strip().rstrip(";").strip()
    if not re.match(r"(?is)^\s*(select|with)\b", stripped):
        return "Error: only SELECT (or WITH...SELECT) statements are allowed."
    if _LIVE_CLOCK_RE.search(stripped):
        return (
            "Error: this query uses SQLite's real-time clock (now/CURRENT_DATE/"
            "CURRENT_TIMESTAMP/CURRENT_TIME), which does not match this database — "
            f"its data is dated relative to the fixed reference date {REFERENCE_DATE.isoformat()}, "
            "not the real calendar. Rewrite the query using literal dates or date() offsets "
            f"from '{REFERENCE_DATE.isoformat()}' instead."
        )
    uri = f"file:{Path(db_path).resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cursor = conn.execute(stripped)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(200)
        if not rows:
            return "(no rows)"
        lines = [", ".join(columns)]
        lines += [", ".join(str(v) for v in row) for row in rows]
        return "\n".join(lines)
    finally:
        conn.close()


def _read_doc(path: str, docs_dir: Path) -> str:
    docs_dir = Path(docs_dir).resolve()
    target = (docs_dir / path).resolve()
    if docs_dir not in target.parents and target != docs_dir:
        return "Error: path escapes the docs directory."
    if not target.is_file():
        available = sorted(str(p.relative_to(docs_dir)).replace("\\", "/") for p in docs_dir.rglob("*.md"))
        return f"Error: no such document '{path}'. Available: {', '.join(available)}"
    return target.read_text(encoding="utf-8")
