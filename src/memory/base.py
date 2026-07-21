"""Shared protocol for the three memory strategies (full/summary/rag) and
the no-memory control (none) — all four run through the same interface so
the harness never has to special-case the control.

A strategy only ever does two things:

- `build_context(task)`: called before a session starts, returns whatever
  the strategy wants injected into the agent's prompt for that task.
- `on_session_end(record)`: called after a session finishes, lets the
  strategy absorb that session into whatever state it keeps.

`run.py` (a later step) owns the loop that calls these in order for each of
the 30 sessions; a strategy never reaches across sessions on its own.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SessionRecord:
    session: int
    task_id: str
    task_prompt: str
    transcript: list[dict]  # serialized Message log, see agent.message_to_dict
    final_answer: str | None
    cited_source_session: int | None


@dataclass
class MemoryContext:
    text: str
    included_sessions: list[int] = field(default_factory=list)
    truncated: bool = False  # full.py only: oldest sessions dropped to fit budget
    retrieved: list[dict] | None = None  # rag.py only: [{"session", "chunk_id", "score"}, ...]


class MemoryStrategy(ABC):
    name: str

    @abstractmethod
    def build_context(self, task: dict) -> MemoryContext: ...

    @abstractmethod
    def on_session_end(self, record: SessionRecord) -> None: ...


def estimate_tokens(text: str) -> int:
    """Rough token-count approximation (~4 chars/token in English prose).

    Good enough for a context-budget heuristic; not meant to match any
    specific tokenizer exactly.
    """
    return max(1, len(text) // 4)


def render_session(record: SessionRecord) -> str:
    """Render a session's transcript as plain text — shared by full.py
    (verbatim context) and rag.py (source text to chunk/index)."""
    lines = [f"[Session {record.session}] Task: {record.task_prompt}"]
    for msg in record.transcript:
        if msg["role"] == "assistant":
            for tc in msg.get("tool_calls") or []:
                lines.append(f"  -> called {tc['name']}({tc['arguments']})")
            if msg.get("content"):
                lines.append(f"  assistant: {msg['content']}")
        elif msg["role"] == "tool":
            lines.append(f"  <- {msg.get('tool_name', 'tool')} result: {msg['content']}")
    cite = f" (cited session {record.cited_source_session})" if record.cited_source_session else ""
    lines.append(f"[Session {record.session}] Final answer: {record.final_answer}{cite}")
    return "\n".join(lines)
