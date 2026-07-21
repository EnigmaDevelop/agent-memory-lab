"""Strategy A — no memory. The control group.

Every session starts from nothing; `on_session_end` doesn't even bother
keeping the record around.
"""

from __future__ import annotations

from .base import MemoryContext, MemoryStrategy, SessionRecord


class NoMemory(MemoryStrategy):
    name = "none"

    def build_context(self, task: dict) -> MemoryContext:
        return MemoryContext(text="", included_sessions=[])

    def on_session_end(self, record: SessionRecord) -> None:
        pass
