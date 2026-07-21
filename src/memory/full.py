"""Strategy B — full history in context, oldest dropped first when it
doesn't fit.

`build_context` reports whether truncation happened (`MemoryContext.truncated`)
so run-time analysis can stratify accuracy by "did this session actually see
its dependency, or was it truncated away" — a truncated dependent-task miss
is a context-budget failure, not evidence of interference, so truncated and
untruncated context conditions must be told apart before scoring.
"""

from __future__ import annotations

from src.memory.base import MemoryContext, MemoryStrategy, SessionRecord, estimate_tokens, render_session


class FullHistoryMemory(MemoryStrategy):
    name = "full"

    def __init__(self, token_budget: int = 6000):
        self.token_budget = token_budget
        self.history: list[SessionRecord] = []

    def build_context(self, task: dict) -> MemoryContext:
        if not self.history:
            return MemoryContext(text="", included_sessions=[])

        kept: list[SessionRecord] = []
        used = 0
        for record in reversed(self.history):  # newest first
            rendered = render_session(record)
            cost = estimate_tokens(rendered)
            if used + cost > self.token_budget and kept:
                break
            kept.append(record)
            used += cost
        kept.reverse()  # back to chronological order

        truncated = len(kept) < len(self.history)
        text = "\n\n".join(render_session(r) for r in kept)
        return MemoryContext(text=text, included_sessions=[r.session for r in kept], truncated=truncated)

    def on_session_end(self, record: SessionRecord) -> None:
        self.history.append(record)
