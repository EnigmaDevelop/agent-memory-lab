"""Strategy C — rolling LLM summary, regenerated at the end of each session.

Summary quality is a confound separate from the strategy itself (summary
faithfulness to the source transcript) — this module doesn't score its own
faithfulness, it just keeps both halves available for a later check: `self.summaries` (what
the strategy actually uses) and the original `record.transcript` (still on
every `SessionRecord` passed to `on_session_end`) so a later script can diff
summary against source.
"""

from __future__ import annotations

from src.llm import LLMClient, Message
from src.memory.base import MemoryContext, MemoryStrategy, SessionRecord, render_session

SUMMARIZER_INSTRUCTION = (
    "Summarize the following session in 1-3 sentences. Preserve any specific "
    "decisions, definitions, thresholds, or category lists that were "
    "established or revised — a later session may need to recall them "
    "precisely, so do not soften exact numbers or names into vague language."
)


class RollingSummaryMemory(MemoryStrategy):
    name = "summary"

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.summaries: list[tuple[int, str]] = []  # (session, summary_text), chronological

    def build_context(self, task: dict) -> MemoryContext:
        if not self.summaries:
            return MemoryContext(text="", included_sessions=[])
        text = "\n".join(f"[Session {session}] {summary}" for session, summary in self.summaries)
        return MemoryContext(text=text, included_sessions=[s for s, _ in self.summaries])

    def on_session_end(self, record: SessionRecord) -> None:
        prompt = f"{SUMMARIZER_INSTRUCTION}\n\n{render_session(record)}"
        response = self.llm_client.complete([Message(role="user", content=prompt)], tools=[])
        self.summaries.append((record.session, response.content.strip()))
