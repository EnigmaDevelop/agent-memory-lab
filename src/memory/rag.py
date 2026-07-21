"""Strategy D — BM25 retrieval over chunked session transcripts.

Every past session is chunked (sliding word window) and indexed; each task
retrieves its own top-k chunks. The retrieved chunk ids are kept on
`MemoryContext.retrieved` — this is the piece the citation-trace judge (a
later step) cross-checks against the agent's `answer(source_session=...)`
to tell "the agent used a stale chunk" apart from "the agent was wrong for
an unrelated reason" (the interference-attribution design).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.memory.base import MemoryContext, MemoryStrategy, SessionRecord, render_session

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class Chunk:
    session: int
    chunk_id: str
    text: str


def _chunk_text(text: str, session: int, window_words: int = 60, overlap_words: int = 15) -> list[Chunk]:
    words = text.split()
    if not words:
        return []
    chunks: list[Chunk] = []
    start, idx = 0, 0
    while start < len(words):
        end = min(start + window_words, len(words))
        chunks.append(Chunk(session=session, chunk_id=f"s{session}_c{idx}", text=" ".join(words[start:end])))
        idx += 1
        if end == len(words):
            break
        start = end - overlap_words
    return chunks


class RAGMemory(MemoryStrategy):
    name = "rag"

    def __init__(self, top_k: int = 4):
        self.top_k = top_k
        self.chunks: list[Chunk] = []

    def build_context(self, task: dict) -> MemoryContext:
        if not self.chunks:
            return MemoryContext(text="", included_sessions=[], retrieved=[])

        corpus = [_tokenize(c.text) for c in self.chunks]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(_tokenize(task["prompt"]))

        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)[: self.top_k]
        ranked = [i for i in ranked if scores[i] > 0]  # drop zero-relevance hits rather than pad with noise

        retrieved_chunks = [self.chunks[i] for i in ranked]
        text = "\n\n".join(f"[Session {c.session} | {c.chunk_id}] {c.text}" for c in retrieved_chunks)
        retrieved_meta = [
            {"session": c.session, "chunk_id": c.chunk_id, "score": float(scores[i])}
            for i, c in zip(ranked, retrieved_chunks)
        ]
        included = sorted({c.session for c in retrieved_chunks})
        return MemoryContext(text=text, included_sessions=included, retrieved=retrieved_meta)

    def on_session_end(self, record: SessionRecord) -> None:
        self.chunks.extend(_chunk_text(render_session(record), record.session))
