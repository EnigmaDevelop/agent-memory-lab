<p align="center"><strong>agent-memory-lab</strong></p>

<p align="center">
  Does giving an LLM agent memory of its own past sessions help, hurt, or both?<br>
  I measured it — 360 sessions, 3 memory strategies + a no-memory control, real billing data.
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <a href="https://github.com/sponsors/EnigmaDevelop"><img alt="Sponsor" src="https://img.shields.io/badge/%E2%9D%A4-Sponsor-ea4aaa.svg"></a>
</p>

Full writeup: coming soon on Level Up Coding — this README will be updated with the link once it's live.

## The finding

Four conditions, the same 30-task sequence, 3 reshuffled orderings, a citation-trace judge that
distinguishes *any* wrong answer on a trap task from a *provable* citation of a specific stale
session:

| Condition | Accuracy (90 attempts) | Avg. cost / 30-session run* |
|---|---|---|
| No memory (control) | 32.2% | **$1.15** |
| Full history replay | **100.0%** | $0.45 |
| Rolling summary | 95.6% | $0.32 |
| RAG (BM25 retrieval) | 97.8% | $0.22 |

\* real per-minute Anthropic billing data, averaged over 2 of the 3 seeds.

Two things that weren't the expected result going in:

1. **The no-memory control was the most expensive condition to run**, not the cheapest — with
   nothing to reference, the agent burns retries trying to guess answers it has no way to reach.
2. **Exactly one of 270 memory-backed attempts produced a confirmed hallucination** (citing a
   session that was later superseded) — and it was RAG, the strategy usually assumed to be the
   safe, grounded one, not full-history replay. The transcript shows exactly why: BM25 retrieval
   never surfaced the one session that mattered, because a correction doesn't share vocabulary
   with the questions asked about it afterward.

## Reproduce it

```bash
uv sync --extra frontier          # installs the anthropic SDK too
uv run environment/build_env.py --seed 42        # deterministic fictional company DB + docs
uv run python -m tasks.generate --seed 0         # 30-task sequence for this seed

export ANTHROPIC_API_KEY=sk-ant-...
uv run python -m src.run --strategy rag --provider anthropic --model claude-sonnet-5 --seed 0

uv run pytest                                     # 72 unit tests, fully deterministic, no network
uv run python -m scripts.analyze                  # bootstrap CI + permutation tests over all seeds
```

`--strategy` is one of `none` / `full` / `summary` / `rag`. A local, key-free path also exists via
Ollama (`--provider ollama --model <name>`) — two local models were tried and retired for the
headline numbers above (a 3B-parameter SQL ceiling, then a coherence-collapse hallucination bug);
see "What's not in this article" below.

## Repo structure

```
environment/     deterministic fictional B2B SaaS company (SQLite + docs), seed-reproducible
tasks/           30-task sequence generator — independent / dependent / trap task types
src/
  agent.py       the tool-use loop (sql_query / read_doc / answer)
  memory/        three memory strategies + a no-memory control behind one shared protocol
  judge.py       deterministic scoring + citation-trace interference classification
  llm.py         Ollama + Anthropic clients (prompt caching, determinism pins)
  stats.py       bootstrap CI + permutation testing
  run.py         wires it all together, writes scores.csv + transcripts.jsonl
scripts/analyze.py   the analysis behind every number in the article
experiments/results/ every run's raw scores + full transcripts (committed — this is the evidence)
```

## What's not in this article

Getting to a trustworthy 30-session harness took two failed local-model attempts first —
`qwen2.5:3b` hit a real 3B-parameter SQL ceiling, and `mistral:latest` executed a query, got the
*correct* answer back, and then hallucinated a fake dataset on the very next turn instead of using
it. That debugging trail — and six other harness bugs invisible to 72 passing unit tests until the
pipeline ran against a real model — is its own piece, not a footnote here.

## License

MIT — see [LICENSE](LICENSE). If this benchmark or its findings were useful to you, a
[sponsor](https://github.com/sponsors/EnigmaDevelop) is always appreciated but never expected.
