<p align="center"><strong>agent-memory-lab</strong></p>

<p align="center">
  Does giving an LLM agent memory of its own past sessions help, hurt, or both?<br>
  I measured it — 360 sessions, 3 memory strategies + a no-memory control, real billing data.
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <a href="https://github.com/sponsors/EnigmaDevelop"><img alt="Sponsor" src="https://img.shields.io/badge/%E2%9D%A4-Sponsor-ea4aaa.svg"></a>
</p>

Full writeup: **[I Benchmarked 3 AI Agent Memory Strategies. Only RAG Cited a Dead Decision as Current.](https://medium.com/gitconnected/i-benchmarked-3-ai-agent-memory-strategies-only-rag-cited-a-dead-decision-as-current-feb4cce68240)** — Level Up Coding.

## The finding

Four conditions, the same 30-task sequence, 3 reshuffled orderings, a citation-trace judge that
distinguishes *any* wrong answer on a trap task from a *provable* citation of a specific stale
session:

| Condition | Accuracy (90 attempts) | 95% CI* | Cost / 30-session run, one seed** |
|---|---|---|---|
| No memory (control) | 32.2% | [16.7%, 47.8%] | **$1.15** |
| Full history replay | **100.0%** | [88.6%, 100.0%]† | $0.45 |
| Rolling summary | 95.6% | [90.0%, 100.0%] | $0.32 |
| RAG (BM25 retrieval) | 97.8% | [94.4%, 100.0%] | $0.22 |

\* Cluster bootstrap over 30 task-clusters, not 90 rows — the attempts are 30 tasks seen under 3
orderings, not 90 independent draws, so the intervals reflect ~30 independent units (wider, honest).
\** per-minute Anthropic billing data, averaged over 2 of the 3 seeds; one seed across all four
conditions is ~$2.1, the full 3-seed study ~$6-7. Reconstructed from billing timestamps, not
computed from the runs — no per-call token usage was logged.
† Wilson interval fed one value per task-cluster. A bootstrap of a zero-variance sample returns
`[100%, 100%]`, which describes the sample rather than the uncertainty; `src/stats.py` detects that.

Significance tests are **paired** on `(task_id, seed)` — every condition runs the same 30 items in
the same 3 orderings, so the attempts are not independent draws. Every memory strategy beats the
control at p<0.0001; no memory-vs-memory pair is significant (p = 0.13 to 0.63).

Three things that weren't the expected result going in:

1. **The no-memory control was the most expensive condition to run**, not the cheapest — the
   transcripts measure the mechanism: no-memory ran a median of 8/8 agent turns and hit its retry
   budget on 56/90 sessions; every memory strategy ran a median of 2 turns and hit it zero times.
2. **Exactly one of 270 memory-backed attempts produced a confirmed false memory** (an answer
   matching a known-stale value *and* citing the superseded session) — and it was RAG, the
   strategy usually assumed to be the safe, grounded one, not full-history replay. In that one case
   the revising session was absent from the retrieved context — but that absence was not the
   pattern: BM25 retrieved the revising session in **22 of 24** RAG trap attempts, and only one of
   the two misses became a wrong answer. A retrieval miss was present in the confirmed failure
   without being sufficient to cause one (`scripts/exposure_report.py`).
3. **All 6 errors the memory strategies made land in one of the 6 task families**, which is only
   18 of each condition's 90 attempts. The benchmark's difficulty is concentrated far more than
   the headline accuracies suggest — `scripts/analyze.py` section 5 prints this.

Scope worth knowing before generalizing: full-history's context peaked around **4,300 tokens**
after 30 sessions, with zero truncation anywhere (`scripts/context_report.py`). Its perfect score
says interference didn't occur at this scale — not that replay resists it at any scale.

## Reproduce it

```bash
uv sync --extra frontier          # installs the anthropic SDK too
uv run environment/build_env.py --seed 42        # deterministic fictional company DB + docs
uv run python -m tasks.generate --seed 0         # 30-task sequence for this seed

export ANTHROPIC_API_KEY=sk-ant-...
uv run python -m src.run --strategy rag --provider anthropic --model claude-sonnet-5 --seed 0

uv run pytest                                     # 88 unit tests, fully deterministic, no network
uv run python -m scripts.analyze                  # CIs + paired permutation tests over all seeds
uv run python -m scripts.context_report           # how large each strategy's context actually got
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
