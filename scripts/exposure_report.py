"""What was the denominator, actually?

Added 2026-07-23. The write-up reported "one confirmed false memory across 270
memory-backed attempts" and left it there. That is one number answering one
question, and the article it supports is about the fact that a single failure
count answers several different questions depending on where in the causal
funnel you measure it.

The funnel this script reconstructs, for BM25 retrieval:

    90 interactions
     -> 24 revision-conflict challenges   (trap tasks: a fact was revised earlier)
         -> N where retrieval missed the revising session
             -> 1 confirmed stale-memory answer

Each stage is a different denominator attached to a different engineering
question — operational incidence, challenge-conditioned susceptibility, and
post-retrieval-miss failure. Section 2 prints all of them side by side.

Two guards live here because both were real mistakes caught during planning:

- Section 1 prints trap attempts PER CONDITION and pooled. The trap population
  is 24 per condition, so pooling across the three memory strategies gives 72,
  not 24. An early draft of the article compared a pooled numerator (1 error
  across all memory strategies) against a per-strategy denominator (24) — the
  same species of error the article is about.

- Section 4 checks that every ordering preserves establish -> revise -> trap.
  This is what makes "exposed to a revision conflict" a property of the task
  set, fixed before the agent runs, rather than a label applied after seeing
  which answers were wrong. If any ordering violated it, the exposure
  population would be a hindsight construct and the article's central move
  would be invalid.

Retrieval hits are read from the `memory_context` field, which records each
session's retrieved chunks as `[Session N | sN_cM]`. `MemoryContext.retrieved`
— the structured field carrying BM25 scores — is never persisted to
`transcripts.jsonl`, so the session ids are parsed back out of the rendered
text. That is a faithful record of what the agent saw, not a re-run.

Usage:
    PYTHONIOENCODING=utf-8 uv run python -m scripts.exposure_report
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from tasks.generate import generate as generate_tasks

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "experiments" / "results"
DB_PATH = ROOT / "environment" / "company.db"

STRATEGIES = ("none", "full", "summary", "rag")
MEMORY_STRATEGIES = ("full", "summary", "rag")
SEEDS = (0, 1, 2)
MODEL_SLUG = "anthropic_claude-sonnet-5"

RETRIEVED_SESSION_RE = re.compile(r"\[Session (\d+) \| s\d+_c\d+\]")


def run_dir(strategy: str, seed: int) -> Path:
    return RESULTS_DIR / f"{strategy}_{MODEL_SLUG}_seed{seed}"


def load_scores(strategy: str, seed: int) -> list[dict]:
    with (run_dir(strategy, seed) / "scores.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_transcripts(strategy: str, seed: int) -> dict[str, dict]:
    """task_id -> transcript row, for the runs where we need memory_context."""
    rows = {}
    with (run_dir(strategy, seed) / "transcripts.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[row["task_id"]] = row
    return rows


def tasks_by_id(seed: int) -> dict[str, dict]:
    """Per-seed task set. tasks/tasks.yaml is seed 0 only — session numbers, and
    therefore expected_source_session, differ per ordering, so regenerate."""
    return {t["task_id"]: t for t in generate_tasks(seed, str(DB_PATH))}


def retrieved_sessions(row: dict) -> set[int]:
    return {int(m) for m in RETRIEVED_SESSION_RE.findall(row.get("memory_context") or "")}


def main() -> None:
    print("=" * 78)
    print("agent-memory-lab — exposure populations and the failure funnel")
    print("=" * 78)
    print(
        "\nOne confirmed false memory. This script prints every denominator it can\n"
        "legitimately be divided by, and which question each one answers.\n"
    )

    all_tasks = {seed: tasks_by_id(seed) for seed in SEEDS}

    # ---------------------------------------------------------------- section 1
    print("-" * 78)
    print("1. Attempts and exposure population, per condition and pooled")
    print("-" * 78)
    print("   'exposed' = trap task: consumes a fact that was revised in an earlier")
    print("   session. Fixed when the task set is generated, before the agent runs.")
    totals = {}
    for strategy in STRATEGIES:
        attempts = exposed = confirmed = 0
        for seed in SEEDS:
            for row in load_scores(strategy, seed):
                attempts += 1
                if row["task_type"] == "trap":
                    exposed += 1
                    if row["category"] == "interference_confirmed":
                        confirmed += 1
        totals[strategy] = (attempts, exposed, confirmed)
        print(f"  {strategy:<8} attempts={attempts:<4} exposed={exposed:<4} confirmed_false_memory={confirmed}")

    pooled_attempts = sum(totals[s][0] for s in MEMORY_STRATEGIES)
    pooled_exposed = sum(totals[s][1] for s in MEMORY_STRATEGIES)
    pooled_confirmed = sum(totals[s][2] for s in MEMORY_STRATEGIES)
    print()
    print(f"  pooled over the 3 memory strategies: attempts={pooled_attempts}  "
          f"exposed={pooled_exposed}  confirmed={pooled_confirmed}")
    print(f"  -> the partner of 1/{pooled_attempts} is 1/{pooled_exposed}, NOT 1/{totals['rag'][1]}.")
    print(f"     1/{totals['rag'][1]} is a per-condition figure and pairs with 1/{totals['rag'][0]}.")

    # ---------------------------------------------------------------- section 2
    print()
    print("-" * 78)
    print("2. The failure funnel (BM25 retrieval)")
    print("-" * 78)
    attempts, exposed, confirmed = totals["rag"]
    missed = 0
    missed_detail = []
    for seed in SEEDS:
        tasks = all_tasks[seed]
        transcripts = load_transcripts("rag", seed)
        for row in load_scores("rag", seed):
            if row["task_type"] != "trap":
                continue
            task = tasks[row["task_id"]]
            revision_session = task["expected_source_session"]
            got = retrieved_sessions(transcripts[row["task_id"]])
            if revision_session not in got:
                missed += 1
                missed_detail.append(
                    (seed, row["task_id"], revision_session, sorted(got), row["category"])
                )

    print(f"  {attempts:>4} interactions")
    print(f"  {exposed:>4} revision-conflict challenges")
    print(f"  {missed:>4} where retrieval missed the revising session")
    print(f"  {confirmed:>4} confirmed stale-memory answers")
    print()
    print("  Three estimands, three owners:")
    print(f"    operational incidence           {confirmed}/{attempts}   "
          f"({confirmed / attempts:.1%})  — how often does a user see it")
    print(f"    challenge-conditioned failure   {confirmed}/{exposed}   "
          f"({confirmed / exposed:.1%})  — what happens when a fact has been revised")
    print(f"    failure after a retrieval miss  {confirmed}/{missed}    "
          f"({confirmed / missed:.1%})  — DESCRIPTIVE ONLY, n={missed}")
    print()
    print(f"  The last line is not an estimate of anything. n={missed} cannot support a rate;")
    print("  it is printed to show how large a 'rate' becomes in a thin enough slice.")

    # ---------------------------------------------------------------- section 3
    print()
    print("-" * 78)
    print("3. Revision-session retrieval, per seed (the misses, in full)")
    print("-" * 78)
    print(f"  revision session retrieved in {exposed - missed}/{exposed} trap attempts")
    if missed_detail:
        for seed, task_id, revision_session, got, category in missed_detail:
            print(f"    MISS seed{seed} {task_id:<12} revision=s{revision_session:<3} "
                  f"retrieved={got}  judge={category}")
        print()
        print("  A retrieval miss is necessary but not sufficient: the misses above do not")
        print("  all become errors. That gap is the whole reason the funnel has four levels")
        print("  instead of two.")

    # ---------------------------------------------------------------- section 4
    print()
    print("-" * 78)
    print("4. Chronology guard — is 'exposed' a design property or hindsight?")
    print("-" * 78)
    print("   Every trap task must come after the revision, which must come after the")
    print("   establishing session. If this fails anywhere, the exposure population is")
    print("   a post-hoc label and section 1's denominator is not defensible.")
    violations = 0
    for seed in SEEDS:
        tasks = all_tasks[seed]
        facts = sorted({t["fact_id"] for t in tasks.values() if t.get("role") == "revise"})
        for fact in facts:
            establish = next(t for t in tasks.values() if t["fact_id"] == fact and t["role"] == "establish")
            revise = next(t for t in tasks.values() if t["fact_id"] == fact and t["role"] == "revise")
            traps = sorted(t["session"] for t in tasks.values()
                           if t["fact_id"] == fact and t["task_type"] == "trap")
            bad = [s for s in traps if not establish["session"] < revise["session"] < s]
            violations += len(bad)
            print(f"  seed{seed} {fact:<28} establish=s{establish['session']:<3} "
                  f"revise=s{revise['session']:<3} traps={traps}  violations={bad or 'none'}")
    print()
    print(f"  TOTAL VIOLATIONS: {violations}"
          f"{'  — exposure is a design property, as claimed' if violations == 0 else '  — CLAIM INVALID'}")


if __name__ == "__main__":
    main()
