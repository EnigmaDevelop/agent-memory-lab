"""Statistical analysis of the frontier-arm (claude-sonnet-5) 3-seed dataset.

Loads experiments/results/{strategy}_anthropic_claude-sonnet-5_seed{0,1,2}/scores.csv
for all 4 strategies, and reports bootstrap CI + permutation test results — a
bare accuracy point isn't enough to support a "strategy X beats strategy Y"
claim:

1. Overall accuracy per strategy with a 95% bootstrap CI (pooled across 3 seeds).
2. Per-task-type (independent/dependent/trap) accuracy per strategy with CI —
   the meaningful breakdown, since 10/12/8 weighting is arbitrary.
3. Paired permutation-test p-values: none vs each memory strategy (does memory
   help?), and full vs rag / summary vs rag (does strategy choice matter?).
4. False-memory rate: confirmed-interference count over all trap-task rows.
5. Error clustering by task family — the check that motivated the paired test.

The pairing is the point (fixed 2026-07-22). Every condition runs the same 30
task items in the same 3 orderings, so each row has a natural partner: same
`task_id`, same seed. An unpaired test that shuffles the condition label across
the pooled 90 rows treats those as 90 independent draws, which they are not.
Section 5 below shows how badly that assumption fails on this dataset: all 6
wrong answers produced by the three memory conditions fall in one task family
that accounts for only 18 of the 90 attempts.

The pseudoreplication caveat still stands even with the paired test — 3
reshuffled orderings of one 30-item task set is 3 looks at one set, not 90
independent draws. Pairing fixes the *test*, not the *sample*.

Usage:
    uv run python -m scripts.analyze
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.stats import (
    clustered_bootstrap_ci,
    count_discordant,
    paired_permutation_test,
    wilson_ci,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "experiments" / "results"

STRATEGIES = ("none", "full", "summary", "rag")
SEEDS = (0, 1, 2)
MODEL_SLUG = "anthropic_claude-sonnet-5"


def load_rows(strategy: str) -> list[dict]:
    rows = []
    for seed in SEEDS:
        path = RESULTS_DIR / f"{strategy}_{MODEL_SLUG}_seed{seed}" / "scores.csv"
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["seed"] = seed
                row["correct"] = row["correct"] == "True"
                rows.append(row)
    return rows


def keyed(rows: list[dict]) -> dict[str, float]:
    """(task_id, seed) -> 0/1 correctness, the matching key for the paired test."""
    return {f"{r['task_id']}|{r['seed']}": (1.0 if r["correct"] else 0.0) for r in rows}


def format_ci(rows: list[dict]) -> str:
    """Cluster-bootstrap CI (one cluster per task_id), Wilson when zero-variance.

    Clustering matters: the rows are 30 task items seen under 3 orderings, not
    90 independent draws, so the interval is computed over task-clusters. When
    a condition has zero variance (e.g. 90/90) the bootstrap degenerates to a
    point, so we report a Wilson interval instead — fed one value per *cluster*
    (task_id), so it reflects ~30 independent units, not an inflated 90.
    """
    values = [1.0 if r["correct"] else 0.0 for r in rows]
    clusters = [r["task_id"] for r in rows]
    boot = clustered_bootstrap_ci(values, clusters, seed=0)
    if not boot.degenerate:
        return (
            f"n={len(values):<3} (clusters={boot.n:<2}) accuracy={boot.mean:.3f} "
            f"95% CI [{boot.ci_low:.3f}, {boot.ci_high:.3f}] (cluster bootstrap)"
        )
    # One row per cluster so Wilson's n is the independent-unit count.
    per_cluster = [1.0 if all(r["correct"] for r in rows if r["task_id"] == tid) else 0.0 for tid in sorted(set(clusters))]
    w = wilson_ci(per_cluster)
    return (
        f"n={len(values):<3} (clusters={w.n:<2}) accuracy={boot.mean:.3f} "
        f"95% CI [{w.ci_low:.3f}, {w.ci_high:.3f}] (Wilson on clusters; bootstrap degenerate)"
    )


def main() -> None:
    print("=" * 78)
    print("agent-memory-lab — 3-seed frontier-arm analysis (claude-sonnet-5)")
    print("=" * 78)
    print(
        "\nNote: pooling 3 order-seeds of the same 30 task-content items is 3\n"
        "reshuffled looks at one task set, not 90 independent samples — CIs\n"
        "below are informative, not a substitute for a larger/varied task set.\n"
        "Significance tests are PAIRED on (task_id, seed) for that reason.\n"
    )

    all_rows: dict[str, list[dict]] = {s: load_rows(s) for s in STRATEGIES}

    print("-" * 78)
    print("1. Overall accuracy per strategy (pooled, 90 rows = 30 tasks x 3 seeds)")
    print("-" * 78)
    overall_values: dict[str, list[float]] = {}
    for strategy in STRATEGIES:
        values = [1.0 if r["correct"] else 0.0 for r in all_rows[strategy]]
        overall_values[strategy] = values
        print(f"  {strategy:<8} {format_ci(all_rows[strategy])}")

    print()
    print("-" * 78)
    print("2. Accuracy by task type (independent / dependent / trap)")
    print("-" * 78)
    for strategy in STRATEGIES:
        print(f"  {strategy}:")
        for task_type in ("independent", "dependent", "trap"):
            subset = [r for r in all_rows[strategy] if r["task_type"] == task_type]
            print(f"    {task_type:<12} {format_ci(subset)}")

    print()
    print("-" * 78)
    print("3. Paired permutation tests, matched on (task_id, seed)")
    print("-" * 78)
    print("   discordant = pairs where the two conditions disagree; only those carry signal.")
    keyed_values = {s: keyed(all_rows[s]) for s in STRATEGIES}
    pairs = [
        ("none", "full"),
        ("none", "summary"),
        ("none", "rag"),
        ("full", "rag"),
        ("summary", "rag"),
        ("full", "summary"),
    ]
    for a, b in pairs:
        p = paired_permutation_test(keyed_values[a], keyed_values[b], seed=0)
        a_wins, b_wins = count_discordant(keyed_values[a], keyed_values[b])
        gap = sum(overall_values[a]) / len(overall_values[a]) - sum(overall_values[b]) / len(overall_values[b])
        sig = "***" if p < 0.001 else ("*" if p < 0.05 else "n.s.")
        print(
            f"  {a:<8} vs {b:<8} gap={gap:+.3f}  p={p:.4f}  {sig:<4} "
            f"discordant: {a} +{a_wins} / {b} +{b_wins}"
        )

    print()
    print("-" * 78)
    print("4. False-memory rate (confirmed interference, trap tasks only)")
    print("-" * 78)
    for strategy in STRATEGIES:
        trap_rows = [r for r in all_rows[strategy] if r["task_type"] == "trap"]
        n_trap = len(trap_rows)
        n_confirmed = sum(1 for r in trap_rows if r["category"] == "interference_confirmed")
        n_suspected = sum(1 for r in trap_rows if r["category"] == "interference_suspected")
        rate = n_confirmed / n_trap if n_trap else float("nan")
        print(
            f"  {strategy:<8} trap_n={n_trap:<3} confirmed={n_confirmed}  suspected={n_suspected}  "
            f"false_memory_rate={rate:.3f}"
        )

    print()
    print("-" * 78)
    print("5. Error clustering by task family (why the tests above are paired)")
    print("-" * 78)
    family_totals: dict[str, int] = {}
    for r in all_rows["rag"]:
        family = r["task_id"].split("_")[0]
        family_totals[family] = family_totals.get(family, 0) + 1
    for strategy in STRATEGIES:
        errors: dict[str, int] = {}
        for r in all_rows[strategy]:
            if not r["correct"]:
                family = r["task_id"].split("_")[0]
                errors[family] = errors.get(family, 0) + 1
        total_errors = sum(errors.values())
        spread = ", ".join(f"{fam}={n}/{family_totals[fam]}" for fam, n in sorted(errors.items()))
        print(f"  {strategy:<8} errors={total_errors:<3} {spread or '(none)'}")
    memory_errors: dict[str, int] = {}
    for strategy in ("full", "summary", "rag"):
        for r in all_rows[strategy]:
            if not r["correct"]:
                family = r["task_id"].split("_")[0]
                memory_errors[family] = memory_errors.get(family, 0) + 1
    total_memory_errors = sum(memory_errors.values())
    print(
        f"\n  memory conditions combined: {total_memory_errors} errors across "
        f"{len(memory_errors)} of {len(family_totals)} task families"
    )
    for fam, n in sorted(memory_errors.items(), key=lambda kv: -kv[1]):
        share = family_totals[fam] / sum(family_totals.values())
        print(f"    {fam:<8} {n} errors; family is {family_totals[fam]}/{sum(family_totals.values())} of attempts ({share:.1%})")

    print()
    print("-" * 78)
    print("6. Agent turns per session (the cost mechanism, measured not assumed)")
    print("-" * 78)
    print("   budget is 8 assistant turns; hitting it means the agent kept retrying.")
    for strategy in STRATEGIES:
        turns = []
        for seed in SEEDS:
            path = RESULTS_DIR / f"{strategy}_{MODEL_SLUG}_seed{seed}" / "transcripts.jsonl"
            with path.open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    turns.append(sum(1 for m in row["transcript"] if m.get("role") == "assistant"))
        turns.sort()
        mean = sum(turns) / len(turns)
        median = turns[len(turns) // 2]
        at_budget = sum(1 for t in turns if t >= 8)
        print(
            f"  {strategy:<8} mean={mean:.2f}  median={median}  "
            f"hit_budget(>=8)={at_budget}/{len(turns)} ({at_budget / len(turns):.0%})"
        )


if __name__ == "__main__":
    main()
