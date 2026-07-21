"""Statistical analysis of the frontier-arm (claude-sonnet-5) 3-seed dataset.

Loads experiments/results/{strategy}_anthropic_claude-sonnet-5_seed{0,1,2}/scores.csv
for all 4 strategies, and reports bootstrap CI + permutation test results — a
bare accuracy point isn't enough to support a "strategy X beats strategy Y"
claim:

1. Overall accuracy per strategy with a 95% bootstrap CI (pooled across 3 seeds).
2. Per-task-type (independent/dependent/trap) accuracy per strategy with CI —
   the meaningful breakdown, since 10/12/8 weighting is arbitrary.
3. Permutation-test p-values: none vs each memory strategy (does memory help?),
   and full vs rag / summary vs rag (does strategy choice matter?).
4. False-memory rate: confirmed-interference count over all trap-task rows.

Honesty note printed with the output: pooling 3 order-seeds of the *same* 30
task-content items across strategies is not 90 independent samples — it's 3
reshuffled looks at the same 30 items (a pseudoreplication validity threat).
Treat the CIs here as informative, not as a substitute for a genuinely
larger/varied task set.

Usage:
    uv run python -m scripts.analyze
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.stats import bootstrap_ci, permutation_test

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


def main() -> None:
    print("=" * 78)
    print("agent-memory-lab — 3-seed frontier-arm analysis (claude-sonnet-5)")
    print("=" * 78)
    print(
        "\nNote: pooling 3 order-seeds of the same 30 task-content items is 3\n"
        "reshuffled looks at one task set, not 90 independent samples — CIs\n"
        "below are informative, not a substitute for a larger/varied task set.\n"
    )

    all_rows: dict[str, list[dict]] = {s: load_rows(s) for s in STRATEGIES}

    print("-" * 78)
    print("1. Overall accuracy per strategy (pooled, 90 rows = 30 tasks x 3 seeds)")
    print("-" * 78)
    overall_values: dict[str, list[float]] = {}
    for strategy in STRATEGIES:
        values = [1.0 if r["correct"] else 0.0 for r in all_rows[strategy]]
        overall_values[strategy] = values
        result = bootstrap_ci(values, seed=0)
        print(
            f"  {strategy:<8} n={result.n:<3} accuracy={result.mean:.3f} "
            f"95% CI [{result.ci_low:.3f}, {result.ci_high:.3f}]"
        )

    print()
    print("-" * 78)
    print("2. Accuracy by task type (independent / dependent / trap)")
    print("-" * 78)
    for strategy in STRATEGIES:
        print(f"  {strategy}:")
        for task_type in ("independent", "dependent", "trap"):
            values = [1.0 if r["correct"] else 0.0 for r in all_rows[strategy] if r["task_type"] == task_type]
            result = bootstrap_ci(values, seed=0)
            print(
                f"    {task_type:<12} n={result.n:<3} accuracy={result.mean:.3f} "
                f"95% CI [{result.ci_low:.3f}, {result.ci_high:.3f}]"
            )

    print()
    print("-" * 78)
    print("3. Permutation tests (does the gap beat label-shuffling by chance?)")
    print("-" * 78)
    pairs = [
        ("none", "full"),
        ("none", "summary"),
        ("none", "rag"),
        ("full", "rag"),
        ("summary", "rag"),
        ("full", "summary"),
    ]
    for a, b in pairs:
        p = permutation_test(overall_values[a], overall_values[b], seed=0)
        gap = sum(overall_values[a]) / len(overall_values[a]) - sum(overall_values[b]) / len(overall_values[b])
        sig = "***" if p < 0.001 else ("*" if p < 0.05 else "n.s.")
        print(f"  {a:<8} vs {b:<8} gap={gap:+.3f}  p={p:.4f}  {sig}")

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


if __name__ == "__main__":
    main()
