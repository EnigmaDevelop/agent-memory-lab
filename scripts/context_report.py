"""How large did each memory strategy's context actually get?

Added 2026-07-22. The write-up reported full-history's 100% accuracy without
ever stating how big the full history got — which is the number a reader needs
in order to know what the result covers. A "context rot" hypothesis is only
tested if the run actually reaches a context size where rot is plausible; this
script reports whether it did.

Character counts come straight from the `memory_context` field recorded for
every session in `transcripts.jsonl`. The token figures are a coarse chars/4
estimate, not a tokenizer count — no per-call token usage was logged during the
runs, so an exact figure is not recoverable without re-running the study. The
estimate is labelled as such everywhere it is reported.

Usage:
    uv run python -m scripts.context_report
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "experiments" / "results"

STRATEGIES = ("none", "full", "summary", "rag")
SEEDS = (0, 1, 2)
MODEL_SLUG = "anthropic_claude-sonnet-5"

# Coarse chars-per-token ratio for English prose. Only used for the estimate
# column, which is always labelled "~est".
CHARS_PER_TOKEN = 4


def main() -> None:
    print("=" * 78)
    print("Memory context size by strategy (chars measured; tokens estimated)")
    print("=" * 78)
    print("\nToken column is a chars/4 ESTIMATE — no per-call token usage was logged.\n")

    for strategy in STRATEGIES:
        print(f"  {strategy}:")
        for seed in SEEDS:
            run_dir = RESULTS_DIR / f"{strategy}_{MODEL_SLUG}_seed{seed}"
            path = run_dir / "transcripts.jsonl"
            sizes = []
            with path.open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    sizes.append(len(row["memory_context"] or ""))

            truncated = 0
            with (run_dir / "scores.csv").open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row["memory_truncated"] == "True":
                        truncated += 1

            final = sizes[-1]
            peak = max(sizes)
            print(
                f"    seed{seed}  sessions={len(sizes):<3} "
                f"final={final:>6} chars (~{final // CHARS_PER_TOKEN:>5} tok est)  "
                f"peak={peak:>6} chars (~{peak // CHARS_PER_TOKEN:>5} tok est)  "
                f"truncated_sessions={truncated}"
            )
        print()


if __name__ == "__main__":
    main()
