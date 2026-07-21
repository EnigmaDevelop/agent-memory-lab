"""Independent-task capability calibration gate.

Early local-model testing found that a 3/10 independent-task result has a
~7%-65% 95% confidence interval — too wide to trust as "is this model
capable enough for the memory-strategy experiment?". This script runs a
candidate model against tasks/generate_calibration.py's larger
independent-only task set and checks it against a gate threshold that
MUST be declared before the run starts (via --gate), not chosen after
looking at the result — the point is avoiding post-hoc model-shopping.

No memory strategy is exercised here — every calibration task is
self-contained by construction, so this measures raw tool-use/SQL
competence in isolation from any memory-strategy effect.

Usage:
    uv run python -m src.calibrate --provider ollama --model mistral:latest --gate 0.80
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.agent import run_session
from src.judge import judge_session
from src.llm import LLMClient, OllamaClient
from src.memory.base import MemoryContext
from src.run import write_results
from tasks.generate_calibration import DEFAULT_N, generate as generate_calibration_tasks

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = ROOT / "environment" / "company.db"
DEFAULT_DOCS_DIR = ROOT / "environment" / "docs"
DEFAULT_RESULTS_DIR = ROOT / "experiments" / "results"


def build_llm_client(provider: str, model: str) -> LLMClient:
    if provider == "ollama":
        return OllamaClient(model=model)
    if provider == "anthropic":
        from src.llm import AnthropicClient

        return AnthropicClient(model=model)
    raise ValueError(f"Unknown provider: {provider!r} (choices: ollama, anthropic)")


def run_calibration(
    provider: str,
    model: str,
    n: int,
    seed: int,
    db_path: Path = DEFAULT_DB_PATH,
    docs_dir: Path = DEFAULT_DOCS_DIR,
    verbose: bool = True,
) -> tuple[list[dict], list[dict]]:
    llm_client = build_llm_client(provider, model)
    tasks = generate_calibration_tasks(n, seed, db_path)
    empty_context = MemoryContext(text="")

    rows: list[dict] = []
    transcripts: list[dict] = []
    for task in tasks:
        record = run_session(task, empty_context, llm_client, db_path, docs_dir)
        result = judge_session(task, record)
        if verbose:
            mark = "OK" if result.correct else "X "
            print(
                f"  [{mark}] session {task['session']:>2} {task['task_id']:<12} "
                f"-> {result.final_answer!r} (expected {result.expected!r})",
                flush=True,
            )
        rows.append(
            {
                "session": task["session"],
                "task_id": task["task_id"],
                "task_type": task["task_type"],
                "correct": result.correct,
                "category": result.category,
                "final_answer": result.final_answer,
                "expected": result.expected,
                "cited_source_session": result.cited_source_session,
                "memory_truncated": False,
                "memory_included_sessions": "",
            }
        )
        transcripts.append(
            {
                "session": task["session"],
                "task_id": task["task_id"],
                "task_prompt": task["prompt"],
                "memory_context": "",
                "transcript": record.transcript,
                "judge": {"correct": result.correct, "category": result.category},
            }
        )
    return rows, transcripts


def make_run_id(provider: str, model: str, seed: int, n: int) -> str:
    safe_model = model.replace(":", "-").replace("/", "-")
    return f"calibration_{provider}_{safe_model}_seed{seed}_n{n}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", required=True, choices=["ollama", "anthropic"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of calibration tasks")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gate",
        type=float,
        required=True,
        help=(
            "Pass threshold (0-1), e.g. 0.80. Must be decided BEFORE running "
            "and passed explicitly every time — there is no default, to force "
            "writing it down instead of picking it after seeing the result."
        ),
    )
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    print(f"GATE (declared before this run): independent-task accuracy >= {args.gate:.0%} required to pass")
    print(f"Model: {args.provider}/{args.model} | calibration set: n={args.n}, seed={args.seed}\n")

    rows, transcripts = run_calibration(args.provider, args.model, args.n, args.seed, args.db_path, args.docs_dir)
    run_id = make_run_id(args.provider, args.model, args.seed, args.n)
    run_dir = write_results(run_id, rows, transcripts, args.out_dir)

    n_correct = sum(1 for r in rows if r["correct"])
    accuracy = n_correct / len(rows)
    verdict = "PASS" if accuracy >= args.gate else "FAIL"
    print(f"\nCalibration {run_id}: {n_correct}/{len(rows)} = {accuracy:.1%} -> {run_dir}")
    print(f"GATE VERDICT: {verdict} (threshold was {args.gate:.0%}, declared before the run)")


if __name__ == "__main__":
    main()
