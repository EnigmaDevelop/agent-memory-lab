"""Orchestrates one strategy x model x seed run across all 30 sessions.

Wires: tasks.generate (per-seed session order) -> memory strategy
(build_context) -> agent.run_session -> judge.judge_session -> memory
strategy (on_session_end), then writes both a scoring CSV and a full
transcript JSONL into `experiments/results/`.

`--seed` controls session order only (task content is fixed — see
tasks/generate.py's CONTENT_SEED), matching the order-effect control the
experiment design relies on: the same 30 tasks, reshuffled, so a strategy
gap can't be explained by "it happened to get the easy ordering."

Usage:
    uv run python -m src.run --strategy none --provider ollama --model qwen2.5:3b --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.agent import run_session
from src.judge import judge_session
from src.llm import LLMClient, OllamaClient
from src.memory.base import MemoryStrategy
from src.memory.full import FullHistoryMemory
from src.memory.none import NoMemory
from src.memory.rag import RAGMemory
from src.memory.summary import RollingSummaryMemory
from tasks.generate import generate as generate_tasks

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = ROOT / "environment" / "company.db"
DEFAULT_DOCS_DIR = ROOT / "environment" / "docs"
DEFAULT_RESULTS_DIR = ROOT / "experiments" / "results"

STRATEGIES = ("none", "full", "summary", "rag")


def build_strategy(name: str, llm_client: LLMClient) -> MemoryStrategy:
    if name == "none":
        return NoMemory()
    if name == "full":
        return FullHistoryMemory()
    if name == "summary":
        return RollingSummaryMemory(llm_client)
    if name == "rag":
        return RAGMemory()
    raise ValueError(f"Unknown strategy: {name!r} (choices: {STRATEGIES})")


def build_llm_client(provider: str, model: str) -> LLMClient:
    if provider == "ollama":
        return OllamaClient(model=model)
    if provider == "anthropic":
        from src.llm import AnthropicClient  # lazy: keeps the local-only path key-free

        return AnthropicClient(model=model)
    raise ValueError(f"Unknown provider: {provider!r} (choices: ollama, anthropic)")


def run_experiment(
    strategy_name: str,
    provider: str,
    model: str,
    seed: int,
    db_path: Path = DEFAULT_DB_PATH,
    docs_dir: Path = DEFAULT_DOCS_DIR,
    verbose: bool = True,
) -> tuple[list[dict], list[dict]]:
    llm_client = build_llm_client(provider, model)
    strategy = build_strategy(strategy_name, llm_client)
    tasks = generate_tasks(seed, db_path)  # already sorted by session

    rows: list[dict] = []
    transcripts: list[dict] = []
    for task in tasks:
        memory_context = strategy.build_context(task)
        record = run_session(task, memory_context, llm_client, db_path, docs_dir)
        strategy.on_session_end(record)
        result = judge_session(task, record)
        if verbose:
            mark = "OK" if result.correct else "X "
            print(
                f"  [{mark}] session {task['session']:>2} {task['task_type']:<11} {task['task_id']:<14} "
                f"-> {result.final_answer!r} (expected {result.expected!r}, {result.category})",
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
                "memory_truncated": memory_context.truncated,
                "memory_included_sessions": ";".join(str(s) for s in memory_context.included_sessions),
            }
        )
        transcripts.append(
            {
                "session": task["session"],
                "task_id": task["task_id"],
                "task_prompt": task["prompt"],
                "memory_context": memory_context.text,
                "transcript": record.transcript,
                "judge": {"correct": result.correct, "category": result.category},
            }
        )
    return rows, transcripts


def make_run_id(strategy: str, provider: str, model: str, seed: int) -> str:
    safe_model = model.replace(":", "-").replace("/", "-")
    return f"{strategy}_{provider}_{safe_model}_seed{seed}"


def write_results(run_id: str, rows: list[dict], transcripts: list[dict], out_dir: Path) -> Path:
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "scores.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (run_dir / "transcripts.jsonl").open("w", encoding="utf-8") as f:
        for t in transcripts:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True, choices=STRATEGIES)
    parser.add_argument("--provider", required=True, choices=["ollama", "anthropic"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=0, help="Session-order seed (see tasks/generate.py)")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    rows, transcripts = run_experiment(args.strategy, args.provider, args.model, args.seed, args.db_path, args.docs_dir)
    run_id = make_run_id(args.strategy, args.provider, args.model, args.seed)
    run_dir = write_results(run_id, rows, transcripts, args.out_dir)

    n_correct = sum(1 for r in rows if r["correct"])
    print(f"Run {run_id}: {n_correct}/{len(rows)} correct -> {run_dir}")
    by_type: dict[str, dict] = {}
    for r in rows:
        d = by_type.setdefault(r["task_type"], {"correct": 0, "total": 0})
        d["total"] += 1
        d["correct"] += int(r["correct"])
    for task_type, d in sorted(by_type.items()):
        print(f"  {task_type}: {d['correct']}/{d['total']}")


if __name__ == "__main__":
    main()
