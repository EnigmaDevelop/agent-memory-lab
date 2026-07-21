"""Independent-only calibration task generator.

Early local-model testing found that a single 10-item independent-task run
cannot reliably answer "is this model capable enough to trust a
memory-strategy comparison?" — a 3-out-of-10 result has a ~7%-65% 95% confidence interval
(binomial). This module generates a larger (default 24), fully independent
task set — no cross-session dependencies, no memory required, no traps —
purely to measure a candidate model's raw tool-use/SQL competence *before*
committing it to the expensive full 30-session x 4-strategy experiment.

This is deliberately a separate artifact from tasks/generate.py's 30-task
battery: mixing more independent items into that battery would change its
fixed 10/12/8 composition, which downstream analysis assumes.

Every answer is computed by running real SQL against the environment's
company.db — nothing hand-typed — matching tasks/generate.py's convention.

Usage:
    uv run python -m tasks.generate_calibration --seed 0
    uv run python -m tasks.generate_calibration --n 24 --out tasks/calibration.yaml
"""

from __future__ import annotations

import argparse
import random
import sqlite3
from pathlib import Path
from typing import Any, Callable

import yaml

from environment.build_env import REFERENCE_DATE

TASKS_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = TASKS_DIR.parent / "environment" / "company.db"
DEFAULT_OUT_PATH = TASKS_DIR / "calibration.yaml"

# Task *content* is fixed regardless of --seed (matches generate.py's
# CONTENT_SEED convention) — --seed only reorders the session numbering,
# which barely matters here since every task is independent, but keeping
# the same seed/order-vs-content separation avoids surprises for anyone
# reusing this alongside the main generator.
CONTENT_SEED = 0
DEFAULT_N = 24


def load_domain_values(conn: sqlite3.Connection) -> dict[str, list]:
    return {
        "industries": [r[0] for r in conn.execute("SELECT DISTINCT industry FROM customers ORDER BY 1")],
        "countries": [r[0] for r in conn.execute("SELECT DISTINCT country FROM customers ORDER BY 1")],
        "plan_tiers": [r[0] for r in conn.execute("SELECT DISTINCT plan_tier FROM customers ORDER BY 1")],
        "statuses": [r[0] for r in conn.execute("SELECT DISTINCT status FROM customers ORDER BY 1")],
        "teams": [r[0] for r in conn.execute("SELECT DISTINCT team FROM employees ORDER BY 1")],
        "ticket_categories": [r[0] for r in conn.execute("SELECT DISTINCT category FROM support_tickets ORDER BY 1")],
        "ticket_priorities": [r[0] for r in conn.execute("SELECT DISTINCT priority FROM support_tickets ORDER BY 1")],
        "event_types": [r[0] for r in conn.execute("SELECT DISTINCT event_type FROM events ORDER BY 1")],
        "invoice_statuses": [r[0] for r in conn.execute("SELECT DISTINCT status FROM invoices ORDER BY 1")],
        "campaign_channels": [r[0] for r in conn.execute("SELECT DISTINCT channel FROM campaigns ORDER BY 1")],
        "campaign_ids": [r[0] for r in conn.execute("SELECT campaign_id FROM campaigns ORDER BY 1")],
        "campaign_names": dict(conn.execute("SELECT campaign_id, name FROM campaigns")),
        "customer_ids": [r[0] for r in conn.execute("SELECT customer_id FROM customers ORDER BY 1")],
        "customer_names": dict(conn.execute("SELECT customer_id, name FROM customers")),
        "employee_ids": [r[0] for r in conn.execute("SELECT employee_id FROM employees ORDER BY 1")],
        "employee_names": dict(conn.execute("SELECT employee_id, name FROM employees")),
        "signup_min_max": conn.execute("SELECT MIN(signup_date), MAX(signup_date) FROM customers").fetchone(),
    }


def _mid_date(domain: dict) -> str:
    from datetime import date

    lo = date.fromisoformat(domain["signup_min_max"][0])
    hi = date.fromisoformat(domain["signup_min_max"][1])
    return (lo + (hi - lo) / 2).isoformat()


# --- Template family builders --------------------------------------------
#
# Each family returns a list of (prompt, answer, answer_type) triples. Every
# family is a distinct SQL shape (single-table filter, join, date-range,
# aggregation) so the calibration set stresses the same failure modes
# observed live: bad joins, wrong table, unwanted extra filters, date
# handling. None of these touch customers_legacy or any stale/superseded
# doc — that adversarial dimension belongs to the main 30-task battery's
# trap tasks, not this raw-capability calibration set.


def family_tickets_by_priority(conn, domain) -> list[tuple[str, Any, str]]:
    out = []
    for priority in domain["ticket_priorities"]:
        answer = conn.execute("SELECT COUNT(*) FROM support_tickets WHERE priority = ?", [priority]).fetchone()[0]
        out.append((f"How many support tickets currently have priority '{priority}'?", answer, "int"))
    return out


def family_tickets_by_category(conn, domain) -> list[tuple[str, Any, str]]:
    out = []
    for category in domain["ticket_categories"][:3]:
        answer = conn.execute("SELECT COUNT(*) FROM support_tickets WHERE category = ?", [category]).fetchone()[0]
        out.append((f"How many support tickets are there in the '{category}' category?", answer, "int"))
    return out


def family_invoice_total_by_customer(conn, domain, rng: random.Random) -> list[tuple[str, Any, str]]:
    out = []
    ids = rng.sample(domain["customer_ids"], k=3)
    for cid in ids:
        name = domain["customer_names"][cid]
        answer = conn.execute("SELECT COALESCE(SUM(amount),0) FROM invoices WHERE customer_id = ?", [cid]).fetchone()[0]
        out.append((f"What is the total invoice amount on record for customer '{name}'?", answer, "int"))
    return out


def family_employees_by_team(conn, domain) -> list[tuple[str, Any, str]]:
    out = []
    for team in domain["teams"][:3]:
        answer = conn.execute("SELECT COUNT(*) FROM employees WHERE team = ?", [team]).fetchone()[0]
        out.append((f"How many employees are on the {team} team?", answer, "int"))
    return out


def family_customers_by_industry(conn, domain, rng: random.Random) -> list[tuple[str, Any, str]]:
    out = []
    for industry in rng.sample(domain["industries"], k=2):
        answer = conn.execute("SELECT COUNT(*) FROM customers WHERE industry = ?", [industry]).fetchone()[0]
        out.append((f"How many customers are in the {industry} industry?", answer, "int"))
    return out


def family_active_subs_mrr_by_plan(conn, domain) -> list[tuple[str, Any, str]]:
    out = []
    for tier in domain["plan_tiers"]:
        answer = conn.execute(
            "SELECT COALESCE(SUM(mrr),0) FROM subscriptions WHERE status = 'active' AND plan_tier = ?", [tier]
        ).fetchone()[0]
        out.append((f"What is the total MRR from active subscriptions on the {tier} plan?", answer, "int"))
    return out


def family_customers_by_status_and_country(conn, domain, rng: random.Random) -> list[tuple[str, Any, str]]:
    out = []
    for country in rng.sample(domain["countries"], k=2):
        answer = conn.execute(
            "SELECT COUNT(*) FROM customers WHERE status = 'active' AND country = ?", [country]
        ).fetchone()[0]
        out.append((f"How many active customers are based in {country}?", answer, "int"))
    return out


def family_events_by_type(conn, domain) -> list[tuple[str, Any, str]]:
    out = []
    for event_type in domain["event_types"][:2]:
        answer = conn.execute("SELECT COUNT(*) FROM events WHERE event_type = ?", [event_type]).fetchone()[0]
        out.append((f"How many '{event_type}' events are recorded in total?", answer, "int"))
    return out


def family_campaign_budget(conn, domain) -> list[tuple[str, Any, str]]:
    out = []
    for cid in domain["campaign_ids"][:2]:
        name = domain["campaign_names"][cid]
        answer = conn.execute("SELECT budget FROM campaigns WHERE campaign_id = ?", [cid]).fetchone()[0]
        out.append((f"What is the budget for the '{name}' campaign?", answer, "int"))
    return out


def family_campaign_attribution_count(conn, domain) -> list[tuple[str, Any, str]]:
    out = []
    for cid in domain["campaign_ids"][2:4]:
        name = domain["campaign_names"][cid]
        answer = conn.execute(
            "SELECT COUNT(*) FROM campaign_attributions WHERE campaign_id = ?", [cid]
        ).fetchone()[0]
        out.append((f"How many customers are attributed to the '{name}' campaign?", answer, "int"))
    return out


def family_invoices_by_status(conn, domain) -> list[tuple[str, Any, str]]:
    out = []
    for status in domain["invoice_statuses"]:
        answer = conn.execute("SELECT COUNT(*) FROM invoices WHERE status = ?", [status]).fetchone()[0]
        out.append((f"How many invoices currently have status '{status}'?", answer, "int"))
    return out


def family_customers_signed_up_before(conn, domain) -> list[tuple[str, Any, str]]:
    cutoff = _mid_date(domain)
    answer = conn.execute("SELECT COUNT(*) FROM customers WHERE signup_date < ?", [cutoff]).fetchone()[0]
    return [(f"How many customers signed up before {cutoff}?", answer, "int")]


def family_tickets_open_by_priority(conn, domain) -> list[tuple[str, Any, str]]:
    out = []
    for priority in domain["ticket_priorities"][:2]:
        answer = conn.execute(
            "SELECT COUNT(*) FROM support_tickets WHERE priority = ? AND closed_date IS NULL", [priority]
        ).fetchone()[0]
        out.append((f"How many currently-open support tickets have priority '{priority}'?", answer, "int"))
    return out


def family_tickets_handled_by_employee(conn, domain, rng: random.Random) -> list[tuple[str, Any, str]]:
    out = []
    support_team_ids = [eid for eid in domain["employee_ids"] if True]
    ids = rng.sample(support_team_ids, k=2)
    for eid in ids:
        name = domain["employee_names"][eid]
        answer = conn.execute("SELECT COUNT(*) FROM support_tickets WHERE handled_by = ?", [eid]).fetchone()[0]
        out.append((f"How many support tickets has {name} handled?", answer, "int"))
    return out


# Ordered so the structurally distinct shapes (date-range, join) sit near
# the front — round-robin sampling in generate() still only guarantees
# every shape appears once every len(FAMILIES) items, so a caller requesting
# a very small --n could otherwise miss the join/date-range families
# entirely and end up with an easier-than-intended calibration set.
FAMILIES: list[Callable[..., list[tuple[str, Any, str]]]] = [
    family_tickets_by_priority,
    family_customers_signed_up_before,  # date-range shape
    family_campaign_attribution_count,  # join shape
    family_tickets_handled_by_employee,  # join shape
    family_active_subs_mrr_by_plan,
    family_tickets_by_category,
    family_invoice_total_by_customer,
    family_employees_by_team,
    family_customers_by_industry,
    family_customers_by_status_and_country,
    family_events_by_type,
    family_campaign_budget,
    family_invoices_by_status,
    family_tickets_open_by_priority,
]

_NEEDS_RNG = {
    family_invoice_total_by_customer,
    family_customers_by_industry,
    family_customers_by_status_and_country,
    family_tickets_handled_by_employee,
}


def generate(n: int, seed: int, db_path: Path) -> list[dict]:
    """`seed` only reorders session numbering; task content is fixed via
    CONTENT_SEED (see module docstring). `n` caps the total number of
    tasks returned (families run in a fixed order until the cap is hit)."""
    content_rng = random.Random(CONTENT_SEED)
    order_rng = random.Random(seed)
    conn = sqlite3.connect(db_path)
    try:
        domain = load_domain_values(conn)

        # Build every family's full item list first, then round-robin across
        # families (not exhaust one family before moving to the next) so a
        # small --n still samples every SQL *shape* (single-table filter,
        # join, aggregation, date-range) instead of only the families that
        # happen to come first — those shapes map to the actual failure
        # modes observed live (bad joins, date handling), so losing shape
        # diversity would quietly make the calibration set easier than it
        # should be.
        per_family: list[list[tuple[str, Any, str]]] = []
        for family in FAMILIES:
            if family in _NEEDS_RNG:
                per_family.append(family(conn, domain, content_rng))
            else:
                per_family.append(family(conn, domain))

        items: list[tuple[str, Any, str]] = []
        round_idx = 0
        while len(items) < n and any(round_idx < len(fam) for fam in per_family):
            for fam in per_family:
                if round_idx < len(fam):
                    items.append(fam[round_idx])
                if len(items) >= n:
                    break
            round_idx += 1

        if len(items) < n:
            raise RuntimeError(
                f"calibration generator only produced {len(items)} tasks, requested {n} — add more template families"
            )
        items = items[:n]

        order = list(range(len(items)))
        order_rng.shuffle(order)

        tasks = []
        for session, idx in enumerate(order, start=1):
            prompt, answer, answer_type = items[idx]
            tasks.append(
                {
                    "session": session,
                    "task_id": f"calib_{idx:03d}",
                    "task_type": "independent",
                    "prompt": prompt,
                    "answer": answer,
                    "answer_type": answer_type,
                }
            )
        return tasks
    finally:
        conn.close()


def write_calibration_yaml(tasks: list[dict], out_path: Path, seed: int) -> None:
    document = {"seed": seed, "n_tasks": len(tasks), "tasks": tasks}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(document, f, sort_keys=False, allow_unicode=True, default_flow_style=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of calibration tasks to generate")
    parser.add_argument("--seed", type=int, default=0, help="Session-order seed (task content is fixed)")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    if not args.db_path.exists():
        raise SystemExit(f"No environment found at {args.db_path} — run environment/build_env.py first.")

    tasks = generate(args.n, args.seed, args.db_path)
    write_calibration_yaml(tasks, args.out, args.seed)
    print(f"Generated {len(tasks)} calibration tasks for seed={args.seed} -> {args.out}")


if __name__ == "__main__":
    main()
