"""Deterministic 30-session task sequence generator for agent-memory-lab.

Builds a dependency graph of "facts" (business decisions narrated inside
task prompts) and schedules 30 tasks — one per session — via a randomized
topological sort. The random ordering is seeded independently from the
environment build seed, so the *same* environment (company.db) can be
scheduled into many different valid session orderings (order-effect
control across experiment seeds).

Task taxonomy (fixed counts):

- 10 independent tasks: self-contained, answerable from the current
  session alone. Three are pure noise (no downstream relevance). The
  other seven *establish* or *revise* a fact — narrating a decision in
  the prompt and asking a question answerable with that same decision,
  right now. This is where facts enter the timeline.
- 12 dependent tasks: reference a fact by label only (never restate its
  value) and require recalling it from an earlier session. Backed by 3
  "stable" facts (established once, never revised), ~4 tasks each.
- 8 trap tasks: same shape as dependent tasks, but backed by 2
  "revisable" facts that were established once and then REVISED in a
  later session. The correct answer requires the *current* (post-revision)
  value; using the pre-revision value produces a distinct, wrong answer
  (`stale_answer`) — this is the interference probe. Every trap task is
  validated at generation time to guarantee `stale_answer != answer`.

Every dependent/trap task carries `expected_source_session` (and, for
traps, `stale_source_session`) so a later citation-trace judge can tell
"the agent was wrong" apart from "the agent used the outdated source" —
the interference-attribution design this whole benchmark hinges on.

All correct answers (and stale distractors) are computed by running real
SQL against the environment's `company.db` — nothing is hand-typed, so
task generation always agrees with whatever the environment actually
contains.

Usage:
    uv run tasks/generate.py --seed 0
    uv run tasks/generate.py --seed 0 --db-path environment/company.db --out tasks/tasks.yaml
"""

from __future__ import annotations

import argparse
import random
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from environment.build_env import REFERENCE_DATE

TASKS_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = TASKS_DIR.parent / "environment" / "company.db"
DEFAULT_OUT_PATH = TASKS_DIR / "tasks.yaml"

N_SESSIONS = 30
# Task *content* (facts, question parameters) is generated from this fixed
# seed regardless of --seed, so that --seed varies only the session order.
CONTENT_SEED = 0
# Minimum |correct - stale| gap a trap task should have to count as a meaningful
# interference probe rather than noise (a 0-vs-1 difference is barely distinguishable
# from randomness). Trap generation prefers the largest available gap; anything below
# this prints a warning for manual review rather than failing generation outright.
MIN_TRAP_GAP = 2


# --- Fact catalog ------------------------------------------------------
#
# 3 stable facts feed the 12 dependent tasks (4 each).
# 2 revisable facts feed the 8 trap tasks (4 each).
# Each fact carries the SQL-backed answer function(s) used both to plant
# the decision (establish/revise tasks) and to probe recall of it later
# (dependent/trap tasks).


@dataclass
class Fact:
    fact_id: str
    kind: str  # "stable" | "revisable"
    label: str  # human-readable name used in dependent/trap prompts, never the value itself
    established_value: Any
    revised_value: Any | None = None
    n_consumers: int = 4


FACTS = [
    Fact(
        fact_id="high_value_mrr",
        kind="stable",
        label="the loyalty-program 'high-value customer' definition",
        established_value=2000,
    ),
    Fact(
        fact_id="priority_industries",
        kind="stable",
        label="the Q3 expansion 'priority industries' list",
        established_value=["Fintech", "Healthtech", "Cybersecurity"],
    ),
    Fact(
        fact_id="campaign_success_mrr",
        kind="stable",
        label="marketing's 'successful campaign' MRR rule",
        established_value=5000,
    ),
    Fact(
        fact_id="engaged_window",
        kind="revisable",
        label="the ad hoc 'engaged customer' lookback window for the retention project",
        established_value=45,
        revised_value=30,
    ),
    Fact(
        fact_id="priority_ticket_categories",
        kind="revisable",
        label="the support team's 'at-risk ticket' category set",
        established_value=["billing", "bug_report"],
        revised_value=["bug_report", "data_export"],
    ),
]
FACTS_BY_ID = {f.fact_id: f for f in FACTS}


# --- SQL-backed answer functions ---------------------------------------


def high_value_count(conn: sqlite3.Connection, threshold: int, variant: str, param: Any) -> int:
    clauses = ["c.status = 'active'", "s.status = 'active'", "s.mrr > ?"]
    params: list[Any] = [threshold]
    if variant == "industry":
        clauses.append("c.industry = ?")
        params.append(param)
    elif variant == "country":
        clauses.append("c.country = ?")
        params.append(param)
    elif variant == "plan_tier":
        clauses.append("c.plan_tier = ?")
        params.append(param)
    elif variant == "signup_before":
        clauses.append("c.signup_date < ?")
        params.append(param)
    else:
        raise ValueError(variant)
    sql = f"""
        SELECT COUNT(*) FROM customers c
        JOIN subscriptions s ON s.customer_id = c.customer_id
        WHERE {' AND '.join(clauses)}
    """
    return conn.execute(sql, params).fetchone()[0]


def priority_industry_metric(conn: sqlite3.Connection, industries: list[str], variant: str, param: Any) -> int:
    placeholders = ",".join("?" for _ in industries)
    if variant == "signup_after":
        sql = f"SELECT COUNT(*) FROM customers WHERE industry IN ({placeholders}) AND signup_date > ?"
        params = list(industries) + [param]
    elif variant == "invoice_sum":
        date_from, date_to = param
        sql = f"""
            SELECT COALESCE(SUM(i.amount), 0) FROM invoices i
            JOIN customers c ON c.customer_id = i.customer_id
            WHERE c.industry IN ({placeholders}) AND i.invoice_date BETWEEN ? AND ?
        """
        params = list(industries) + [date_from, date_to]
    elif variant == "plan_tier":
        sql = f"SELECT COUNT(*) FROM customers WHERE industry IN ({placeholders}) AND plan_tier = ?"
        params = list(industries) + [param]
    elif variant == "tickets":
        sql = f"""
            SELECT COUNT(*) FROM support_tickets t
            JOIN customers c ON c.customer_id = t.customer_id
            WHERE c.industry IN ({placeholders})
        """
        params = list(industries)
    else:
        raise ValueError(variant)
    return conn.execute(sql, params).fetchone()[0]


def campaign_combined_mrr(conn: sqlite3.Connection, campaign_id: int) -> int:
    sql = """
        SELECT COALESCE(SUM(s.mrr), 0) FROM campaign_attributions ca
        JOIN subscriptions s ON s.customer_id = ca.customer_id AND s.status = 'active'
        WHERE ca.campaign_id = ?
    """
    return conn.execute(sql, [campaign_id]).fetchone()[0]


def campaign_success(conn: sqlite3.Connection, threshold: int, campaign_id: int) -> bool:
    return campaign_combined_mrr(conn, campaign_id) > threshold


def campaign_success_count(conn: sqlite3.Connection, threshold: int, campaign_ids: list[int]) -> int:
    return sum(1 for cid in campaign_ids if campaign_success(conn, threshold, cid))


def engaged_count(conn: sqlite3.Connection, window_days: int, variant: str, param: Any) -> int:
    clauses = [
        "e.event_type IN ('login','feature_use:dashboard','feature_use:export','feature_use:api')",
        "e.event_date >= date(?, ?)",
    ]
    params: list[Any] = [REFERENCE_DATE.isoformat(), f"-{window_days} day"]
    join = ""
    if variant != "all":
        join = "JOIN customers c ON c.customer_id = e.customer_id"
        column = {"industry": "c.industry", "country": "c.country", "plan_tier": "c.plan_tier"}[variant]
        clauses.append(f"{column} = ?")
        params.append(param)
    sql = f"SELECT COUNT(DISTINCT e.customer_id) FROM events e {join} WHERE {' AND '.join(clauses)}"
    return conn.execute(sql, params).fetchone()[0]


def priority_ticket_count(conn: sqlite3.Connection, categories: list[str], date_from: str, date_to: str) -> int:
    placeholders = ",".join("?" for _ in categories)
    sql = f"""
        SELECT COUNT(*) FROM support_tickets
        WHERE category IN ({placeholders}) AND opened_date BETWEEN ? AND ?
    """
    params = list(categories) + [date_from, date_to]
    return conn.execute(sql, params).fetchone()[0]


# --- Domain value pools (read from the DB, not hardcoded) ---------------


def load_domain_values(conn: sqlite3.Connection) -> dict[str, list]:
    return {
        "industries": [r[0] for r in conn.execute("SELECT DISTINCT industry FROM customers ORDER BY 1")],
        "countries": [r[0] for r in conn.execute("SELECT DISTINCT country FROM customers ORDER BY 1")],
        "plan_tiers": [r[0] for r in conn.execute("SELECT DISTINCT plan_tier FROM customers ORDER BY 1")],
        "campaign_ids": [r[0] for r in conn.execute("SELECT campaign_id FROM campaigns ORDER BY 1")],
        "campaign_names": dict(conn.execute("SELECT campaign_id, name FROM campaigns")),
        "signup_min_max": conn.execute("SELECT MIN(signup_date), MAX(signup_date) FROM customers").fetchone(),
    }


# --- Task node model ------------------------------------------------------


@dataclass
class TaskNode:
    node_id: str
    role: str  # "noise" | "establish" | "revise" | "consume"
    task_type: str  # "independent" | "dependent" | "trap"
    fact_id: str | None
    build: Callable[[], dict]
    depends_on: list[str] = field(default_factory=list)
    task: dict | None = None  # filled in after scheduling assigns a session


def build_noise_tasks(rng: random.Random, conn: sqlite3.Connection, domain: dict) -> list[TaskNode]:
    # All randomness resolved here, at construction time — not inside the
    # build() closures, which run later in schedule() order. If a closure
    # drew from `rng` lazily, two different --seed runs would consume this
    # generator's random calls in a different relative order and silently
    # produce different task content (order and content must stay decoupled;
    # see CONTENT_SEED / test_different_seed_reorders_but_keeps_same_task_content).
    noise_customer_id, noise_customer_name = conn.execute(
        "SELECT customer_id, name FROM customers ORDER BY customer_id LIMIT 1 OFFSET ?",
        [rng.randrange(0, N_customers_count(conn))],
    ).fetchone()
    noise_team = rng.choice(sorted({r[0] for r in conn.execute("SELECT DISTINCT team FROM employees")}))

    def noise_1():
        return {
            "prompt": "How many support tickets currently have priority 'urgent'?",
            "answer": conn.execute("SELECT COUNT(*) FROM support_tickets WHERE priority = 'urgent'").fetchone()[0],
            "answer_type": "int",
        }

    def noise_2():
        total = conn.execute(
            "SELECT COALESCE(SUM(amount),0) FROM invoices WHERE customer_id = ?", [noise_customer_id]
        ).fetchone()[0]
        return {
            "prompt": f"What is the total invoice amount on record for customer '{noise_customer_name}'?",
            "answer": total,
            "answer_type": "int",
        }

    def noise_3():
        count = conn.execute("SELECT COUNT(*) FROM employees WHERE team = ?", [noise_team]).fetchone()[0]
        return {
            "prompt": f"How many employees are on the {noise_team} team?",
            "answer": count,
            "answer_type": "int",
        }

    builders = [noise_1, noise_2, noise_3]
    return [
        TaskNode(node_id=f"noise_{i+1}", role="noise", task_type="independent", fact_id=None, build=b)
        for i, b in enumerate(builders)
    ]


def N_customers_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]


# --- Fact-specific establish/revise/consume builders ---------------------


def _best_candidate(candidates: list, score: Callable[[Any], int]) -> Any:
    """Pick the candidate with the highest score (ties broken by input order,
    which callers should have pre-shuffled via `rng` for determinism)."""
    best, best_score = candidates[0], -1
    for candidate in candidates:
        s = score(candidate)
        if s > best_score:
            best, best_score = candidate, s
    return best


def build_high_value_nodes(rng: random.Random, conn: sqlite3.Connection, domain: dict) -> list[TaskNode]:
    fact = FACTS_BY_ID["high_value_mrr"]
    threshold = fact.established_value
    signup_candidates = [_random_signup_cutoff(rng, domain) for _ in range(6)]
    param_pools = {
        "industry": rng.sample(domain["industries"], k=len(domain["industries"])),
        "country": rng.sample(domain["countries"], k=len(domain["countries"])),
        "plan_tier": rng.sample(domain["plan_tiers"], k=len(domain["plan_tiers"])),
        "signup_before": signup_candidates,
    }
    # Prefer non-degenerate (higher-count) params over a blind pick — an
    # answer of 0 is legitimate but easy to hit by a no-memory default guess.
    params = {
        "industry": [_best_candidate(param_pools["industry"], lambda v: high_value_count(conn, threshold, "industry", v))],
        "country": [_best_candidate(param_pools["country"], lambda v: high_value_count(conn, threshold, "country", v))],
        "plan_tier": [_best_candidate(param_pools["plan_tier"], lambda v: high_value_count(conn, threshold, "plan_tier", v))],
        "signup_before": [_best_candidate(param_pools["signup_before"], lambda v: high_value_count(conn, threshold, "signup_before", v))],
    }

    def establish():
        industry = _best_candidate(domain["industries"], lambda v: high_value_count(conn, threshold, "industry", v))
        answer = high_value_count(conn, threshold, "industry", industry)
        return {
            "prompt": (
                "In today's revenue-ops sync, the team decided: for the new loyalty program, "
                f"a customer counts as 'high-value' if their status is active and their active "
                f"subscription MRR exceeds ${threshold}. Using this new definition, how many "
                f"high-value customers are there in the {industry} industry?"
            ),
            "answer": answer,
            "answer_type": "int",
            "decision_planted": fact.fact_id,
        }

    nodes = [TaskNode(node_id="hv_establish", role="establish", task_type="independent", fact_id=fact.fact_id, build=establish)]

    consume_specs = [
        ("industry", params["industry"][0], "in the {v} industry"),
        ("country", params["country"][0], "based in {v}"),
        ("plan_tier", params["plan_tier"][0], "on the {v} plan"),
        ("signup_before", params["signup_before"][0], "who signed up before {v}"),
    ]
    for i, (variant, param, phrase) in enumerate(consume_specs):
        def build(variant=variant, param=param, phrase=phrase):
            answer = high_value_count(conn, threshold, variant, param)
            return {
                "prompt": (
                    f"Using {fact.label} decided earlier, how many high-value customers are there "
                    + phrase.format(v=param) + "?"
                ),
                "answer": answer,
                "answer_type": "int",
            }

        nodes.append(
            TaskNode(node_id=f"hv_consume_{i+1}", role="consume", task_type="dependent", fact_id=fact.fact_id,
                      build=build, depends_on=["hv_establish"])
        )
    return nodes


def _random_signup_cutoff(rng: random.Random, domain: dict) -> str:
    from datetime import date, timedelta

    lo = date.fromisoformat(domain["signup_min_max"][0])
    hi = date.fromisoformat(domain["signup_min_max"][1])
    span = (hi - lo).days
    return (lo + timedelta(days=rng.randint(span // 4, 3 * span // 4))).isoformat()


def build_priority_industry_nodes(rng: random.Random, conn: sqlite3.Connection, domain: dict) -> list[TaskNode]:
    fact = FACTS_BY_ID["priority_industries"]
    industries = fact.established_value

    def establish():
        answer = priority_industry_metric(conn, industries, "signup_after", domain["signup_min_max"][0])
        return {
            "prompt": (
                "In the Q3 expansion planning session, leadership decided the priority industries "
                f"for expansion are: {', '.join(industries)}. Using this list, how many customers in "
                f"priority industries signed up after {domain['signup_min_max'][0]}?"
            ),
            "answer": answer,
            "answer_type": "int",
            "decision_planted": fact.fact_id,
        }

    nodes = [TaskNode(node_id="pi_establish", role="establish", task_type="independent", fact_id=fact.fact_id, build=establish)]

    cutoff = _best_candidate(
        [_random_signup_cutoff(rng, domain) for _ in range(6)],
        lambda v: priority_industry_metric(conn, industries, "signup_after", v),
    )
    date_range = _best_candidate(
        [_random_date_range(rng, domain) for _ in range(10)],
        lambda v: priority_industry_metric(conn, industries, "invoice_sum", v),
    )
    plan_tier = _best_candidate(
        rng.sample(domain["plan_tiers"], k=len(domain["plan_tiers"])),
        lambda v: priority_industry_metric(conn, industries, "plan_tier", v),
    )
    consume_specs = [
        ("signup_after", cutoff, f"How many priority-industry customers signed up after {cutoff}?"),
        ("invoice_sum", date_range, f"What is the total invoice amount from priority-industry customers between {date_range[0]} and {date_range[1]}?"),
        ("plan_tier", plan_tier, f"How many priority-industry customers are on the {plan_tier} plan?"),
        ("tickets", None, "How many support tickets have been opened by priority-industry customers?"),
    ]
    for i, (variant, param, question) in enumerate(consume_specs):
        def build(variant=variant, param=param, question=question):
            answer = priority_industry_metric(conn, industries, variant, param)
            return {
                "prompt": f"Using {fact.label} decided earlier, {question[0].lower()}{question[1:]}",
                "answer": answer,
                "answer_type": "int",
            }

        nodes.append(
            TaskNode(node_id=f"pi_consume_{i+1}", role="consume", task_type="dependent", fact_id=fact.fact_id,
                      build=build, depends_on=["pi_establish"])
        )
    return nodes


def _random_date_range(rng: random.Random, domain: dict, span_days: tuple[int, int] = (60, 150)) -> tuple[str, str]:
    from datetime import date, timedelta

    lo = date.fromisoformat(domain["signup_min_max"][0])
    start = lo + timedelta(days=rng.randint(30, 200))
    end = start + timedelta(days=rng.randint(*span_days))
    if end > REFERENCE_DATE:
        end = REFERENCE_DATE
    return start.isoformat(), end.isoformat()


def build_campaign_success_nodes(rng: random.Random, conn: sqlite3.Connection, domain: dict) -> list[TaskNode]:
    fact = FACTS_BY_ID["campaign_success_mrr"]
    threshold = fact.established_value
    campaign_ids = list(domain["campaign_ids"])
    rng.shuffle(campaign_ids)
    establish_campaign, bool_campaign, bool_campaign_2, mrr_campaign = campaign_ids[0], campaign_ids[1], campaign_ids[2], campaign_ids[3]

    def establish():
        name = domain["campaign_names"][establish_campaign]
        is_success = campaign_success(conn, threshold, establish_campaign)
        return {
            "prompt": (
                "Marketing decided a campaign counts as 'successful' if the combined active-subscription "
                f"MRR of customers attributed to it exceeds ${threshold}. Using this rule, is the "
                f"'{name}' campaign successful? Answer 'yes' or 'no'."
            ),
            "answer": "yes" if is_success else "no",
            "answer_type": "string",
            "decision_planted": fact.fact_id,
        }

    nodes = [TaskNode(node_id="cs_establish", role="establish", task_type="independent", fact_id=fact.fact_id, build=establish)]

    def build_bool():
        name = domain["campaign_names"][bool_campaign]
        is_success = campaign_success(conn, threshold, bool_campaign)
        return {
            "prompt": (
                f"Using {fact.label} decided earlier, is the '{name}' campaign successful? Answer 'yes' or 'no'."
            ),
            "answer": "yes" if is_success else "no",
            "answer_type": "string",
        }

    def build_bool_2():
        name = domain["campaign_names"][bool_campaign_2]
        is_success = campaign_success(conn, threshold, bool_campaign_2)
        return {
            "prompt": (
                f"Using {fact.label} decided earlier, is the '{name}' campaign successful? Answer 'yes' or 'no'."
            ),
            "answer": "yes" if is_success else "no",
            "answer_type": "string",
        }

    def build_mrr():
        name = domain["campaign_names"][mrr_campaign]
        return {
            "prompt": (
                f"Using {fact.label} decided earlier (the MRR threshold that defines success), "
                f"what is the combined attributed active-subscription MRR for the '{name}' campaign, in dollars?"
            ),
            "answer": campaign_combined_mrr(conn, mrr_campaign),
            "answer_type": "int",
        }

    def build_count():
        return {
            "prompt": f"Using {fact.label} decided earlier, how many of the company's campaigns are currently successful?",
            "answer": campaign_success_count(conn, threshold, domain["campaign_ids"]),
            "answer_type": "int",
        }

    for i, builder in enumerate([build_bool, build_bool_2, build_mrr, build_count]):
        nodes.append(
            TaskNode(node_id=f"cs_consume_{i+1}", role="consume", task_type="dependent", fact_id=fact.fact_id,
                      build=builder, depends_on=["cs_establish"])
        )
    return nodes


def build_engaged_window_nodes(rng: random.Random, conn: sqlite3.Connection, domain: dict) -> list[TaskNode]:
    fact = FACTS_BY_ID["engaged_window"]
    old_window, new_window = fact.established_value, fact.revised_value

    def establish():
        answer = engaged_count(conn, old_window, "all", None)
        return {
            "prompt": (
                "For the customer retention project, the team decided (this is a one-off project "
                f"convention, separate from the official Active Customer policy) that an 'engaged' "
                f"customer is one with a qualifying usage event in the trailing {old_window} days. "
                f"Using this definition, how many engaged customers are there right now?"
            ),
            "answer": answer,
            "answer_type": "int",
            "decision_planted": fact.fact_id,
        }

    def revise():
        answer = engaged_count(conn, new_window, "all", None)
        return {
            "prompt": (
                "The retention project team re-reviewed the numbers and decided the engaged-customer "
                f"lookback window was too loose — they're changing it from {old_window} days to "
                f"{new_window} days effective immediately. Using this updated window, how many "
                f"engaged customers are there right now?"
            ),
            "answer": answer,
            "answer_type": "int",
            "decision_planted": fact.fact_id,
        }

    nodes = [
        TaskNode(node_id="ew_establish", role="establish", task_type="independent", fact_id=fact.fact_id, build=establish),
        TaskNode(node_id="ew_revise", role="revise", task_type="independent", fact_id=fact.fact_id, build=revise,
                  depends_on=["ew_establish"]),
    ]

    variants = ["industry", "country", "plan_tier", "all"]
    candidates = {
        "industry": rng.sample(domain["industries"], k=len(domain["industries"])),
        "country": rng.sample(domain["countries"], k=len(domain["countries"])),
        "plan_tier": rng.sample(domain["plan_tiers"], k=len(domain["plan_tiers"])),
        "all": [None],
    }
    phrase = {
        "industry": lambda v: f"in the {v} industry",
        "country": lambda v: f"based in {v}",
        "plan_tier": lambda v: f"on the {v} plan",
        "all": lambda v: "company-wide",
    }
    for i, variant in enumerate(variants):
        # Among all candidate params for this variant, pick the one with the
        # largest pre-/post-revision gap — not just the first that differs.
        # A 0-vs-1 trap is barely distinguishable from noise; a bigger gap is
        # a more meaningful interference signal.
        param, best_gap = candidates[variant][0], -1
        for candidate in candidates[variant]:
            gap = abs(engaged_count(conn, new_window, variant, candidate) - engaged_count(conn, old_window, variant, candidate))
            if gap > best_gap:
                param, best_gap = candidate, gap
        if best_gap < MIN_TRAP_GAP:
            print(f"  [warn] ew_trap variant={variant} best gap is only {best_gap} (< {MIN_TRAP_GAP}) — weak trap, review manually")

        def build(variant=variant, param=param):
            correct = engaged_count(conn, new_window, variant, param)
            stale = engaged_count(conn, old_window, variant, param)
            return {
                "prompt": (
                    f"Using {fact.label}, how many engaged customers are there {phrase[variant](param)}?"
                ),
                "answer": correct,
                "answer_type": "int",
                "stale_answer": stale,
            }

        nodes.append(
            TaskNode(node_id=f"ew_trap_{i+1}", role="consume", task_type="trap", fact_id=fact.fact_id,
                      build=build, depends_on=["ew_revise"])
        )
    return nodes


def build_priority_ticket_nodes(rng: random.Random, conn: sqlite3.Connection, domain: dict) -> list[TaskNode]:
    fact = FACTS_BY_ID["priority_ticket_categories"]
    old_categories, new_categories = fact.established_value, fact.revised_value

    def establish():
        date_from, date_to = domain["signup_min_max"][0], REFERENCE_DATE.isoformat()
        answer = priority_ticket_count(conn, old_categories, date_from, date_to)
        return {
            "prompt": (
                "The support team decided that 'at-risk' tickets — the ones that get escalated to the "
                f"account owner — are categories: {', '.join(old_categories)}. Using this rule, how many "
                f"at-risk tickets have been opened in total?"
            ),
            "answer": answer,
            "answer_type": "int",
            "decision_planted": fact.fact_id,
        }

    def revise():
        date_from, date_to = domain["signup_min_max"][0], REFERENCE_DATE.isoformat()
        answer = priority_ticket_count(conn, new_categories, date_from, date_to)
        return {
            "prompt": (
                f"Support leadership revised the at-risk category set — data showed billing tickets rarely "
                f"led to churn, while data_export tickets did. The at-risk categories are now: "
                f"{', '.join(new_categories)}. Using this updated rule, how many at-risk tickets have been "
                f"opened in total?"
            ),
            "answer": answer,
            "answer_type": "int",
            "decision_planted": fact.fact_id,
        }

    nodes = [
        TaskNode(node_id="pt_establish", role="establish", task_type="independent", fact_id=fact.fact_id, build=establish),
        TaskNode(node_id="pt_revise", role="revise", task_type="independent", fact_id=fact.fact_id, build=revise,
                  depends_on=["pt_establish"]),
    ]

    # Sample a pool of candidate date ranges, score each by |correct - stale|,
    # and keep the 4 with the largest gap — a bigger gap is a more meaningful
    # interference signal than a bare 0-vs-1 difference (see MIN_TRAP_GAP).
    pool: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    attempts = 0
    while len(pool) < 40 and attempts < 400:
        attempts += 1
        # Ticket categories are sparse (~20 tickets/category over the whole
        # timeline), so a short window rarely collects more than 0-1 per
        # category — widen the span to get a meaningful billing-vs-data_export gap.
        candidate = _random_date_range(rng, domain, span_days=(300, 700))
        if candidate in seen:
            continue
        seen.add(candidate)
        pool.append(candidate)

    scored = []
    for date_from, date_to in pool:
        correct = priority_ticket_count(conn, new_categories, date_from, date_to)
        stale = priority_ticket_count(conn, old_categories, date_from, date_to)
        if correct != stale:
            scored.append((abs(correct - stale), (date_from, date_to)))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    if len(scored) < 4:
        raise RuntimeError(
            "priority_ticket_categories: couldn't find 4 non-degenerate date ranges — "
            "the old/new category sets may be too similar in this environment."
        )
    date_ranges = [dr for _, dr in scored[:4]]
    if scored[3][0] < MIN_TRAP_GAP:
        print(f"  [warn] pt_trap weakest selected gap is {scored[3][0]} (< {MIN_TRAP_GAP}) — review manually")

    for i, (date_from, date_to) in enumerate(date_ranges):
        def build(date_from=date_from, date_to=date_to):
            correct = priority_ticket_count(conn, new_categories, date_from, date_to)
            stale = priority_ticket_count(conn, old_categories, date_from, date_to)
            return {
                "prompt": (
                    f"Using {fact.label}, how many at-risk tickets were opened between {date_from} and {date_to}?"
                ),
                "answer": correct,
                "answer_type": "int",
                "stale_answer": stale,
            }

        nodes.append(
            TaskNode(node_id=f"pt_trap_{i+1}", role="consume", task_type="trap", fact_id=fact.fact_id,
                      build=build, depends_on=["pt_revise"])
        )
    return nodes


# --- Scheduling: randomized topological sort -----------------------------


def schedule(nodes: list[TaskNode], rng: random.Random) -> list[TaskNode]:
    by_id = {n.node_id: n for n in nodes}
    indegree = {n.node_id: len(n.depends_on) for n in nodes}
    dependents: dict[str, list[str]] = {n.node_id: [] for n in nodes}
    for n in nodes:
        for dep in n.depends_on:
            dependents[dep].append(n.node_id)

    ready = [n.node_id for n in nodes if indegree[n.node_id] == 0]
    ordered: list[TaskNode] = []
    while ready:
        rng.shuffle(ready)
        chosen = ready.pop()
        ordered.append(by_id[chosen])
        for dep_id in dependents[chosen]:
            indegree[dep_id] -= 1
            if indegree[dep_id] == 0:
                ready.append(dep_id)

    if len(ordered) != len(nodes):
        raise RuntimeError("Cycle detected in fact dependency graph — this is a generator bug, not a data issue.")
    return ordered


# --- Validation ------------------------------------------------------------


def validate_tasks(tasks: list[dict]) -> None:
    by_session = {t["session"]: t for t in tasks}
    assert sorted(by_session) == list(range(1, N_SESSIONS + 1)), "session indices must be exactly 1..30"

    counts = {"independent": 0, "dependent": 0, "trap": 0}
    for t in tasks:
        counts[t["task_type"]] += 1
    assert counts == {"independent": 10, "dependent": 12, "trap": 8}, f"task-type counts off: {counts}"

    for t in tasks:
        if t["task_type"] in ("dependent", "trap"):
            source = t["expected_source_session"]
            assert source < t["session"], (
                f"session {t['session']} ({t['task_type']}) cites source session {source}, "
                "which is not in its past"
            )
        if t["task_type"] == "trap":
            assert t["answer"] != t["stale_answer"], (
                f"session {t['session']}: trap task's correct answer equals the stale distractor "
                f"({t['answer']!r}) — this trap doesn't actually trap anything, fix the fact/variant"
            )
            assert t["stale_source_session"] < t["expected_source_session"] < t["session"]


# --- Orchestration ---------------------------------------------------------


def generate(seed: int, db_path: Path) -> list[dict]:
    """`seed` controls session ORDER only. Task content (which facts exist,
    which parameters/questions probe them) is generated from a fixed
    internal seed, so the same 30 tasks get reshuffled into a new valid
    topological order per seed — isolating order effects from content
    differences.
    """
    content_rng = random.Random(CONTENT_SEED)
    order_rng = random.Random(seed)
    conn = sqlite3.connect(db_path)
    try:
        domain = load_domain_values(conn)

        nodes: list[TaskNode] = []
        nodes += build_noise_tasks(content_rng, conn, domain)
        nodes += build_high_value_nodes(content_rng, conn, domain)
        nodes += build_priority_industry_nodes(content_rng, conn, domain)
        nodes += build_campaign_success_nodes(content_rng, conn, domain)
        nodes += build_engaged_window_nodes(content_rng, conn, domain)
        nodes += build_priority_ticket_nodes(content_rng, conn, domain)
        assert len(nodes) == N_SESSIONS, f"expected {N_SESSIONS} task nodes, built {len(nodes)}"

        ordered = schedule(nodes, order_rng)

        session_of = {n.node_id: i + 1 for i, n in enumerate(ordered)}
        tasks = []
        for node in ordered:
            payload = node.build()
            task = {
                "session": session_of[node.node_id],
                "task_id": node.node_id,
                "task_type": node.task_type,
                "role": node.role,
                "fact_id": node.fact_id,
                **payload,
            }
            if node.task_type in ("dependent", "trap"):
                # For dependent tasks the source is the fact's establish node;
                # for trap tasks it's the *revise* node (the up-to-date source).
                source_node_id = node.depends_on[0]
                task["expected_source_session"] = session_of[source_node_id]
                if node.task_type == "trap":
                    # stale source = the establish node, i.e. the revise node's own dependency
                    establish_node_id = by_id_lookup(ordered, source_node_id).depends_on[0]
                    task["stale_source_session"] = session_of[establish_node_id]
            tasks.append(task)

        validate_tasks(tasks)
        return tasks
    finally:
        conn.close()


def by_id_lookup(nodes: list[TaskNode], node_id: str) -> TaskNode:
    for n in nodes:
        if n.node_id == node_id:
            return n
    raise KeyError(node_id)


def write_tasks_yaml(tasks: list[dict], out_path: Path, seed: int) -> None:
    tasks_sorted = sorted(tasks, key=lambda t: t["session"])
    document = {
        "seed": seed,
        "n_sessions": N_SESSIONS,
        "tasks": tasks_sorted,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(document, f, sort_keys=False, allow_unicode=True, default_flow_style=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="Session-order seed (task content is fixed; independent of the environment build seed)")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    if not args.db_path.exists():
        raise SystemExit(f"No environment found at {args.db_path} — run environment/build_env.py first.")

    tasks = generate(args.seed, args.db_path)
    write_tasks_yaml(tasks, args.out, args.seed)

    by_type: dict[str, int] = {}
    for t in tasks:
        by_type[t["task_type"]] = by_type.get(t["task_type"], 0) + 1
    print(f"Generated {len(tasks)} tasks for seed={args.seed} -> {args.out}")
    print(f"  Counts: {by_type}")


if __name__ == "__main__":
    main()
