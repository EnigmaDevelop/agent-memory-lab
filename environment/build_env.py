"""Deterministic generator for the agent-memory-lab fictional company environment.

Builds a SQLite database and a set of markdown reference documents for a
fictional B2B SaaS company ("Solace Metrics"). Everything is derived from a
single integer seed via ``random.Random`` — no wall-clock, no network, no
external data files — so the same seed always reproduces byte-identical
markdown docs and identical SQLite *content* (schema + rows; see
``content_hash`` for why we don't hash the raw .db file).

Two intentional traps live in this environment, matched to this benchmark's
"trap task" design:

- ``customers_legacy``: a deprecated CRM export table with different column
  names, a different id space, and old tier labels for a subset of the same
  companies as ``customers`` — a "similarly named different table" trap.
- ``docs/policies/active_customer_definition_v1_DEPRECATED.md``: an old
  "active = 60 days" policy that the current doc explicitly supersedes with
  90 days — a stale-definition trap for retrieval-based memory strategies.

Usage:
    uv run environment/build_env.py --seed 42
    uv run environment/build_env.py --seed 42 --out-dir /tmp/env-check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

ENV_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = ENV_DIR / "company.db"
DEFAULT_DOCS_DIR = ENV_DIR / "docs"
DEFAULT_MANIFEST_PATH = ENV_DIR / "manifest.json"

COMPANY_NAME = "Solace Metrics"
COMPANY_FOUNDED = date(2022, 3, 1)
# Fixed synthetic "as of" date for the whole environment. Never derived from
# wall-clock time, so the same seed always yields the same generated dates.
REFERENCE_DATE = date(2026, 1, 1)

SCHEMA_SQL = """
CREATE TABLE customers (
    customer_id     INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    industry        TEXT NOT NULL,
    country         TEXT NOT NULL,
    plan_tier       TEXT NOT NULL,
    signup_date     TEXT NOT NULL,
    status          TEXT NOT NULL,
    owner_employee_id INTEGER NOT NULL REFERENCES employees(employee_id)
);

-- Deprecated CRM export. Different column names, different id space, old
-- tier labels. Superseded by `customers` but kept for historical audit —
-- an intentional schema-drift trap (see module docstring).
CREATE TABLE customers_legacy (
    legacy_id       INTEGER PRIMARY KEY,
    company_name    TEXT NOT NULL,
    joined_on       TEXT NOT NULL,
    tier            TEXT NOT NULL,
    notes           TEXT
);

CREATE TABLE employees (
    employee_id     INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    team            TEXT NOT NULL,
    role            TEXT NOT NULL,
    hire_date       TEXT NOT NULL
);

CREATE TABLE subscriptions (
    subscription_id INTEGER PRIMARY KEY,
    customer_id     INTEGER NOT NULL REFERENCES customers(customer_id),
    plan_tier       TEXT NOT NULL,
    mrr             INTEGER NOT NULL,
    start_date      TEXT NOT NULL,
    end_date        TEXT,
    status          TEXT NOT NULL
);

CREATE TABLE events (
    event_id        INTEGER PRIMARY KEY,
    customer_id     INTEGER NOT NULL REFERENCES customers(customer_id),
    event_type      TEXT NOT NULL,
    event_date      TEXT NOT NULL
);

CREATE TABLE invoices (
    invoice_id      INTEGER PRIMARY KEY,
    customer_id     INTEGER NOT NULL REFERENCES customers(customer_id),
    amount          INTEGER NOT NULL,
    invoice_date    TEXT NOT NULL,
    status          TEXT NOT NULL
);

CREATE TABLE support_tickets (
    ticket_id       INTEGER PRIMARY KEY,
    customer_id     INTEGER NOT NULL REFERENCES customers(customer_id),
    opened_date     TEXT NOT NULL,
    closed_date     TEXT,
    category        TEXT NOT NULL,
    priority        TEXT NOT NULL,
    handled_by      INTEGER NOT NULL REFERENCES employees(employee_id)
);

CREATE TABLE campaigns (
    campaign_id     INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    channel         TEXT NOT NULL,
    start_date      TEXT NOT NULL,
    end_date        TEXT NOT NULL,
    budget          INTEGER NOT NULL
);

CREATE TABLE campaign_attributions (
    campaign_id     INTEGER NOT NULL REFERENCES campaigns(campaign_id),
    customer_id     INTEGER NOT NULL REFERENCES customers(customer_id),
    attributed_date TEXT NOT NULL,
    PRIMARY KEY (campaign_id, customer_id)
);
"""

INDUSTRIES = [
    "Fintech", "Healthtech", "Logistics", "E-commerce", "EdTech",
    "Manufacturing", "Media", "Real Estate", "Insurance", "Gaming",
    "Cybersecurity", "AgTech",
]
COUNTRIES = ["US", "UK", "DE", "FR", "CA", "NL", "SE", "AU", "SG", "BR"]
PLAN_TIERS = ["Starter", "Growth", "Enterprise"]
LEGACY_TIER_LABELS = {"Starter": "Basic", "Growth": "Pro", "Enterprise": "Enterprise"}
CUSTOMER_STATUSES_WEIGHTED = [
    ("active", 30), ("active", 30), ("active", 30), ("churned", 8), ("trial", 8),
]

COMPANY_NAME_PARTS_1 = [
    "Nimbus", "Vertex", "Cobalt", "Lattice", "Beacon", "Marrow", "Solene",
    "Copper", "Anvil", "Quartz", "Halcyon", "Ferro", "Umber", "Solvent",
    "Granite", "Ember", "Fathom", "Ironclad", "Meridian", "Pallet",
    "Rivet", "Slate", "Tundra", "Vellum", "Wharf", "Yarrow", "Zephyr",
    "Basalt", "Cinder", "Driftwood", "Elmwood", "Foxglove", "Gantry",
    "Harbor", "Ivywood", "Juniper", "Kestrel", "Lanternfish", "Millbrook",
    "Nettle",
]
COMPANY_NAME_PARTS_2 = [
    "Systems", "Works", "Labs", "Dynamics", "Group", "Partners", "Networks",
    "Analytics", "Freight", "Robotics", "Solutions", "Collective",
]

FIRST_NAMES = [
    "Ava", "Noah", "Mia", "Leo", "Zara", "Kai", "Priya", "Omar", "Elin",
    "Diego", "Yuki", "Sana", "Theo", "Nadia", "Felix", "Ines", "Jonas",
    "Amara", "Lucas", "Freya", "Ravi", "Clara", "Tomas", "Nia",
]
LAST_NAMES = [
    "Berg", "Fontaine", "Osei", "Nakamura", "Petrov", "Alvarez", "Lindqvist",
    "Haddad", "Rossi", "Novak", "Chen", "Okafor", "Moreau", "Kowalski",
    "Silva", "Andersen", "Farhan", "Costa", "Voss", "Ibarra",
]
TEAMS_ROLES = [
    ("Sales", "Account Executive"),
    ("Sales", "Sales Development Rep"),
    ("Customer Success", "Customer Success Manager"),
    ("Support", "Support Engineer"),
    ("Data", "Data Analyst"),
    ("Data", "Analytics Engineer"),
    ("Engineering", "Backend Engineer"),
    ("Marketing", "Growth Marketer"),
]
EVENT_TYPES_WEIGHTED = [
    ("login", 40), ("feature_use:dashboard", 20), ("feature_use:export", 12),
    ("feature_use:api", 10), ("support_contact", 8), ("invoice_viewed", 10),
]
TICKET_CATEGORIES = ["billing", "bug_report", "feature_request", "onboarding", "data_export"]
TICKET_PRIORITIES_WEIGHTED = [("low", 40), ("medium", 35), ("high", 20), ("urgent", 5)]
CAMPAIGN_CHANNELS = ["paid_search", "content", "webinar", "partner", "outbound", "conference"]
CAMPAIGN_NAME_ADJ = ["Northwind", "Ascend", "Signal", "Compass", "Horizon", "Catalyst"]
CAMPAIGN_NAME_NOUN = ["Q1 Push", "Launch", "Roadshow", "Series", "Summit", "Sprint"]

N_CUSTOMERS = 80
N_LEGACY_CUSTOMERS = 20
N_EMPLOYEES = 12
N_EVENTS = 600
N_INVOICES = 160
N_TICKETS = 100
N_CAMPAIGNS = 6


def _random_date(rng: random.Random, start: date, end: date) -> date:
    span = (end - start).days
    return start + timedelta(days=rng.randint(0, max(span, 0)))


def _weighted_choice(rng: random.Random, weighted: list[tuple[str, int]]) -> str:
    total = sum(w for _, w in weighted)
    pick = rng.uniform(0, total)
    upto = 0.0
    for value, weight in weighted:
        upto += weight
        if pick <= upto:
            return value
    return weighted[-1][0]


@dataclass
class Employee:
    employee_id: int
    name: str
    team: str
    role: str
    hire_date: date


@dataclass
class Customer:
    customer_id: int
    name: str
    industry: str
    country: str
    plan_tier: str
    signup_date: date
    status: str
    owner_employee_id: int


def generate_company_name(rng: random.Random, used: set[str]) -> str:
    while True:
        candidate = f"{rng.choice(COMPANY_NAME_PARTS_1)} {rng.choice(COMPANY_NAME_PARTS_2)}"
        if candidate not in used:
            used.add(candidate)
            return candidate


def generate_person_name(rng: random.Random, used: set[str]) -> str:
    while True:
        candidate = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        if candidate not in used:
            used.add(candidate)
            return candidate


def generate_employees(rng: random.Random) -> list[Employee]:
    used_names: set[str] = set()
    employees = []
    for i in range(1, N_EMPLOYEES + 1):
        team, role = TEAMS_ROLES[(i - 1) % len(TEAMS_ROLES)]
        hire_date = _random_date(rng, COMPANY_FOUNDED, REFERENCE_DATE - timedelta(days=30))
        employees.append(
            Employee(
                employee_id=i,
                name=generate_person_name(rng, used_names),
                team=team,
                role=role,
                hire_date=hire_date,
            )
        )
    return employees


def generate_customers(rng: random.Random, employees: list[Employee]) -> list[Customer]:
    used_names: set[str] = set()
    customers = []
    account_owners = [e.employee_id for e in employees if e.team in ("Sales", "Customer Success")]
    for i in range(1, N_CUSTOMERS + 1):
        signup_date = _random_date(rng, COMPANY_FOUNDED + timedelta(days=60), REFERENCE_DATE - timedelta(days=7))
        customers.append(
            Customer(
                customer_id=i,
                name=generate_company_name(rng, used_names),
                industry=rng.choice(INDUSTRIES),
                country=rng.choice(COUNTRIES),
                plan_tier=rng.choice(PLAN_TIERS),
                signup_date=signup_date,
                status=_weighted_choice(rng, CUSTOMER_STATUSES_WEIGHTED),
                owner_employee_id=rng.choice(account_owners),
            )
        )
    return customers


def generate_legacy_customers(rng: random.Random, customers: list[Customer]) -> list[tuple]:
    """Old CRM export: a subset of *current* customers under a disjoint id
    space, pre-migration tier labels, and a coarser join date field — plus a
    few legacy-only companies that no longer exist in `customers` at all.
    """
    rows = []
    legacy_id = 1000  # disjoint from customers.customer_id on purpose
    overlap_sample = rng.sample(customers, k=min(10, len(customers)))
    for customer in overlap_sample:
        joined_on = customer.signup_date - timedelta(days=rng.randint(0, 20))
        rows.append(
            (
                legacy_id,
                customer.name,
                joined_on.isoformat(),
                LEGACY_TIER_LABELS[customer.plan_tier],
                "Migrated to `customers` table in 2024 CRM cutover.",
            )
        )
        legacy_id += 1
    used_names = {c.name for c in customers}
    while legacy_id < 1000 + N_LEGACY_CUSTOMERS:
        name = generate_company_name(rng, used_names)
        joined_on = _random_date(rng, COMPANY_FOUNDED, COMPANY_FOUNDED + timedelta(days=400))
        rows.append(
            (
                legacy_id,
                name,
                joined_on.isoformat(),
                rng.choice(list(LEGACY_TIER_LABELS.values())),
                "Contract lapsed pre-2024 CRM cutover; not carried into `customers`.",
            )
        )
        legacy_id += 1
    return rows


def generate_subscriptions(rng: random.Random, customers: list[Customer]) -> list[tuple]:
    mrr_by_tier = {"Starter": (49, 149), "Growth": (299, 899), "Enterprise": (1500, 6000)}
    rows = []
    sub_id = 1
    for customer in customers:
        lo, hi = mrr_by_tier[customer.plan_tier]
        start = customer.signup_date
        if customer.status == "churned":
            end = _random_date(rng, start + timedelta(days=30), REFERENCE_DATE)
            status = "canceled"
        elif customer.status == "trial":
            end = None
            status = "trialing"
        else:
            end = None
            status = "active"
        rows.append((sub_id, customer.customer_id, customer.plan_tier, rng.randint(lo, hi),
                     start.isoformat(), end.isoformat() if end else None, status))
        sub_id += 1
    return rows


def generate_events(rng: random.Random, customers: list[Customer]) -> list[tuple]:
    rows = []
    active_customers = [c for c in customers if c.status != "churned"]
    for event_id in range(1, N_EVENTS + 1):
        customer = rng.choice(active_customers if rng.random() < 0.85 else customers)
        window_start = max(customer.signup_date, REFERENCE_DATE - timedelta(days=180))
        event_date = _random_date(rng, window_start, REFERENCE_DATE)
        rows.append((event_id, customer.customer_id, _weighted_choice(rng, EVENT_TYPES_WEIGHTED),
                     event_date.isoformat()))
    return rows


def generate_invoices(rng: random.Random, customers: list[Customer]) -> list[tuple]:
    amount_by_tier = {"Starter": (49, 149), "Growth": (299, 899), "Enterprise": (1500, 6000)}
    rows = []
    for invoice_id in range(1, N_INVOICES + 1):
        customer = rng.choice(customers)
        lo, hi = amount_by_tier[customer.plan_tier]
        invoice_date = _random_date(rng, customer.signup_date, REFERENCE_DATE)
        status = _weighted_choice(rng, [("paid", 80), ("overdue", 12), ("refunded", 8)])
        rows.append((invoice_id, customer.customer_id, rng.randint(lo, hi), invoice_date.isoformat(), status))
    return rows


def generate_tickets(rng: random.Random, customers: list[Customer], employees: list[Employee]) -> list[tuple]:
    support_staff = [e.employee_id for e in employees if e.team == "Support"] or [e.employee_id for e in employees]
    rows = []
    for ticket_id in range(1, N_TICKETS + 1):
        customer = rng.choice(customers)
        opened = _random_date(rng, customer.signup_date, REFERENCE_DATE)
        is_closed = rng.random() < 0.75
        closed = opened + timedelta(days=rng.randint(0, 10)) if is_closed else None
        rows.append((ticket_id, customer.customer_id, opened.isoformat(),
                     closed.isoformat() if closed else None,
                     rng.choice(TICKET_CATEGORIES), _weighted_choice(rng, TICKET_PRIORITIES_WEIGHTED),
                     rng.choice(support_staff)))
    return rows


def generate_campaigns(rng: random.Random) -> list[tuple]:
    rows = []
    used_names: set[str] = set()
    for campaign_id in range(1, N_CAMPAIGNS + 1):
        while True:
            name = f"{rng.choice(CAMPAIGN_NAME_ADJ)} {rng.choice(CAMPAIGN_NAME_NOUN)}"
            if name not in used_names:
                used_names.add(name)
                break
        start = _random_date(rng, COMPANY_FOUNDED, REFERENCE_DATE - timedelta(days=30))
        end = start + timedelta(days=rng.randint(14, 60))
        rows.append((campaign_id, name, rng.choice(CAMPAIGN_CHANNELS), start.isoformat(), end.isoformat(),
                     rng.randint(2000, 40000)))
    return rows


def generate_campaign_attributions(rng: random.Random, campaigns: list[tuple], customers: list[Customer]) -> list[tuple]:
    rows = []
    seen: set[tuple[int, int]] = set()
    n_attributions = 60
    attempts = 0
    while len(rows) < n_attributions and attempts < n_attributions * 20:
        attempts += 1
        campaign = rng.choice(campaigns)
        customer = rng.choice(customers)
        key = (campaign[0], customer.customer_id)
        if key in seen:
            continue
        seen.add(key)
        campaign_start = date.fromisoformat(campaign[3])
        campaign_end = date.fromisoformat(campaign[4])
        if customer.signup_date > campaign_end + timedelta(days=30):
            continue
        attributed_date = _random_date(rng, campaign_start, campaign_end)
        rows.append((campaign[0], customer.customer_id, attributed_date.isoformat()))
    return rows


def build_database(rng: random.Random, db_path: Path) -> dict:
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)

        employees = generate_employees(rng)
        conn.executemany(
            "INSERT INTO employees VALUES (?, ?, ?, ?, ?)",
            [(e.employee_id, e.name, e.team, e.role, e.hire_date.isoformat()) for e in employees],
        )

        customers = generate_customers(rng, employees)
        conn.executemany(
            "INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(c.customer_id, c.name, c.industry, c.country, c.plan_tier,
              c.signup_date.isoformat(), c.status, c.owner_employee_id) for c in customers],
        )

        conn.executemany("INSERT INTO customers_legacy VALUES (?, ?, ?, ?, ?)",
                          generate_legacy_customers(rng, customers))
        conn.executemany("INSERT INTO subscriptions VALUES (?, ?, ?, ?, ?, ?, ?)",
                          generate_subscriptions(rng, customers))
        conn.executemany("INSERT INTO events VALUES (?, ?, ?, ?)", generate_events(rng, customers))
        conn.executemany("INSERT INTO invoices VALUES (?, ?, ?, ?, ?)", generate_invoices(rng, customers))
        conn.executemany("INSERT INTO support_tickets VALUES (?, ?, ?, ?, ?, ?, ?)",
                          generate_tickets(rng, customers, employees))

        campaigns = generate_campaigns(rng)
        conn.executemany("INSERT INTO campaigns VALUES (?, ?, ?, ?, ?, ?)", campaigns)
        conn.executemany("INSERT INTO campaign_attributions VALUES (?, ?, ?)",
                          generate_campaign_attributions(rng, campaigns, customers))

        conn.commit()
        content_hash = hashlib.sha256("\n".join(conn.iterdump()).encode("utf-8")).hexdigest()
        table_counts = {
            row[0]: conn.execute(f"SELECT COUNT(*) FROM {row[0]}").fetchone()[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        conn.close()
    return {
        "employees": employees,
        "customers": customers,
        "content_hash": content_hash,
        "table_counts": table_counts,
    }


# --- Documents -------------------------------------------------------------


def doc_active_customer_definition_current() -> str:
    return f"""# Policy: Active Customer Definition (v2 — current)

**Status:** Active · **Effective:** 2025-06-01 · **Owner:** Data team
**Supersedes:** `active_customer_definition_v1_DEPRECATED.md` (60-day window)

## Definition

A customer is **Active** if they have logged at least one qualifying
engagement event in the trailing **90 days**. Qualifying events are any row
in `events` with `event_type` in:

- `login`
- `feature_use:dashboard`
- `feature_use:export`
- `feature_use:api`

`support_contact` and `invoice_viewed` do **not** count as engagement on
their own — a customer who only files tickets or looks at invoices is not
considered Active under this policy.

## Why this changed from 60 to 90 days

The 60-day window (v1) was flagging too many Enterprise accounts as
"at risk" during normal end-of-quarter lulls in usage. The Data team and
Customer Success agreed on the 90-day window at the 2025-06-01 policy
review; this is the definition all churn and health-score dashboards
should use going forward.

## Usage note

If you see a metric or report still using a 60-day window, it is using the
deprecated v1 definition and should be flagged for correction.
"""


def doc_active_customer_definition_deprecated() -> str:
    return f"""# Policy: Active Customer Definition (v1 — DEPRECATED)

**Status:** DEPRECATED as of 2025-06-01 · **Owner:** Data team
**Superseded by:** `active_customer_definition.md` (90-day window)

## Definition (no longer in use)

A customer is Active if they have logged at least one engagement event in
the trailing **60 days**.

This document is retained only for historical audit purposes (some
2024-era reports were generated against this definition and are cited in
old board decks). Do not use this 60-day window for any new analysis —
see the current policy for the 90-day definition in effect since
2025-06-01.
"""


def doc_metrics_glossary() -> str:
    return """# Metrics Glossary

**Owner:** Data team

## MRR (Monthly Recurring Revenue)
Sum of `subscriptions.mrr` across all subscriptions with `status = 'active'`.
Trialing and canceled subscriptions are excluded.

## ARPU (Average Revenue Per User)
MRR divided by the count of customers with `status = 'active'`.

## Churn Rate (monthly)
Count of customers whose `status` changed to `churned` in a given month,
divided by the count of customers with `status = 'active'` at the start of
that month.

## NRR (Net Revenue Retention)
(Starting MRR of a cohort + expansion − contraction − churned MRR) /
Starting MRR of that cohort, measured over a trailing 12-month window.

## Active Customer
See `policies/active_customer_definition.md` for the current definition
(90-day engagement window). Do not hardcode a different window in ad hoc
analysis — always cite the current policy doc.
"""


def doc_support_sla() -> str:
    return """# Support SLA

**Owner:** Support team

| Priority | First Response | Resolution Target |
|----------|-----------------|--------------------|
| urgent   | 1 hour          | 8 hours            |
| high     | 4 hours         | 2 business days    |
| medium   | 1 business day  | 5 business days    |
| low      | 2 business days | best effort        |

Priority is set at ticket creation (`support_tickets.priority`) and should
not be downgraded without Support lead sign-off.
"""


def doc_refund_policy() -> str:
    return """# Refund Policy

**Owner:** Finance, in coordination with Customer Success

- Refunds are issued for billing errors (double charge, incorrect plan
  tier applied) with no time limit.
- Dissatisfaction-based refund requests are honored within **14 days** of
  the invoice date, prorated to unused days in the billing period.
- Enterprise contracts follow the refund terms in their signed order form
  where those differ from this default policy.
- All refunds are recorded by setting `invoices.status = 'refunded'`.
"""


def doc_data_team_charter() -> str:
    return f"""# Data Team Charter

**Company:** {COMPANY_NAME}

The Data team owns the definitions used across all internal reporting —
active-customer status, MRR, churn, and campaign attribution — so that
Sales, Success, and Finance dashboards agree with each other.

When a definition changes (see `policies/`), the Data team is responsible
for updating the glossary and flagging any downstream report still using
the old definition. Cross-check `policies/active_customer_definition.md`
before citing an "active customer" count in any new analysis — a
deprecated version of this policy exists in the same folder for audit
purposes only and should never be used going forward.
"""


def doc_team_roster(employees: list[Employee]) -> str:
    lines = [
        "# Team Roster",
        "",
        "**Owner:** People Ops · generated from `employees` table",
        "",
        "| Employee ID | Name | Team | Role | Hire Date |",
        "|---|---|---|---|---|",
    ]
    for e in sorted(employees, key=lambda e: e.employee_id):
        lines.append(f"| {e.employee_id} | {e.name} | {e.team} | {e.role} | {e.hire_date.isoformat()} |")
    lines.append("")
    return "\n".join(lines)


def write_docs(docs_dir: Path, employees: list[Employee]) -> dict:
    if docs_dir.exists():
        for path in sorted(docs_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            else:
                path.rmdir()
    docs = {
        "policies/active_customer_definition.md": doc_active_customer_definition_current(),
        "policies/active_customer_definition_v1_DEPRECATED.md": doc_active_customer_definition_deprecated(),
        "glossary/metrics_glossary.md": doc_metrics_glossary(),
        "handbook/support_sla.md": doc_support_sla(),
        "handbook/refund_policy.md": doc_refund_policy(),
        "handbook/data_team_charter.md": doc_data_team_charter(),
        "team/roster.md": doc_team_roster(employees),
    }
    doc_hashes = {}
    for relative_path, content in docs.items():
        full_path = docs_dir / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8", newline="\n")
        doc_hashes[relative_path] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return doc_hashes


def build(seed: int, db_path: Path, docs_dir: Path, manifest_path: Path) -> dict:
    rng = random.Random(seed)
    db_result = build_database(rng, db_path)
    doc_hashes = write_docs(docs_dir, db_result["employees"])
    manifest = {
        "seed": seed,
        "company_name": COMPANY_NAME,
        "reference_date": REFERENCE_DATE.isoformat(),
        "db_content_hash": db_result["content_hash"],
        "table_counts": db_result["table_counts"],
        "doc_hashes": doc_hashes,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed (default: 42)")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    args = parser.parse_args()

    manifest = build(args.seed, args.db_path, args.docs_dir, args.manifest_path)
    print(f"Built environment for seed={args.seed}")
    print(f"  DB:       {args.db_path}")
    print(f"  Docs:     {args.docs_dir} ({len(manifest['doc_hashes'])} files)")
    print(f"  Manifest: {args.manifest_path}")
    print(f"  Table counts: {manifest['table_counts']}")


if __name__ == "__main__":
    main()
