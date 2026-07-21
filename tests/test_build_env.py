from pathlib import Path

import pytest

from environment.build_env import (
    DEFAULT_DB_PATH,
    N_CAMPAIGNS,
    N_CUSTOMERS,
    N_EMPLOYEES,
    N_EVENTS,
    N_INVOICES,
    N_LEGACY_CUSTOMERS,
    N_TICKETS,
    build,
)


@pytest.fixture
def env_dirs(tmp_path: Path):
    return {
        "db_path": tmp_path / "company.db",
        "docs_dir": tmp_path / "docs",
        "manifest_path": tmp_path / "manifest.json",
    }


def test_same_seed_produces_identical_content(tmp_path: Path):
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    run_a.mkdir()
    run_b.mkdir()

    manifest_a = build(42, run_a / "company.db", run_a / "docs", run_a / "manifest.json")
    manifest_b = build(42, run_b / "company.db", run_b / "docs", run_b / "manifest.json")

    assert manifest_a["db_content_hash"] == manifest_b["db_content_hash"]
    assert manifest_a["doc_hashes"] == manifest_b["doc_hashes"]
    assert manifest_a["table_counts"] == manifest_b["table_counts"]

    for relative_path in manifest_a["doc_hashes"]:
        text_a = (run_a / "docs" / relative_path).read_text(encoding="utf-8")
        text_b = (run_b / "docs" / relative_path).read_text(encoding="utf-8")
        assert text_a == text_b


def test_different_seed_produces_different_content(tmp_path: Path):
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    run_a.mkdir()
    run_b.mkdir()

    manifest_a = build(42, run_a / "company.db", run_a / "docs", run_a / "manifest.json")
    manifest_b = build(43, run_b / "company.db", run_b / "docs", run_b / "manifest.json")

    assert manifest_a["db_content_hash"] != manifest_b["db_content_hash"]


def test_table_row_counts_match_constants(env_dirs):
    manifest = build(42, **env_dirs)
    counts = manifest["table_counts"]
    assert counts["customers"] == N_CUSTOMERS
    assert counts["customers_legacy"] == N_LEGACY_CUSTOMERS
    assert counts["employees"] == N_EMPLOYEES
    assert counts["events"] == N_EVENTS
    assert counts["invoices"] == N_INVOICES
    assert counts["support_tickets"] == N_TICKETS
    assert counts["campaigns"] == N_CAMPAIGNS


def test_expected_docs_are_written(env_dirs):
    manifest = build(42, **env_dirs)
    expected = {
        "policies/active_customer_definition.md",
        "policies/active_customer_definition_v1_DEPRECATED.md",
        "glossary/metrics_glossary.md",
        "handbook/support_sla.md",
        "handbook/refund_policy.md",
        "handbook/data_team_charter.md",
        "team/roster.md",
    }
    assert set(manifest["doc_hashes"]) == expected
    for relative_path in expected:
        assert (env_dirs["docs_dir"] / relative_path).exists()


def test_deprecated_doc_states_60_days_current_states_90(env_dirs):
    build(42, **env_dirs)
    current = (env_dirs["docs_dir"] / "policies/active_customer_definition.md").read_text(encoding="utf-8")
    deprecated = (env_dirs["docs_dir"] / "policies/active_customer_definition_v1_DEPRECATED.md").read_text(encoding="utf-8")
    assert "90 days" in current
    assert "60 days" in deprecated
    assert "DEPRECATED" in deprecated


def test_legacy_customers_use_disjoint_id_space_from_customers(env_dirs):
    import sqlite3

    build(42, **env_dirs)
    conn = sqlite3.connect(env_dirs["db_path"])
    try:
        customer_ids = {row[0] for row in conn.execute("SELECT customer_id FROM customers")}
        legacy_ids = {row[0] for row in conn.execute("SELECT legacy_id FROM customers_legacy")}
    finally:
        conn.close()
    assert customer_ids.isdisjoint(legacy_ids)


def test_referential_integrity(env_dirs):
    import sqlite3

    build(42, **env_dirs)
    conn = sqlite3.connect(env_dirs["db_path"])
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        customer_ids = {row[0] for row in conn.execute("SELECT customer_id FROM customers")}
        employee_ids = {row[0] for row in conn.execute("SELECT employee_id FROM employees")}

        for table, col in [
            ("subscriptions", "customer_id"),
            ("events", "customer_id"),
            ("invoices", "customer_id"),
            ("support_tickets", "customer_id"),
            ("campaign_attributions", "customer_id"),
        ]:
            referenced = {row[0] for row in conn.execute(f"SELECT DISTINCT {col} FROM {table}")}
            assert referenced.issubset(customer_ids), f"{table}.{col} has dangling customer_id"

        owner_ids = {row[0] for row in conn.execute("SELECT DISTINCT owner_employee_id FROM customers")}
        assert owner_ids.issubset(employee_ids)

        handler_ids = {row[0] for row in conn.execute("SELECT DISTINCT handled_by FROM support_tickets")}
        assert handler_ids.issubset(employee_ids)
    finally:
        conn.close()
