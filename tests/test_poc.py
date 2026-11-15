"""Tests for the Part 2 ingestion proof of concept.

They need PostgreSQL (`docker compose up -d`, or POC_DATABASE_URL) and are
skipped when it isn't reachable. Each test runs in its own throwaway schema.

Run with:  python -m pytest tests -q
"""

import json
import os
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

psycopg = pytest.importorskip("psycopg")

from poc.db import DEFAULT_URL, SCHEMA  # noqa: E402
from poc.register import register  # noqa: E402
from poc.worker import (LeaseLost, claim_job, process, record_failure,  # noqa: E402
                        save_results)


def patent(number, title="WHEEL", claim="1 . A wheel comprising a rim and a hub."):
    return {"doc_number": number, "title": title, "classification": "B60B100FI",
            "abstract": f"A wheel numbered {number}.", "claims": [claim],
            "detailed_description": ["The wheel has a rim and a hub joined by spokes."]}


def database_url():
    return os.environ.get("POC_DATABASE_URL", DEFAULT_URL)


@pytest.fixture
def conn():
    try:
        connection = psycopg.connect(database_url(), autocommit=True, connect_timeout=3)
    except psycopg.OperationalError:
        pytest.skip("PostgreSQL not reachable; start it with `docker compose up -d`")
    schema = f"test_{uuid.uuid4().hex[:12]}"
    connection.execute(f"CREATE SCHEMA {schema}")
    connection.execute(f"SET search_path TO {schema}")
    connection.execute(SCHEMA)
    yield connection
    connection.execute(f"DROP SCHEMA {schema} CASCADE")
    connection.close()


@pytest.fixture
def write(tmp_path, monkeypatch):
    """Write a weekly file; paths are made relative, as register stores them."""
    monkeypatch.chdir(tmp_path)

    def _write(name, records):
        path = tmp_path / name
        path.write_text(records if isinstance(records, str) else json.dumps(records))
        return name
    return _write


def one(conn, sql, *params):
    return conn.execute(sql, params).fetchone()


def run(conn, worker="w1", **kwargs):
    job = claim_job(conn, worker)
    return job, process(conn, job, **kwargs)


# -- registering ---------------------------------------------------------------
def test_same_file_registered_twice_is_one_job(conn, write):
    path = write("patents_ipa240101.json", [patent("1")])
    first, created, _ = register(conn, path)
    second, created_again, _ = register(conn, path)
    assert created and not created_again and first == second
    assert one(conn, "SELECT count(*) FROM ingestion_jobs")[0] == 1


def test_changed_file_is_a_new_job(conn, write):
    path = write("patents_ipa240101.json", [patent("1")])
    register(conn, path)
    write("patents_ipa240101.json", [patent("1"), patent("2")])
    _, created, _ = register(conn, path)
    assert created and one(conn, "SELECT count(*) FROM ingestion_jobs")[0] == 2


# -- processing ----------------------------------------------------------------
def test_worker_saves_every_patent_and_completes(conn, write):
    register(conn, write("patents_ipa240101.json", [patent("1"), patent("2")]))
    _, summary = run(conn)
    assert summary.startswith("complete (2 saved")
    state, total, saved, token = one(
        conn, "SELECT state, records_total, records_saved, lease_token FROM ingestion_jobs")
    assert (state, total, saved, token) == ("complete", 2, 2, None)
    assert one(conn, "SELECT count(*), sum(passages) FROM publication_versions") == (2, 6)


def test_bad_records_are_quarantined_and_the_rest_saved(conn, write):
    records = [patent("1"), patent(""), "not a record",
               {"doc_number": "4", "title": "", "claims": []}]
    register(conn, write("patents_ipa240101.json", records))
    run(conn)
    assert one(conn, "SELECT state, records_quarantined FROM ingestion_jobs") == \
        ("complete_with_errors", 3)
    reasons = [r for (r,) in conn.execute(
        "SELECT error_reason FROM source_records WHERE outcome = 'quarantined' "
        "ORDER BY record_position")]
    assert reasons == ["missing doc_number (patent ID)", "record is not a JSON object",
                       "no title, abstract, claims or description"]
    assert one(conn, "SELECT count(*) FROM publication_versions")[0] == 1


def test_corrupt_file_fails_at_once_without_retrying(conn, write):
    register(conn, write("patents_ipa240101.json", '[{"doc_number": "1", "title": "CUT'))
    _, summary = run(conn)
    assert summary.startswith("failed: file is not valid JSON")
    assert one(conn, "SELECT state, attempt FROM ingestion_jobs") == ("failed", 1)


def test_temporary_error_is_retried_and_gives_up_after_max_attempts(conn, write):
    register(conn, write("patents_ipa240101.json", [patent("1")]))
    _, summary = run(conn, fail_on_attempt=1)
    assert summary.startswith("retry_wait")
    assert claim_job(conn, "w1") is None                  # not due yet
    conn.execute("UPDATE ingestion_jobs SET next_attempt_at = now()")
    job, summary = run(conn, fail_on_attempt=1)
    assert job.attempt == 2 and summary.startswith("complete")

    # A job that keeps failing ends as failed after its last attempt.
    register(conn, write("patents_ipa240108.json", [patent("2")]))
    conn.execute("UPDATE ingestion_jobs SET attempt = max_attempts - 1 WHERE state = 'queued'")
    job = claim_job(conn, "w1")
    assert record_failure(conn, job, RuntimeError("still down"), permanent=False) == "failed"


# -- several workers, crashes and duplicates -------------------------------------
def test_a_job_locked_by_one_worker_is_skipped_by_another(conn, write):
    register(conn, write("patents_ipa240101.json", [patent("1")]))
    register(conn, write("patents_ipa240108.json", [patent("2")]))
    # conn.info.dsn leaves the password out, so connect with the full URL.
    other = psycopg.connect(database_url(), autocommit=False)
    other.execute(f"SET search_path TO {conn.execute('SHOW search_path').fetchone()[0]}")
    other.execute("SELECT id FROM ingestion_jobs WHERE id = 1 FOR UPDATE")  # worker A, mid-claim
    job = claim_job(conn, "B")
    other.rollback()
    other.close()
    assert job.id == 2


def test_expired_lease_is_taken_over_and_the_old_worker_cannot_finish(conn, write):
    register(conn, write("patents_ipa240101.json", [patent("1"), patent("2")]))
    stale = claim_job(conn, "A", lease_seconds=0)           # A "dies": its lease expires
    fresh = claim_job(conn, "B")
    assert fresh.id == stale.id and fresh.attempt == 2
    assert fresh.previous_error == "lease of A expired"
    assert process(conn, fresh).startswith("complete")
    # A wakes up and tries to finish: its token is no longer valid.
    with pytest.raises(LeaseLost):
        save_results(conn, stale, [])
    assert process(conn, stale).startswith("lease lost")
    assert one(conn, "SELECT count(*) FROM publication_versions")[0] == 2


def test_lease_expiring_on_the_final_attempt_fails_the_job(conn, write):
    register(conn, write("patents_ipa240101.json", [patent("1")]))
    conn.execute("UPDATE ingestion_jobs SET attempt = max_attempts - 1")
    claim_job(conn, "A", lease_seconds=0)
    assert claim_job(conn, "B") is None
    state, error = one(conn, "SELECT state, last_error FROM ingestion_jobs")
    assert state == "failed" and "expired on the final attempt" in error


def test_same_patent_again_is_unchanged_and_different_content_is_a_conflict(conn, write):
    register(conn, write("patents_ipa240101.json", [patent("1")]))
    run(conn)
    register(conn, write("patents_ipa240108.json", [patent("1")]))              # same text
    register(conn, write("patents_ipa240115.json", [patent("1", "WHEEL, CORRECTED")]))
    assert run(conn)[1].startswith("complete (1 unchanged")
    assert run(conn)[1].startswith("complete (1 conflict")
    rows = conn.execute("SELECT state FROM publication_versions ORDER BY saved_at").fetchall()
    assert [state for (state,) in rows] == ["saved", "conflict"]