"""Process ingestion jobs: claim one, parse and clean its file, save the patents.

    python -m poc.worker                      # run until every job is finished
    python -m poc.worker --name w2 --slow 1   # 1 s per record, to stop it mid-job

How the guarantees from docs/PART2_DESIGN.md (section 5) are met:

* Claiming: a short transaction with FOR UPDATE SKIP LOCKED, so two workers
  never take the same job. The job then gets a lease (lease_until) and a fresh
  lease_token.
* Crash recovery: if a worker dies, its lease runs out and another worker
  claims the job again. Every later write checks the lease_token, so a worker
  that wakes up after losing its lease can't overwrite the new attempt.
* No duplicates: a file's results are written in one transaction together
  with the job's final state, and patents are keyed by (ID, content hash).
  Re-processing a file changes nothing; the same ID with different content is
  kept as a conflict, never overwritten.
* Retries: a temporary error sends the job to retry_wait with a growing delay;
  after max_attempts it becomes failed. A file that can never be read (invalid
  JSON) fails at once.
* Bad records are quarantined with their reason; the rest of the file is
  saved, and the job ends as complete_with_errors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.load_data import parse_record, publication_date_from_filename  # noqa: E402

from .db import connect  # noqa: E402

LEASE_SECONDS = 30
# Wait before attempt 2, attempt 3, ... Short, so retries are visible in a demo.
RETRY_DELAYS_S = (5, 20, 80)
HEARTBEAT_EVERY_S = 5


class PermanentError(Exception):
    """Retrying can't help, for example a file that isn't valid JSON."""


class LeaseLost(Exception):
    """Another worker now owns the job, because this worker's lease ran out."""


@dataclass
class Job:
    id: int
    source_key: str
    attempt: int
    max_attempts: int
    lease_token: object
    previous_error: str | None


@dataclass
class Outcome:
    position: int
    publication_id: str | None
    error: str = ""
    patent: dict = field(default_factory=dict)


# -- claiming and leases -------------------------------------------------------
def claim_job(conn: psycopg.Connection, worker: str,
              lease_seconds: int = LEASE_SECONDS) -> Job | None:
    """Take the next runnable job, or None. Expired leases are taken over."""
    with conn.transaction():
        # A job whose final attempt died with its worker can't be retried again.
        conn.execute(
            """UPDATE ingestion_jobs
               SET state = 'failed', finished_at = now(), lease_token = NULL,
                   last_error = 'lease of ' || coalesce(worker, '?') ||
                                ' expired on the final attempt'
               WHERE state = 'processing' AND lease_until < now()
                 AND attempt >= max_attempts""")
        row = conn.execute(
            """WITH picked AS (
                 SELECT id FROM ingestion_jobs
                 WHERE (state IN ('queued', 'retry_wait') AND next_attempt_at <= now())
                    OR (state = 'processing' AND lease_until < now())
                 ORDER BY id
                 FOR UPDATE SKIP LOCKED
                 LIMIT 1)
               UPDATE ingestion_jobs j
               SET state = 'processing',
                   attempt = j.attempt + 1,
                   lease_until = now() + make_interval(secs => %s),
                   lease_token = gen_random_uuid(),
                   worker = %s,
                   started_at = coalesce(j.started_at, now()),
                   last_error = CASE WHEN j.state = 'processing'
                                     THEN 'lease of ' || coalesce(j.worker, '?') || ' expired'
                                     ELSE j.last_error END
               FROM picked WHERE j.id = picked.id
               RETURNING j.id, j.source_key, j.attempt, j.max_attempts, j.lease_token,
                         j.last_error""",
            (lease_seconds, worker)).fetchone()
    return Job(*row) if row else None


def extend_lease(conn: psycopg.Connection, job: Job, lease_seconds: int) -> None:
    updated = conn.execute(
        """UPDATE ingestion_jobs SET lease_until = now() + make_interval(secs => %s)
           WHERE id = %s AND lease_token = %s AND state = 'processing'""",
        (lease_seconds, job.id, job.lease_token)).rowcount
    if not updated:
        raise LeaseLost(f"job {job.id}")


# -- parsing ---------------------------------------------------------------------
def content_hash(patent) -> str:
    """Hash of what the patent says; the file name and date are left out."""
    content = {"title": patent.title, "abstract": patent.abstract,
               "classification": patent.classification,
               "claims": [claim.raw_text for claim in patent.claims],
               "paragraphs": patent.raw_paragraphs}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode("utf-8")).hexdigest()


def check_record(record) -> str:
    """Why a record can't be used, or "" when it can."""
    if not isinstance(record, dict):
        return "record is not a JSON object"
    if not str(record.get("doc_number") or "").strip():
        return "missing doc_number (patent ID)"
    if not any(record.get(key) for key in ("title", "abstract", "claims",
                                           "detailed_description")):
        return "no title, abstract, claims or description"
    return ""


def read_file(path: str) -> list:
    try:
        with open(path, encoding="utf-8") as handle:
            records = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise PermanentError(f"file is not valid JSON: {error}") from None
    if not isinstance(records, list):
        raise PermanentError("file does not contain a list of patent records")
    return records


def parse_file(path: str, heartbeat, slow_s: float = 0.0) -> list[Outcome]:
    published = publication_date_from_filename(path)
    outcomes = []
    for position, record in enumerate(read_file(path), start=1):
        heartbeat()
        if slow_s:
            time.sleep(slow_s)
        problem = check_record(record)
        publication_id = (str(record.get("doc_number") or "").strip() or None
                          if isinstance(record, dict) else None)
        if problem:
            outcomes.append(Outcome(position, publication_id, problem))
            continue
        try:
            patent = parse_record(record, published)
        except Exception as error:  # one bad record must not sink the file
            outcomes.append(Outcome(position, publication_id,
                                    f"could not parse: {type(error).__name__}: {error}"))
            continue
        outcomes.append(Outcome(position, patent.patent_id, patent={
            "hash": content_hash(patent), "title": patent.title,
            "classification": patent.classification,
            "publication_date": patent.publication_date or None,
            "passages": int(bool(patent.abstract)) + len(patent.claims)
                        + len(patent.paragraphs)}))
    return outcomes


# -- saving ----------------------------------------------------------------------
def save_results(conn: psycopg.Connection, job: Job, outcomes: list[Outcome]) -> dict:
    """Write every patent, every record's outcome and the job's end state at once."""
    counts = {"saved": 0, "unchanged": 0, "conflict": 0, "quarantined": 0}
    with conn.transaction():
        owner = conn.execute(
            """SELECT 1 FROM ingestion_jobs
               WHERE id = %s AND lease_token = %s AND state = 'processing' FOR UPDATE""",
            (job.id, job.lease_token)).fetchone()
        if not owner:
            raise LeaseLost(f"job {job.id}")
        for outcome in outcomes:
            if outcome.error:
                result = "quarantined"
            else:
                p = outcome.patent
                # Serialise writers of the same patent ID, so "is it new?" can't race.
                conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",
                             (outcome.publication_id,))
                known = {h for (h,) in conn.execute(
                    "SELECT content_hash FROM publication_versions WHERE publication_id = %s",
                    (outcome.publication_id,))}
                if p["hash"] in known:
                    result = "unchanged"
                else:
                    result = "conflict" if known else "saved"
                    conn.execute(
                        """INSERT INTO publication_versions
                             (publication_id, content_hash, title, classification,
                              publication_date, passages, job_id, state)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                        (outcome.publication_id, p["hash"], p["title"], p["classification"],
                         p["publication_date"], p["passages"], job.id, result))
            counts[result] += 1
            conn.execute(
                """INSERT INTO source_records
                     (job_id, record_position, publication_id, outcome, error_reason)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (job_id, record_position) DO UPDATE
                   SET publication_id = EXCLUDED.publication_id,
                       outcome = EXCLUDED.outcome, error_reason = EXCLUDED.error_reason""",
                (job.id, outcome.position, outcome.publication_id, result,
                 outcome.error or None))
        state = "complete_with_errors" if counts["quarantined"] else "complete"
        conn.execute(
            """UPDATE ingestion_jobs
               SET state = %s, finished_at = now(), lease_until = NULL, lease_token = NULL,
                   records_total = %s, records_saved = %s, records_unchanged = %s,
                   records_conflict = %s, records_quarantined = %s
               WHERE id = %s""",
            (state, len(outcomes), counts["saved"], counts["unchanged"], counts["conflict"],
             counts["quarantined"], job.id))
    return {"state": state, **counts}


def record_failure(conn: psycopg.Connection, job: Job, error: Exception,
                   permanent: bool) -> str:
    """Send the job to retry_wait or failed; returns the new state."""
    final = permanent or job.attempt >= job.max_attempts
    delay = RETRY_DELAYS_S[min(job.attempt, len(RETRY_DELAYS_S)) - 1]
    state = "failed" if final else "retry_wait"
    updated = conn.execute(
        """UPDATE ingestion_jobs
           SET state = %s, last_error = %s, lease_until = NULL, lease_token = NULL,
               next_attempt_at = now() + make_interval(secs => %s),
               finished_at = CASE WHEN %s THEN now() ELSE NULL END
           WHERE id = %s AND lease_token = %s AND state = 'processing'""",
        (state, f"attempt {job.attempt}: {error}", 0 if final else delay, final,
         job.id, job.lease_token)).rowcount
    if not updated:
        raise LeaseLost(f"job {job.id}")
    return state if final else f"retry_wait (next try in {delay} s)"


# -- the loop --------------------------------------------------------------------
def process(conn: psycopg.Connection, job: Job, lease_seconds: int = LEASE_SECONDS,
            slow_s: float = 0.0, fail_on_attempt: int | None = None) -> str:
    """Run one claimed job to its next state; returns a one-line summary."""
    last_beat = [time.monotonic()]

    def heartbeat() -> None:
        if time.monotonic() - last_beat[0] >= HEARTBEAT_EVERY_S:
            extend_lease(conn, job, lease_seconds)
            last_beat[0] = time.monotonic()

    try:
        outcomes = parse_file(job.source_key, heartbeat, slow_s)
        if fail_on_attempt is not None and job.attempt == fail_on_attempt:
            raise RuntimeError("simulated temporary failure (--fail-on-attempt)")
        result = save_results(conn, job, outcomes)
    except LeaseLost:
        return "lease lost: another worker took this job over; nothing written"
    except PermanentError as error:
        record_failure(conn, job, error, True)
        return f"failed: {error}"
    except Exception as error:  # temporary until proven otherwise
        return f"{record_failure(conn, job, error, False)}: {error}"
    detail = ", ".join(f"{n} {name}" for name, n in result.items()
                       if name != "state" and n)
    return f"{result['state']} ({detail or 'no records'})"


def unfinished(conn: psycopg.Connection) -> int:
    return conn.execute(
        """SELECT count(*) FROM ingestion_jobs
           WHERE state IN ('queued', 'processing', 'retry_wait')""").fetchone()[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Process ingestion jobs.")
    parser.add_argument("--name", default=f"worker-{os.getpid()}")
    parser.add_argument("--lease-seconds", type=int, default=LEASE_SECONDS)
    parser.add_argument("--slow", type=float, default=0.0, metavar="SECONDS",
                        help="Pause per record, so the worker can be stopped mid-job")
    parser.add_argument("--fail-on-attempt", type=int, default=None, metavar="N",
                        help="Simulate a temporary error on attempt N of every job")
    parser.add_argument("--once", action="store_true", help="Process at most one job")
    args = parser.parse_args()

    conn = connect()
    waiting_shown = None
    while True:
        job = claim_job(conn, args.name, args.lease_seconds)
        if job:
            waiting_shown = None
            note = f" (previous: {job.previous_error})" if job.attempt > 1 and \
                job.previous_error else ""
            print(f"[{args.name}] job {job.id} attempt {job.attempt}: "
                  f"{os.path.basename(job.source_key)}{note}", flush=True)
            try:
                summary = process(conn, job, args.lease_seconds, args.slow,
                                  args.fail_on_attempt)
            except KeyboardInterrupt:
                print(f"\n[{args.name}] stopped mid-job. Job {job.id} stays 'processing' "
                      f"until its lease expires, then another worker retries it.")
                return
            print(f"[{args.name}]   → {summary}", flush=True)
            if args.once:
                return
            continue
        remaining = unfinished(conn)
        if not remaining:
            print(f"[{args.name}] all jobs finished.")
            return
        if args.once:
            return
        if remaining != waiting_shown:
            print(f"[{args.name}] waiting: {remaining} job(s) in retry_wait or held by "
                  "another worker", flush=True)
            waiting_shown = remaining
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
