"""Show what the pipeline contains and what state it is in.

    python -m poc.status           # summary
    python -m poc.status --jobs    # plus one line per job
"""

from __future__ import annotations

import argparse
import os

from .db import connect

STATES = ("queued", "processing", "retry_wait", "complete", "complete_with_errors", "failed")


def gather(conn) -> dict:
    jobs = dict.fromkeys(STATES, 0)
    jobs.update(dict(conn.execute(
        "SELECT state, count(*) FROM ingestion_jobs GROUP BY state").fetchall()))
    outcomes = dict(conn.execute(
        "SELECT outcome, count(*) FROM source_records GROUP BY outcome").fetchall())
    patents = conn.execute(
        """SELECT count(DISTINCT publication_id),
                  count(*) FILTER (WHERE state = 'conflict'),
                  coalesce(sum(passages) FILTER (WHERE state = 'saved'), 0)
           FROM publication_versions""").fetchone()
    oldest = conn.execute(
        """SELECT extract(epoch FROM now() - min(created_at))
           FROM ingestion_jobs WHERE state IN ('queued', 'retry_wait', 'processing')"""
    ).fetchone()[0]
    retried = conn.execute(
        "SELECT count(*) FROM ingestion_jobs WHERE attempt > 1").fetchone()[0]
    quarantined = conn.execute(
        """SELECT j.source_key, r.record_position, r.publication_id, r.error_reason
           FROM source_records r JOIN ingestion_jobs j ON j.id = r.job_id
           WHERE r.outcome = 'quarantined'
           ORDER BY j.id, r.record_position LIMIT 10""").fetchall()
    failed = conn.execute(
        """SELECT id, source_key, last_error FROM ingestion_jobs
           WHERE state = 'failed' ORDER BY id LIMIT 10""").fetchall()
    return {"jobs": jobs, "outcomes": outcomes, "patents": patents[0],
            "conflicts": patents[1], "passages": patents[2], "oldest_s": oldest,
            "retried": retried, "quarantined": quarantined, "failed": failed}


def age(seconds) -> str:
    if seconds is None:
        return "none"
    seconds = int(seconds)
    return f"{seconds // 3600}h {seconds % 3600 // 60}m" if seconds >= 3600 else \
        f"{seconds // 60}m {seconds % 60}s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline contents and status.")
    parser.add_argument("--jobs", action="store_true", help="List every job")
    args = parser.parse_args()
    conn = connect()
    s = gather(conn)
    j, o = s["jobs"], s["outcomes"]

    print(f"JOBS      {sum(j.values())} total · " +
          " · ".join(f"{state} {j[state]}" for state in STATES))
    print(f"          oldest unfinished: {age(s['oldest_s'])} · "
          f"needed more than one attempt: {s['retried']}")
    print(f"RECORDS   {sum(o.values())} read · saved {o.get('saved', 0)} · "
          f"unchanged {o.get('unchanged', 0)} · conflict {o.get('conflict', 0)} · "
          f"quarantined {o.get('quarantined', 0)}")
    print(f"PATENTS   {s['patents']} unique · {s['conflicts']} conflicting versions · "
          f"{s['passages']} passages")

    if s["quarantined"]:
        print("\nQuarantined records:")
        for key, position, publication_id, reason in s["quarantined"]:
            print(f"  {os.path.basename(key)} record {position}"
                  f"{f' ({publication_id})' if publication_id else ''}: {reason}")
    if s["failed"]:
        print("\nFailed jobs:")
        for job_id, key, error in s["failed"]:
            print(f"  job {job_id} {os.path.basename(key)}: {error}")

    if args.jobs:
        print("\n  id  state                 try  records  file")
        for row in conn.execute(
                """SELECT id, state, attempt, coalesce(records_total, 0), source_key, worker
                   FROM ingestion_jobs ORDER BY id"""):
            job_id, state, attempt, total, key, worker = row
            print(f"{job_id:4d}  {state:<20}  {attempt:>3}  {total:>7}  "
                  f"{os.path.basename(key)}  [{worker or '-'}]")


if __name__ == "__main__":
    main()
