"""Register patent files as ingestion jobs (the "file arrives" step).

    python -m poc.register data/patent_data_small
    python -m poc.register data/patent_data_small/patents_ipa240215.json poc/fixtures

A job is identified by (file path, SHA-256 of its contents, pipeline version).
Registering the same unchanged file again finds the existing job instead of
creating a new one; a file whose contents changed becomes a new job.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os

import psycopg

from .db import connect

# Bump when parsing or cleaning rules change: every file is then processed again.
PIPELINE_VERSION = "v1"


def file_checksum(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_key(path: str) -> str:
    """A stable name for the file: its path relative to the project, with / separators."""
    return os.path.relpath(os.path.abspath(path)).replace(os.sep, "/")


def expand(arguments: list[str]) -> list[str]:
    """Files, folders (their *.json files) and wildcards, which Windows shells don't expand."""
    paths: list[str] = []
    for argument in arguments:
        if os.path.isdir(argument):
            paths += sorted(glob.glob(os.path.join(argument, "*.json")))
        elif any(ch in argument for ch in "*?["):
            paths += sorted(glob.glob(argument))
        else:
            paths.append(argument)
    return paths


def register(conn: psycopg.Connection, path: str,
             pipeline_version: str = PIPELINE_VERSION) -> tuple[int, bool, str]:
    """Return (job id, created now?, state)."""
    key, checksum = source_key(path), file_checksum(path)
    row = conn.execute(
        """INSERT INTO ingestion_jobs (source_key, checksum, pipeline_version)
           VALUES (%s, %s, %s)
           ON CONFLICT (source_key, checksum, pipeline_version) DO NOTHING
           RETURNING id, state""", (key, checksum, pipeline_version)).fetchone()
    if row:
        return row[0], True, row[1]
    job_id, state = conn.execute(
        """SELECT id, state FROM ingestion_jobs
           WHERE source_key = %s AND checksum = %s AND pipeline_version = %s""",
        (key, checksum, pipeline_version)).fetchone()
    return job_id, False, state


def main() -> None:
    parser = argparse.ArgumentParser(description="Register patent files as ingestion jobs.")
    parser.add_argument("paths", nargs="+", help="Files, folders or wildcards")
    args = parser.parse_args()

    paths = expand(args.paths)
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f"Not found: {', '.join(missing)}")
    if not paths:
        raise SystemExit("No .json files found.")

    conn = connect()
    created = 0
    for path in paths:
        job_id, new, state = register(conn, path)
        created += new
        if not new:
            print(f"  already registered: {source_key(path)} (job {job_id}, {state})")
    print(f"{created} new job(s) queued, {len(paths) - created} already registered.")


if __name__ == "__main__":
    main()
