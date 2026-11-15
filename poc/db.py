"""Database connection and schema for the proof of concept.

The connection string comes from POC_DATABASE_URL; the default matches
docker-compose.yml. Run `python -m poc.db reset` to start the demo afresh.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import psycopg

DEFAULT_URL = "postgresql://poc:poc@localhost:5433/patents_poc"
SCHEMA = (Path(__file__).with_name("schema.sql")).read_text(encoding="utf-8")


def connect(url: str | None = None) -> psycopg.Connection:
    """Open a connection and make sure the tables exist."""
    url = url or os.environ.get("POC_DATABASE_URL", DEFAULT_URL)
    try:
        conn = psycopg.connect(url, autocommit=True)
    except psycopg.OperationalError as error:
        raise SystemExit(
            f"Cannot reach PostgreSQL at {url}.\n"
            "Start it with `docker compose up -d` (and wait a few seconds), "
            f"or set POC_DATABASE_URL.\nDetails: {error}") from None
    conn.execute(SCHEMA)
    return conn


def reset(conn: psycopg.Connection) -> None:
    """Delete every job, patent and record (the tables are recreated empty)."""
    conn.execute("DROP TABLE IF EXISTS source_records, publication_versions, ingestion_jobs")
    conn.execute(SCHEMA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Proof-of-concept database tools.")
    parser.add_argument("command", choices=["init", "reset"])
    args = parser.parse_args()
    conn = connect()
    if args.command == "reset":
        reset(conn)
        print("All jobs, patents and records deleted; empty tables recreated.")
    else:
        print("Tables are ready.")


if __name__ == "__main__":
    main()
