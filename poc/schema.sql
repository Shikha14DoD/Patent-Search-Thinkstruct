-- Part 2 proof of concept: the ingestion records from docs/PART2_DESIGN.md, section 5.2.
-- Safe to run repeatedly.

CREATE TABLE IF NOT EXISTS ingestion_jobs (
  id                  bigserial PRIMARY KEY,
  source_key          text NOT NULL,
  checksum            text NOT NULL,
  pipeline_version    text NOT NULL,
  state               text NOT NULL DEFAULT 'queued'
                      CHECK (state IN ('queued', 'processing', 'retry_wait', 'complete',
                                       'complete_with_errors', 'failed')),
  attempt             int  NOT NULL DEFAULT 0,
  max_attempts        int  NOT NULL DEFAULT 3,
  next_attempt_at     timestamptz NOT NULL DEFAULT now(),
  lease_until         timestamptz,
  lease_token         uuid,
  worker              text,
  records_total       int,
  records_saved       int,
  records_unchanged   int,
  records_conflict    int,
  records_quarantined int,
  last_error          text,
  created_at          timestamptz NOT NULL DEFAULT now(),
  started_at          timestamptz,
  finished_at         timestamptz,
  UNIQUE (source_key, checksum, pipeline_version)
);

CREATE TABLE IF NOT EXISTS publication_versions (
  publication_id   text NOT NULL,
  content_hash     text NOT NULL,
  jurisdiction     text NOT NULL DEFAULT 'US',
  language         text NOT NULL DEFAULT 'unknown',
  title            text,
  classification   text,
  publication_date date,
  passages         int  NOT NULL,
  job_id           bigint NOT NULL REFERENCES ingestion_jobs(id),
  state            text NOT NULL CHECK (state IN ('saved', 'conflict')),
  saved_at         timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (publication_id, content_hash)
);

CREATE TABLE IF NOT EXISTS source_records (
  job_id           bigint NOT NULL REFERENCES ingestion_jobs(id),
  record_position  int  NOT NULL,
  publication_id   text,
  outcome          text NOT NULL CHECK (outcome IN ('saved', 'unchanged', 'conflict',
                                                    'quarantined')),
  error_reason     text,
  PRIMARY KEY (job_id, record_position)
);
