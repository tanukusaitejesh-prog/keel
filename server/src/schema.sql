-- Keel runtime schema. Applied by migrate() only when this file changes.
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS runs (
  id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  agent             text NOT NULL,
  input             text NOT NULL,
  model             text NOT NULL,              -- e.g. groq:openai/gpt-oss-120b
  status            text NOT NULL DEFAULT 'queued'
                    CHECK (status IN ('queued', 'running', 'awaiting_approval', 'waiting', 'succeeded', 'failed', 'cancelled')),
  -- multi-agent: a child run answers one tool call of its parent
  parent_id         uuid REFERENCES runs(id) ON DELETE CASCADE,
  parent_call_id    text,
  depth             int NOT NULL DEFAULT 0,
  -- time-travel: this run replays another run's log up to a turn
  forked_from       uuid REFERENCES runs(id) ON DELETE SET NULL,
  forked_at_seq     int,
  -- scheduling and ownership
  not_before        timestamptz NOT NULL DEFAULT now(),
  lease_owner       text,
  lease_token       int NOT NULL DEFAULT 0,
  lease_expires_at  timestamptz,
  -- budgets and accounting
  max_turns         int NOT NULL,
  max_tokens        int NOT NULL,
  input_tokens      int NOT NULL DEFAULT 0,
  output_tokens     int NOT NULL DEFAULT 0,
  transient_errors  int NOT NULL DEFAULT 0,
  rate_limit_waits  int NOT NULL DEFAULT 0,
  result            text,
  error             text,
  created_at        timestamptz NOT NULL DEFAULT now(),
  updated_at        timestamptz NOT NULL DEFAULT now(),
  finished_at       timestamptz,
  CHECK ((parent_id IS NULL) = (parent_call_id IS NULL))
);
CREATE INDEX IF NOT EXISTS runs_claimable_idx ON runs (not_before) WHERE status IN ('queued', 'running');
CREATE UNIQUE INDEX IF NOT EXISTS runs_one_child_per_call ON runs (parent_id, parent_call_id) WHERE parent_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS runs_parent_idx ON runs (parent_id);

-- Append-only event log; the conversation is rebuilt from it on every resume.
CREATE TABLE IF NOT EXISTS steps (
  run_id      uuid NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  seq         int NOT NULL,
  kind        text NOT NULL CHECK (kind IN ('model', 'tool')),
  call_id     text,
  name        text,
  payload     jsonb NOT NULL,
  latency_ms  int,
  created_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (run_id, seq)
);
CREATE UNIQUE INDEX IF NOT EXISTS steps_tool_once ON steps (run_id, call_id) WHERE kind = 'tool';

CREATE TABLE IF NOT EXISTS approvals (
  run_id      uuid NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  call_id     text NOT NULL,
  tool_name   text NOT NULL,
  input       jsonb NOT NULL,
  status      text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
  note        text,
  created_at  timestamptz NOT NULL DEFAULT now(),
  decided_at  timestamptz,
  PRIMARY KEY (run_id, call_id)
);

-- Intent log for external (non-transactional) tools: one row per tool call,
-- attempts > 1 means the call was retried after a crash with the same key.
CREATE TABLE IF NOT EXISTS external_calls (
  idempotency_key  text PRIMARY KEY,
  run_id           uuid NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  tool_name        text NOT NULL,
  attempts         int NOT NULL DEFAULT 1,
  first_attempt_at timestamptz NOT NULL DEFAULT now(),
  last_attempt_at  timestamptz NOT NULL DEFAULT now()
);

-- Transactional outbox for write tools that notify the outside world.
CREATE TABLE IF NOT EXISTS outbox (
  id          bigserial PRIMARY KEY,
  run_id      uuid NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  call_id     text NOT NULL,
  channel     text NOT NULL,
  payload     jsonb NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now(),
  sent_at     timestamptz,
  UNIQUE (run_id, call_id)
);

-- Token buckets shared by every worker (one per provider:model).
CREATE TABLE IF NOT EXISTS rate_buckets (
  key             text PRIMARY KEY,
  capacity        double precision NOT NULL,
  tokens          double precision NOT NULL,
  refill_per_sec  double precision NOT NULL,
  updated_at      timestamptz NOT NULL DEFAULT clock_timestamp()
);

-- Live updates (LISTEN keel_events).
CREATE OR REPLACE FUNCTION keel_notify_step() RETURNS trigger AS $$
BEGIN PERFORM pg_notify('keel_events', NEW.run_id::text); RETURN NEW; END $$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION keel_notify_run() RETURNS trigger AS $$
BEGIN PERFORM pg_notify('keel_events', NEW.id::text); RETURN NEW; END $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS steps_notify ON steps;
CREATE TRIGGER steps_notify AFTER INSERT ON steps FOR EACH ROW EXECUTE FUNCTION keel_notify_step();
DROP TRIGGER IF EXISTS approvals_notify ON approvals;
CREATE TRIGGER approvals_notify AFTER INSERT OR UPDATE ON approvals FOR EACH ROW EXECUTE FUNCTION keel_notify_step();
DROP TRIGGER IF EXISTS runs_notify ON runs;
CREATE TRIGGER runs_notify AFTER INSERT OR UPDATE OF status ON runs FOR EACH ROW EXECUTE FUNCTION keel_notify_run();
