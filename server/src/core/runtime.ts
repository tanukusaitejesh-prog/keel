import { isTransientDbError, withTx, type Db, type Tx } from "../db.js";
import { toolSpec, type AgentRegistry } from "./define.js";
import type { ProviderResolver } from "./providers/index.js";
import { drain, estimateTokens, reconcile, reserve, type Limits } from "./ratelimit.js";
import { RateLimited, ToolError, TransientModelError, type AgentDef, type Message, type ModelTurn, type ToolCall, type ToolDef, type ToolSpec } from "./types.js";

/**
 * Durable agent runtime.
 *
 * Each run is an append-only log of model turns and tool results in Postgres.
 * A worker claims a run with a lease, rebuilds the conversation from the log,
 * and takes the next step. Because every step is committed atomically with
 * its effects, a worker can die at any instant and another one continues:
 * recorded model turns are replayed rather than re-requested, and committed
 * tool calls never run again.
 *
 * Ownership is fenced: each claim increments lease_token and every write
 * first locks the run row and checks the token, so a stalled worker that lost
 * its lease cannot commit anything after another worker took over.
 */

export type FaultPoint =
  | "model:before-call"
  | "model:before-commit"
  | "model:after-commit"
  | "tool:before-run"
  | "tool:before-commit"
  | "tool:after-commit"
  | "agent:before-commit"
  | "approval:before-commit";

/** Emulates process death in tests: the worker drops the run without releasing its lease. */
export class SimulatedCrash extends Error {}
/** Another worker owns the run now (or it was cancelled). Stop and write nothing. */
export class LeaseLost extends Error {}

export interface Lease {
  runId: string;
  owner: string;
  token: number;
}

export interface RunRow {
  id: string;
  agent: string;
  input: string;
  model: string;
  depth: number;
  parent_id: string | null;
  lease_token: number;
  max_turns: number;
  max_tokens: number;
  input_tokens: number;
  output_tokens: number;
}

export interface RuntimeOptions {
  db: Db;
  agents: AgentRegistry;
  providers: ProviderResolver;
  leaseMs: number;
  limits?: Limits;
  maxDepth?: number;
  /** Test hook: throw SimulatedCrash at a named point. */
  fault?: (point: FaultPoint, lease: Lease) => void;
}

export type RunOutcome = "succeeded" | "failed" | "paused" | "waiting" | "deferred" | "abandoned";

export const TERMINAL = ["succeeded", "failed", "cancelled"] as const;
export const MAX_TRANSIENT_ERRORS = 5;
const DEFAULTS = { maxTurns: 20, maxTokens: 100_000, maxOutputTokens: 4096, maxOutputChars: 6000, maxDepth: 3 };

const RUN_COLUMNS = `id, agent, input, model, depth, parent_id, lease_token, max_turns, max_tokens, input_tokens, output_tokens`;

// ---------------------------------------------------------------------------
// Claiming
// ---------------------------------------------------------------------------

/** Claim the oldest runnable run: queued and due, or running with an expired lease. */
export async function claimRun(db: Db, owner: string, leaseMs: number): Promise<RunRow | null> {
  const { rows } = await db.query<RunRow>(
    `UPDATE runs
        SET status = 'running', lease_owner = $1, lease_token = lease_token + 1,
            recoveries = recoveries + (status = 'running')::int,
            lease_expires_at = now() + make_interval(secs => $2::float / 1000), updated_at = now()
      WHERE id = (
              SELECT id FROM runs
               WHERE (status = 'queued' AND not_before <= now())
                  OR (status = 'running' AND lease_expires_at < now())
               ORDER BY not_before
               FOR UPDATE SKIP LOCKED
               LIMIT 1)
        AND ((status = 'queued' AND not_before <= now()) OR (status = 'running' AND lease_expires_at < now()))
    RETURNING ${RUN_COLUMNS}`,
    [owner, leaseMs],
  );
  return rows[0] ?? null;
}

/** Lock the run row and prove we still hold the lease. First statement of every write. */
async function fence(tx: Tx, lease: Lease): Promise<void> {
  const res = await tx.query(
    `SELECT 1 FROM runs WHERE id = $1 AND lease_owner = $2 AND lease_token = $3 AND status = 'running' FOR UPDATE`,
    [lease.runId, lease.owner, lease.token],
  );
  if (res.rowCount !== 1) throw new LeaseLost(`lease on ${lease.runId} lost`);
}

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------

export interface StepRow {
  seq: number;
  kind: "model" | "tool";
  call_id: string | null;
  name: string | null;
  payload: any;
}

export interface ReplayState {
  messages: Message[];
  nextSeq: number;
  turns: number;
  last: ModelTurn | null;
  /** Tool calls of the last turn that have no recorded result yet, in order. */
  pending: ToolCall[];
}

export function replay(input: string, steps: StepRow[]): ReplayState {
  const messages: Message[] = [{ role: "user", content: input }];
  let last: ModelTurn | null = null;
  let turns = 0;
  const answered = new Set<string>();
  for (const s of steps) {
    if (s.kind === "model") {
      last = s.payload.turn as ModelTurn;
      messages.push({ role: "assistant", content: last.text, toolCalls: last.toolCalls, provider: last.provider, raw: last.raw });
      turns++;
      answered.clear();
    } else {
      messages.push({ role: "tool", callId: s.call_id!, name: s.name!, content: s.payload.content, isError: s.payload.isError });
      answered.add(s.call_id!);
    }
  }
  const pending = last?.stop === "tool_calls" ? last.toolCalls.filter((c) => !answered.has(c.id)) : [];
  return { messages, nextSeq: steps.length, turns, last, pending };
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

const specCache = new WeakMap<AgentDef, ToolSpec[]>();
function specsFor(agent: AgentDef): ToolSpec[] {
  let s = specCache.get(agent);
  if (!s) specCache.set(agent, (s = agent.tools.map(toolSpec)));
  return s;
}

export async function executeRun(run: RunRow, owner: string, opts: RuntimeOptions): Promise<RunOutcome> {
  const lease: Lease = { runId: run.id, owner, token: run.lease_token };
  const fault = (p: FaultPoint) => opts.fault?.(p, lease);
  const abort = new AbortController();
  let lost = false;
  const heartbeat = setInterval(async () => {
    try {
      const r = await opts.db.query(
        `UPDATE runs SET lease_expires_at = now() + make_interval(secs => $4::float / 1000)
          WHERE id = $1 AND lease_owner = $2 AND lease_token = $3 AND status = 'running'`,
        [lease.runId, lease.owner, lease.token, opts.leaseMs],
      );
      if (r.rowCount !== 1) {
        lost = true;
        abort.abort();
      }
    } catch {
      /* transient: the next beat or the fence decides */
    }
  }, Math.max(50, Math.floor(opts.leaseMs / 3)));

  let tokensUsed = run.input_tokens + run.output_tokens;
  try {
    const agent = opts.agents.get(run.agent);
    for (;;) {
      if (lost) throw new LeaseLost(`lease on ${run.id} lost`);
      const { rows } = await opts.db.query<StepRow>(
        `SELECT seq, kind, call_id, name, payload FROM steps WHERE run_id = $1 ORDER BY seq`,
        [run.id],
      );
      const st = replay(run.input, rows);

      // 1. Unanswered tool calls from the last turn.
      if (st.pending.length) {
        const r = await handleCall(st.pending[0]!, st, run, agent, lease, opts, fault, abort.signal);
        if (r !== "done") return r;
        continue;
      }

      // 2. The model has finished, or stopped in a way we cannot continue.
      const last = st.last;
      if (last && last.stop !== "tool_calls") {
        if (last.stop === "end") {
          await finish(opts.db, lease, run, "succeeded", { result: last.text.trim() || "(empty answer)" });
          return "succeeded";
        }
        const why = last.stop === "length" ? "model output was truncated (max output tokens)" : last.stop === "refusal" ? "model declined the request" : "model stopped unexpectedly";
        await finish(opts.db, lease, run, "failed", { error: why });
        return "failed";
      }

      // 3. Ask the model for the next turn, within budget and rate limits.
      if (st.turns >= run.max_turns) {
        await finish(opts.db, lease, run, "failed", { error: `turn budget exhausted (${run.max_turns} model turns)` });
        return "failed";
      }
      if (tokensUsed >= run.max_tokens) {
        await finish(opts.db, lease, run, "failed", { error: `token budget exhausted (${tokensUsed} of ${run.max_tokens})` });
        return "failed";
      }
      const provider = opts.providers(run.model);
      const req = {
        system: agent.system,
        messages: st.messages,
        tools: specsFor(agent),
        maxOutputTokens: agent.maxOutputTokens ?? DEFAULTS.maxOutputTokens,
        reasoningEffort: agent.reasoningEffort,
        signal: abort.signal,
      };
      const tpm = opts.limits?.tpm(run.model);
      const estimate = estimateTokens([req.system, req.messages, req.tools], req.maxOutputTokens);
      let reserved = 0;
      if (tpm) {
        const r = await reserve(opts.db, run.model, tpm, estimate);
        if (r.waitMs > 0) {
          await park(opts.db, lease, r.waitMs);
          return "deferred";
        }
        reserved = r.reserved;
      }
      fault("model:before-call");
      const t0 = performance.now();
      let turn: ModelTurn;
      try {
        turn = await provider.complete(req);
      } catch (err) {
        if (err instanceof RateLimited) {
          if (tpm) await drain(opts.db, run.model);
          await park(opts.db, lease, err.retryAfterMs);
          return "deferred";
        }
        throw err;
      }
      const latency = Math.round(performance.now() - t0);
      if (tpm) await reconcile(opts.db, run.model, turn.usage.input + turn.usage.output - reserved);
      await withTx(opts.db, async (tx) => {
        await fence(tx, lease);
        await tx.query(`INSERT INTO steps (run_id, seq, kind, payload, latency_ms) VALUES ($1, $2, 'model', $3, $4)`, [
          run.id,
          st.nextSeq,
          JSON.stringify({ turn }),
          latency,
        ]);
        await tx.query(
          `UPDATE runs SET input_tokens = input_tokens + $2, output_tokens = output_tokens + $3, updated_at = now() WHERE id = $1`,
          [run.id, turn.usage.input, turn.usage.output],
        );
        fault("model:before-commit");
      });
      tokensUsed += turn.usage.input + turn.usage.output;
      fault("model:after-commit");
    }
  } catch (err) {
    if (err instanceof SimulatedCrash || err instanceof LeaseLost || lost) return "abandoned";
    const message = (err instanceof Error ? err.message : String(err)).slice(0, 2000);
    try {
      if (isTransientDbError(err) || err instanceof TransientModelError) {
        if (await requeue(opts.db, lease, message)) return "deferred";
      }
      await finish(opts.db, lease, run, "failed", { error: message });
      return "failed";
    } catch (inner) {
      if (inner instanceof LeaseLost || isTransientDbError(inner)) return "abandoned"; // lease expiry retries it
      throw inner;
    }
  } finally {
    clearInterval(heartbeat);
  }
}

async function handleCall(
  call: ToolCall,
  st: ReplayState,
  run: RunRow,
  agent: AgentDef,
  lease: Lease,
  opts: RuntimeOptions,
  fault: (p: FaultPoint) => void,
  signal: AbortSignal,
): Promise<"done" | "paused" | "waiting"> {
  const db = opts.db;
  const seq = st.nextSeq;
  const record = async (tx: Tx, content: string, isError: boolean, latency?: number) => {
    await tx.query(
      `INSERT INTO steps (run_id, seq, kind, call_id, name, payload, latency_ms) VALUES ($1, $2, 'tool', $3, $4, $5, $6)`,
      [run.id, seq, call.id, call.name, JSON.stringify({ args: call.args, content, isError }), latency ?? null],
    );
  };
  const recordError = async (content: string) => {
    await withTx(db, async (tx) => {
      await fence(tx, lease);
      await record(tx, content, true);
      fault("tool:before-commit");
    });
    fault("tool:after-commit");
    return "done" as const;
  };

  const tool = agent.tools.find((t) => t.name === call.name);
  if (!tool) return recordError(`Unknown tool "${call.name}". Available: ${agent.tools.map((t) => t.name).join(", ")}.`);
  const args = call.args as Record<string, unknown> | null;
  if (args && typeof args === "object" && "__invalid_json" in args) return recordError("Your arguments were not valid JSON. Call the tool again with a valid JSON object.");
  const parsed = tool.input.safeParse(call.args ?? {});
  if (!parsed.success) {
    return recordError(`Invalid arguments: ${parsed.error.issues.map((i) => `${i.path.join(".") || "(root)"}: ${i.message}`).join("; ")}`);
  }
  const input = parsed.data;

  if (tool.effect === "agent") return handleAgentCall(call, st, run, agent, lease, opts, fault, record);

  // Human approval gate.
  const needs = typeof tool.needsApproval === "function" ? tool.needsApproval(input) : !!tool.needsApproval;
  if (needs) {
    const { rows } = await db.query<{ status: string; note: string | null }>(`SELECT status, note FROM approvals WHERE run_id = $1 AND call_id = $2`, [run.id, call.id]);
    const decision = rows[0];
    if (!decision || decision.status === "pending") {
      await withTx(db, async (tx) => {
        await fence(tx, lease);
        await tx.query(`INSERT INTO approvals (run_id, call_id, tool_name, input) VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING`, [
          run.id,
          call.id,
          tool.name,
          JSON.stringify(input),
        ]);
        await tx.query(`UPDATE runs SET status = 'awaiting_approval', lease_owner = NULL, lease_expires_at = NULL, updated_at = now() WHERE id = $1`, [run.id]);
        fault("approval:before-commit");
      });
      return "paused";
    }
    if (decision.status === "rejected") return recordError(`A human reviewer rejected this call.${decision.note ? ` Their note: ${decision.note}` : ""}`);
  }

  const ctx = { runId: run.id, callId: call.id, idempotencyKey: `${run.id}:${call.id}`, db, signal };
  const limit = tool.maxOutputChars ?? DEFAULTS.maxOutputChars;
  const t0 = performance.now();

  if (tool.effect === "write") {
    // Side effects and the step record commit together: exactly once.
    await withTx(db, async (tx) => {
      await fence(tx, lease);
      await tx.query("SAVEPOINT tool_call");
      let content: string;
      let isError = false;
      try {
        fault("tool:before-run");
        content = format(await tool.run!({ ...ctx, tx }, input), limit);
        await tx.query("RELEASE SAVEPOINT tool_call");
      } catch (err) {
        if (!isToolFailure(err)) throw err;
        await tx.query("ROLLBACK TO SAVEPOINT tool_call");
        content = failureMessage(err);
        isError = true;
      }
      await record(tx, content, isError, Math.round(performance.now() - t0));
      fault("tool:before-commit");
    });
    fault("tool:after-commit");
    return "done";
  }

  if (tool.effect === "external") {
    // Log the intent first; a retry after a crash reuses the same idempotency key.
    await withTx(db, async (tx) => {
      await fence(tx, lease);
      await tx.query(
        `INSERT INTO external_calls (idempotency_key, run_id, tool_name) VALUES ($1, $2, $3)
         ON CONFLICT (idempotency_key) DO UPDATE SET attempts = external_calls.attempts + 1, last_attempt_at = now()`,
        [ctx.idempotencyKey, run.id, tool.name],
      );
    });
  }

  fault("tool:before-run");
  let content: string;
  let isError = false;
  try {
    content = format(await tool.run!(ctx, input), limit);
  } catch (err) {
    if (!isToolFailure(err)) throw err;
    content = failureMessage(err);
    isError = true;
  }
  const latency = Math.round(performance.now() - t0);
  await withTx(db, async (tx) => {
    await fence(tx, lease);
    await record(tx, content, isError, latency);
    fault("tool:before-commit");
  });
  fault("tool:after-commit");
  return "done";
}

/**
 * Delegation to a sub-agent. All agent calls of the turn are spawned together
 * (parallel fan-out) in one transaction, then the parent releases its worker
 * and sleeps until a child finishes and wakes it.
 */
async function handleAgentCall(
  call: ToolCall,
  st: ReplayState,
  run: RunRow,
  agent: AgentDef,
  lease: Lease,
  opts: RuntimeOptions,
  fault: (p: FaultPoint) => void,
  record: (tx: Tx, content: string, isError: boolean, latency?: number) => Promise<void>,
): Promise<"done" | "waiting"> {
  const maxDepth = opts.maxDepth ?? DEFAULTS.maxDepth;
  const outcome = await withTx(opts.db, async (tx) => {
    await fence(tx, lease);
    if (run.depth + 1 > maxDepth) {
      await record(tx, `Cannot delegate: maximum agent nesting depth (${maxDepth}) reached. Do the work yourself.`, true);
      return "done" as const;
    }
    for (const c of st.pending) {
      const t = agent.tools.find((x) => x.name === c.name);
      if (t?.effect !== "agent") continue;
      const parsed = t.input.safeParse(c.args ?? {});
      if (!parsed.success) continue; // reported when this call comes up
      const child = opts.agents.get(t.agent!);
      const task = t.toTask ? t.toTask(parsed.data) : JSON.stringify(parsed.data);
      await tx.query(
        `INSERT INTO runs (agent, input, model, parent_id, parent_call_id, depth, max_turns, max_tokens)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
         ON CONFLICT (parent_id, parent_call_id) WHERE parent_id IS NOT NULL DO NOTHING`,
        [child.name, task, child.model, run.id, c.id, run.depth + 1, child.maxTurns ?? DEFAULTS.maxTurns, child.maxTokens ?? DEFAULTS.maxTokens],
      );
    }
    const { rows } = await tx.query<{ status: string; result: string | null; error: string | null; id: string }>(
      `SELECT id, status, result, error FROM runs WHERE parent_id = $1 AND parent_call_id = $2`,
      [run.id, call.id],
    );
    const child = rows[0]!;
    if ((TERMINAL as readonly string[]).includes(child.status)) {
      const ok = child.status === "succeeded";
      await record(tx, ok ? child.result ?? "" : `Sub-agent ${child.status}: ${child.error ?? "no details"}`, !ok);
      fault("tool:before-commit");
      return "done" as const;
    }
    await tx.query(`UPDATE runs SET status = 'waiting', lease_owner = NULL, lease_expires_at = NULL, updated_at = now() WHERE id = $1`, [run.id]);
    fault("agent:before-commit");
    return "waiting" as const;
  });
  if (outcome === "done") fault("tool:after-commit");
  return outcome;
}

function isToolFailure(err: unknown): boolean {
  if (err instanceof SimulatedCrash || err instanceof LeaseLost) return false;
  if (isTransientDbError(err)) return false;
  return err instanceof Error;
}

function failureMessage(err: unknown): string {
  if (err instanceof ToolError) return err.message;
  return `Tool failed: ${err instanceof Error ? err.message : String(err)}`;
}

function format(out: unknown, limit: number): string {
  const s = typeof out === "string" ? out : JSON.stringify(out);
  return s.length > limit ? `${s.slice(0, limit)}\n...[truncated ${s.length - limit} characters]` : s;
}

async function finish(db: Db, lease: Lease, run: RunRow, status: "succeeded" | "failed", out: { result?: string; error?: string }) {
  await withTx(db, async (tx) => {
    await fence(tx, lease);
    await tx.query(
      `UPDATE runs SET status = $2, result = $3, error = $4, lease_owner = NULL, lease_expires_at = NULL,
              updated_at = now(), finished_at = now() WHERE id = $1`,
      [run.id, status, out.result ?? null, out.error ?? null],
    );
    if (run.parent_id) await wakeParent(tx, run.parent_id);
  });
}

async function wakeParent(tx: Tx, parentId: string) {
  await tx.query(`UPDATE runs SET status = 'queued', not_before = now(), updated_at = now() WHERE id = $1 AND status = 'waiting'`, [parentId]);
}

/** Give the run back to the queue until `waitMs` from now (rate limit). */
async function park(db: Db, lease: Lease, waitMs: number) {
  await withTx(db, async (tx) => {
    await fence(tx, lease);
    await tx.query(
      `UPDATE runs SET status = 'queued', not_before = now() + make_interval(secs => $2::float / 1000),
              lease_owner = NULL, lease_expires_at = NULL, rate_limit_waits = rate_limit_waits + 1, updated_at = now()
        WHERE id = $1`,
      [lease.runId, Math.max(250, waitMs)],
    );
  });
}

/** Re-queue after a transient failure with backoff, unless it has failed transiently too often. */
async function requeue(db: Db, lease: Lease, message: string): Promise<boolean> {
  return withTx(db, async (tx) => {
    await fence(tx, lease);
    const res = await tx.query(
      `UPDATE runs SET status = 'queued', lease_owner = NULL, lease_expires_at = NULL,
              not_before = now() + make_interval(secs => least(30, 0.5 * power(2, transient_errors))),
              transient_errors = transient_errors + 1, error = $2, updated_at = now()
        WHERE id = $1 AND transient_errors < $3`,
      [lease.runId, `retrying after: ${message}`, MAX_TRANSIENT_ERRORS],
    );
    return res.rowCount === 1;
  });
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

export async function startRun(db: Db, agents: AgentRegistry, args: { agent: string; input: string; model?: string }): Promise<string> {
  const a = agents.get(args.agent);
  const { rows } = await db.query<{ id: string }>(
    `INSERT INTO runs (agent, input, model, max_turns, max_tokens) VALUES ($1, $2, $3, $4, $5) RETURNING id`,
    [a.name, args.input, args.model ?? a.model, a.maxTurns ?? DEFAULTS.maxTurns, a.maxTokens ?? DEFAULTS.maxTokens],
  );
  return rows[0]!.id;
}

export class ConflictError extends Error {}

export async function decide(db: Db, runId: string, callId: string, decision: "approved" | "rejected", note?: string): Promise<void> {
  await withTx(db, async (tx) => {
    const res = await tx.query(
      `UPDATE approvals SET status = $3, note = $4, decided_at = now() WHERE run_id = $1 AND call_id = $2 AND status = 'pending'`,
      [runId, callId, decision, note ?? null],
    );
    if (res.rowCount !== 1) throw new ConflictError("no pending approval with that id");
    await tx.query(`UPDATE runs SET status = 'queued', not_before = now(), updated_at = now() WHERE id = $1 AND status = 'awaiting_approval'`, [runId]);
  });
}

/** Cancel a run and every descendant. Workers holding them lose the fence and stop. */
export async function cancelRun(db: Db, runId: string): Promise<number> {
  return withTx(db, async (tx) => {
    const { rows } = await tx.query<{ id: string; parent_id: string | null }>(
      `WITH RECURSIVE tree AS (
         SELECT id FROM runs WHERE id = $1
         UNION ALL SELECT r.id FROM runs r JOIN tree t ON r.parent_id = t.id)
       UPDATE runs SET status = 'cancelled', lease_owner = NULL, lease_expires_at = NULL, error = coalesce(error, 'cancelled'),
              updated_at = now(), finished_at = now()
        WHERE id IN (SELECT id FROM tree) AND status NOT IN ('succeeded', 'failed', 'cancelled')
       RETURNING id, parent_id`,
      [runId],
    );
    const root = rows.find((r) => r.id === runId);
    if (root?.parent_id) await wakeParent(tx, root.parent_id);
    return rows.length;
  });
}

/**
 * Time travel: start a new run whose log is a copy of `runId` up to (not
 * including) model turn `atSeq`, optionally on a different model. Recorded
 * tool results are reused, so no side effect before that point runs again.
 */
export async function forkRun(db: Db, runId: string, atSeq: number, opts: { model?: string } = {}): Promise<string> {
  return withTx(db, async (tx) => {
    const src = (await tx.query(`SELECT * FROM runs WHERE id = $1`, [runId])).rows[0];
    if (!src) throw new ConflictError("run not found");
    const at = (await tx.query(`SELECT kind FROM steps WHERE run_id = $1 AND seq = $2`, [runId, atSeq])).rows[0];
    if (at?.kind !== "model") throw new ConflictError("can only fork at a model turn");
    const { rows } = await tx.query<{ id: string }>(
      `INSERT INTO runs (agent, input, model, max_turns, max_tokens, forked_from, forked_at_seq)
       VALUES ($1, $2, $3, $4, $5, $6, $7) RETURNING id`,
      [src.agent, src.input, opts.model ?? src.model, src.max_turns, src.max_tokens, runId, atSeq],
    );
    const id = rows[0]!.id;
    await tx.query(
      `INSERT INTO steps (run_id, seq, kind, call_id, name, payload, latency_ms)
       SELECT $1, seq, kind, call_id, name, payload, latency_ms FROM steps WHERE run_id = $2 AND seq < $3`,
      [id, runId, atSeq],
    );
    return id;
  });
}
