import { z } from "zod";
import { createPool, migrate, type Db } from "../src/db.js";
import { AgentRegistry, agentTool, defineAgent, defineTool } from "../src/core/define.js";
import { conv, envProviders, ScriptedProvider, type Policy } from "../src/core/providers/index.js";
import { decide, TERMINAL, type RuntimeOptions } from "../src/core/runtime.js";
import { startWorker } from "../src/core/worker.js";
import { ToolError } from "../src/core/types.js";

export const TEST_DB_URL = process.env.TEST_DATABASE_URL ?? "postgres://postgres@localhost:55432/keel_test";

export async function freshDb(): Promise<Db> {
  const db = createPool(TEST_DB_URL);
  await migrate(db);
  await db.query(`CREATE TABLE IF NOT EXISTS test_ledger (run_id uuid NOT NULL, call_id text NOT NULL, amount int NOT NULL, PRIMARY KEY (run_id, call_id))`);
  return db;
}

export async function reset(db: Db) {
  await db.query(`TRUNCATE runs, rate_buckets, test_ledger CASCADE`);
  remote.calls.clear();
}

/** A fake remote service that deduplicates by idempotency key, like Stripe or SES would. */
export const remote = {
  calls: new Map<string, number>(),
  /** Replaceable: the chaos script records deliveries in Postgres so all processes share them. */
  async record(key: string): Promise<void> {
    this.calls.set(key, (this.calls.get(key) ?? 0) + 1);
  },
};

// ---------------------------------------------------------------------------
// Test agents
// ---------------------------------------------------------------------------
const lookup = defineTool({
  name: "lookup",
  description: "read a value",
  effect: "read",
  input: z.object({ key: z.string() }),
  async run(_ctx, { key }) {
    return { key, value: 42 };
  },
});
const addEntry = defineTool({
  name: "add_entry",
  description: "write a ledger entry",
  effect: "write",
  needsApproval: (i: { amount: number }) => i.amount > 100,
  input: z.object({ amount: z.number().int() }),
  async run({ tx, runId, callId }, { amount }) {
    if (amount < 0) throw new ToolError("amount must be positive");
    await tx!.query(`INSERT INTO test_ledger (run_id, call_id, amount) VALUES ($1, $2, $3)`, [runId, callId, amount]);
    return { ok: true, amount };
  },
});
const notify = defineTool({
  name: "notify",
  description: "call a remote service",
  effect: "external",
  input: z.object({ message: z.string() }),
  async run({ idempotencyKey }, { message }) {
    await remote.record(idempotencyKey);
    return { delivered: message };
  },
});

export const policies: Record<string, Policy> = {
  ledger: (m) => {
    const t = conv.turns(m);
    if (t === 0) return { text: "checking", calls: [{ name: "lookup", args: { key: "balance" } }] };
    if (t === 1)
      return {
        calls: [
          { name: "add_entry", args: { amount: 10 } },
          { name: "add_entry", args: { amount: 20 } },
          { name: "notify", args: { message: "hi" } },
        ],
      };
    if (t === 2) return { calls: [{ name: "add_entry", args: { amount: 150 } }] }; // needs approval
    return { text: `done: ${conv.toolResults(m).map((r) => (r.isError ? "E" : "ok")).join(",")}` };
  },
  boss: (m) => {
    if (conv.turns(m) === 0) return { calls: ["a", "b", "c"].map((task) => ({ name: "delegate", args: { task } })) };
    return { text: `combined: ${conv.toolResults(m).map((r) => r.content).join(" + ")}` };
  },
  helper: (m) => {
    const task = (m[0] as { content: string }).content;
    if (task === "fail") return { calls: [{ name: "add_entry", args: { amount: 0.5 } }] }; // invalid: not an int
    if (conv.turns(m) === 0) return { calls: [{ name: "add_entry", args: { amount: 1 } }] };
    if (task === "fail2") return { text: "" };
    return { text: `did ${task}` };
  },
  recursive: (m) => (conv.turns(m) === 0 ? { calls: [{ name: "delegate", args: { task: "deeper" } }] } : { text: `depth result: ${conv.toolResults(m)[0]?.content}` }),
};

export const agents = () =>
  new AgentRegistry([
    defineAgent({ name: "ledger", description: "", model: "scripted:ledger", system: "s", tools: [lookup, addEntry, notify] }),
    defineAgent({ name: "boss", description: "", model: "scripted:boss", system: "s", tools: [agentTool({ name: "delegate", description: "d", agent: "helper" })] }),
    defineAgent({ name: "helper", description: "", model: "scripted:helper", system: "s", tools: [addEntry] }),
    defineAgent({ name: "recursive", description: "", model: "scripted:recursive", system: "s", tools: [agentTool({ name: "delegate", description: "d", agent: "recursive" })] }),
  ]);

export const scripted = (latencyMs = 0) => envProviders({ scripted: (model) => new ScriptedProvider(model, policies[model]!, latencyMs) });

export type Approver = (a: { run_id: string; call_id: string; input: any }) => { decision: "approved" | "rejected"; note?: string };

/** Run workers until every run (and descendant) in `ids` is terminal, approving like a human would. */
export async function settle(
  db: Db,
  ids: string[],
  opts: Partial<RuntimeOptions> & { approver?: Approver; workers?: number; concurrency?: number; timeoutMs?: number } = {},
): Promise<void> {
  const base: RuntimeOptions = { db, agents: opts.agents ?? agents(), providers: opts.providers ?? scripted(), leaseMs: opts.leaseMs ?? 5000, limits: opts.limits, fault: opts.fault, maxDepth: opts.maxDepth };
  const workers = Array.from({ length: opts.workers ?? 1 }, (_, i) => startWorker({ ...base, concurrency: opts.concurrency ?? 1, pollMs: 15, owner: `w${i}` }));
  const deadline = Date.now() + (opts.timeoutMs ?? 60_000);
  try {
    for (;;) {
      const pending = await db.query(`SELECT run_id, call_id, input FROM approvals WHERE status = 'pending'`);
      for (const a of pending.rows) {
        const approver: Approver = opts.approver ?? (() => ({ decision: "approved" }));
        const d = approver(a);
        await decide(db, a.run_id, a.call_id, d.decision, d.note).catch(() => {});
      }
      const open = await db.query(
        `WITH RECURSIVE tree AS (SELECT id FROM runs WHERE id = ANY($1) UNION ALL SELECT r.id FROM runs r JOIN tree t ON r.parent_id = t.id)
         SELECT count(*)::int AS n FROM runs WHERE id IN (SELECT id FROM tree) AND status <> ALL($2)`,
        [ids, TERMINAL],
      );
      if (open.rows[0].n === 0) return;
      if (Date.now() > deadline) {
        const stuck = await db.query(`SELECT agent, status, not_before > now() AS parked, lease_owner, lease_token, transient_errors, left(error, 200) AS error FROM runs WHERE status <> ALL($1)`, [TERMINAL]);
        throw new Error(`timed out with ${open.rows[0].n} run(s) open: ${JSON.stringify(stuck.rows)}`);
      }
      await new Promise((r) => setTimeout(r, 15));
    }
  } finally {
    await Promise.all(workers.map((w) => w.stop()));
  }
}

export async function runRow(db: Db, id: string) {
  return (await db.query(`SELECT * FROM runs WHERE id = $1`, [id])).rows[0];
}
export async function stepsOf(db: Db, id: string) {
  return (await db.query(`SELECT seq, kind, call_id, name, payload FROM steps WHERE run_id = $1 ORDER BY seq`, [id])).rows;
}
