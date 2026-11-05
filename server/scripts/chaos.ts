/**
 * Chaos test with real processes. Several worker processes run a mix of
 * single-agent runs (read, write, external and approval-gated tools) and
 * multi-agent runs (a parent fanning out to three sub-agents), while this
 * script SIGKILLs a random worker every `--kill-every` ms and starts a new one.
 * Afterwards it checks every invariant the runtime promises.
 *
 *   TEST_DATABASE_URL=... npm run chaos -w server -- --runs 100 --workers 4
 */
import { spawn, type ChildProcess } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { decide, startRun, TERMINAL } from "../src/core/runtime.js";
import { agents, freshDb, reset } from "../test/helpers.js";

const arg = (n: string, d: number) => {
  const i = process.argv.indexOf(`--${n}`);
  return i > 0 ? Number(process.argv[i + 1]) : d;
};
const RUNS = arg("runs", 100);
const WORKERS = arg("workers", 4);
const KILL_EVERY = arg("kill-every", 500);
const LATENCY = arg("latency", 200);
const TIMEOUT = arg("timeout", 600_000);

const db = await freshDb();
await reset(db);
await db.query(`CREATE TABLE IF NOT EXISTS test_remote (key text PRIMARY KEY, deliveries int NOT NULL)`);
await db.query(`TRUNCATE test_remote`);
const reg = agents();
const ids: string[] = [];
for (let i = 0; i < RUNS; i++) ids.push(await startRun(db, reg, { agent: i % 2 ? "boss" : "ledger", input: "go" }));
const ledgers = Math.ceil(RUNS / 2);
const bosses = RUNS - ledgers;

const entry = path.join(path.dirname(fileURLToPath(import.meta.url)), "chaos-worker.ts");
const procs: ChildProcess[] = [];
let started = 0;
const spawnWorker = (slot: number) => {
  procs[slot] = spawn(process.execPath, ["--import", "tsx", entry], {
    env: { ...process.env, LATENCY_MS: String(LATENCY), LEASE_MS: "1500", PG_POOL_MAX: "8" },
    stdio: ["ignore", "ignore", "inherit"],
  });
  started++;
};
for (let i = 0; i < WORKERS; i++) spawnWorker(i);
let kills = 0;
let seed = 4242;
const rand = () => (seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32;
const killer = setInterval(() => {
  const slot = Math.floor(rand() * WORKERS);
  procs[slot]?.kill("SIGKILL");
  kills++;
  spawnWorker(slot);
}, KILL_EVERY);

const t0 = Date.now();
let open = Infinity;
while (open > 0 && Date.now() - t0 < TIMEOUT) {
  for (const a of (await db.query(`SELECT run_id, call_id FROM approvals WHERE status = 'pending'`)).rows) {
    await decide(db, a.run_id, a.call_id, "approved").catch(() => {});
  }
  open = (await db.query(`SELECT count(*)::int AS n FROM runs WHERE status <> ALL($1)`, [TERMINAL])).rows[0].n;
  await new Promise((r) => setTimeout(r, 100));
}
clearInterval(killer);
procs.forEach((p) => p.kill("SIGKILL"));

const q = async (sql: string) => (await db.query(sql)).rows[0];
const statuses = (await db.query(`SELECT agent, status, count(*)::int AS n FROM runs GROUP BY 1, 2 ORDER BY 1, 2`)).rows;
const ledger = await q(`SELECT count(*)::int AS n, coalesce(sum(amount), 0)::int AS total FROM test_ledger`);
const remoteStats = await q(`SELECT count(*)::int AS keys, coalesce(sum(deliveries), 0)::int AS deliveries FROM test_remote`);
const external = await q(`SELECT count(*)::int AS keys, coalesce(sum(attempts), 0)::int AS attempts FROM external_calls`);
const rec = await q(`SELECT count(*) FILTER (WHERE recoveries > 0)::int AS runs, coalesce(sum(recoveries), 0)::int AS total FROM runs`);
const gaps = await q(`SELECT count(*)::int AS n FROM (SELECT run_id FROM steps GROUP BY run_id HAVING max(seq) + 1 <> count(*)) g`);
const totals = await q(`SELECT count(*)::int AS steps, count(*) FILTER (WHERE kind = 'tool')::int AS tool_calls FROM steps`);

const expectLedgerRows = ledgers * 3 + bosses * 3;
const expectLedgerSum = ledgers * 180 + bosses * 3;
const checks = {
  "every run succeeded": open === 0 && statuses.every((s: { status: string }) => s.status === "succeeded"),
  "write tools ran exactly once": ledger.n === expectLedgerRows && ledger.total === expectLedgerSum,
  "one idempotency key per external call": remoteStats.keys === ledgers && external.keys === ledgers,
  "no gaps in any step log": gaps.n === 0,
};
console.log(JSON.stringify({
  runs: RUNS, singleAgentRuns: ledgers, multiAgentRuns: bosses, subAgentRuns: bosses * 3,
  statuses, seconds: Math.round((Date.now() - t0) / 1000),
  workerProcessesKilled: kills, workerProcessesStarted: started,
  runsRecoveredAfterCrash: rec.runs, totalRecoveries: rec.total,
  steps: totals.steps, toolCalls: totals.tool_calls,
  ledgerRows: ledger.n, expectedLedgerRows: expectLedgerRows,
  externalCalls: remoteStats.keys, externalDeliveries: remoteStats.deliveries, externalAttempts: external.attempts,
  checks,
}, null, 2));
await db.end();
const ok = Object.values(checks).every(Boolean);
console.log(ok ? "CHAOS TEST PASSED" : "CHAOS TEST FAILED");
process.exit(ok ? 0 : 1);
