// Tries to escape the analyst's SQL sandbox. Exits non-zero if any attack succeeds
// or if the one legitimate query is blocked.
import { createPool } from "../src/db.js";
import { analyst, closeAnalystPool } from "../src/agents/analyst.js";

const db = createPool();
const runSql = analyst.tools.find((t) => t.name === "run_sql")!;
const ctx = { runId: "x", callId: "y", idempotencyKey: "x:y", db, signal: new AbortController().signal };
const SAFE = "SELECT count(*) FROM orders"; // must be allowed
const attacks = [
  "DELETE FROM orders", // write
  "SELECT 1; DROP TABLE orders", // multi-statement
  "COMMIT; DROP TABLE shop.orders", // escape the read-only transaction
  "SELECT * FROM runs", // runtime tables are in another schema
  "SELECT * FROM public.runs",
  "SELECT pg_sleep(10)", // statement timeout
  "CREATE TABLE evil (x int)",
  "SELECT set_config('role', 'postgres', true)", // privilege escalation
  "SELECT set_config('role', 'postgres', true), pg_read_file('postgresql.conf')",
  "SELECT pg_ls_dir('.')", // file system access
  "SET ROLE postgres",
  "SELECT count(*) FROM pg_authid", // password hashes
];

let failures = 0;
for (const sql of [SAFE, ...attacks]) {
  let allowed = false;
  let detail: string;
  try {
    detail = String(await runSql.run!(ctx, { sql })).split("\n").slice(0, 2).join(" / ");
    allowed = true;
  } catch (e) {
    detail = (e as Error).message.slice(0, 110);
  }
  const ok = allowed === (sql === SAFE);
  if (!ok) failures++;
  console.log(`${allowed ? "ALLOWED" : "BLOCKED"}  ${ok ? "" : "<- UNEXPECTED "}${sql}\n         -> ${detail}`);
}
const still = await db.query("SELECT count(*) FROM shop.orders");
console.log(`orders still present: ${still.rows[0].count}`);
await db.end();
await closeAnalystPool();
console.log(failures ? `SANDBOX CHECK FAILED (${failures})` : `SANDBOX CHECK PASSED: ${attacks.length} attacks blocked, normal reads allowed`);
process.exit(failures ? 1 : 0);
