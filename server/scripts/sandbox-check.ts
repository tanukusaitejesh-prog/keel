// Tries to escape the analyst's SQL sandbox. Every attack must fail.
import { createPool } from "../src/db.js";
import { analyst, closeAnalystPool } from "../src/agents/analyst.js";

const db = createPool();
const runSql = analyst.tools.find((t) => t.name === "run_sql")!;
const ctx = { runId: "x", callId: "y", idempotencyKey: "x:y", db, signal: new AbortController().signal };
const attacks = [
  "SELECT count(*) FROM orders",                                  // allowed
  "DELETE FROM orders",                                           // write
  "SELECT 1; DROP TABLE orders",                                  // multi-statement
  "COMMIT; DROP TABLE shop.orders",                               // escape the read-only tx
  "SELECT * FROM runs",                                           // runtime tables, other schema
  "SELECT * FROM public.runs",
  "SELECT pg_sleep(10)",                                          // timeout
  "CREATE TABLE evil (x int)",
  "SELECT set_config('role', 'postgres', true)",
  "SELECT set_config('role', 'postgres', true), pg_read_file('postgresql.conf')",
  "SELECT pg_ls_dir('.')",
  "SET ROLE postgres",
  "SELECT count(*) FROM pg_authid",
];
for (const sql of attacks) {
  try {
    const out = String(await runSql.run!(ctx, { sql }));
    console.log(`ALLOWED  ${sql}\n         -> ${out.split("\n").slice(0, 2).join(" / ")}`);
  } catch (e) {
    console.log(`BLOCKED  ${sql}\n         -> ${(e as Error).message.slice(0, 110)}`);
  }
}
const still = await db.query("SELECT count(*) FROM shop.orders");
console.log("orders still present:", still.rows[0].count);
await db.end();
await closeAnalystPool();
