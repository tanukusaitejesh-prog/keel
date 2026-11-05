import pg from "pg";
import { z } from "zod";
import { analystPassword } from "./shop-data.js";
import { defineAgent, defineTool } from "../core/define.js";
import { ToolError } from "../core/types.js";

const MAX_ROWS = 50;
const TEXT_TIME_OIDS = new Set([1082, 1083, 1114, 1184, 1266]); // date, time, timestamp, timestamptz, timetz

/** Connections as the unprivileged keel_analyst login role (created by seedShop). */
let pool: pg.Pool | null = null;
export function analystPool(): pg.Pool {
  if (!pool) {
    const url = new URL(process.env.ANALYST_DATABASE_URL ?? process.env.DATABASE_URL ?? "");
    if (!process.env.ANALYST_DATABASE_URL) {
      url.username = "keel_analyst";
      url.password = analystPassword();
    }
    pool = new pg.Pool({
      connectionString: url.toString(),
      max: 5,
      // Show dates and timestamps exactly as Postgres prints them. Converting to JS
      // Dates and back shifts them across time zones (2025-01-01 became 2024-12-31 in IST).
      types: { getTypeParser: ((oid: number, format?: "text" | "binary") => (TEXT_TIME_OIDS.has(oid) ? (v: string) => v : pg.types.getTypeParser(oid, format))) as typeof pg.types.getTypeParser },
    });
  }
  return pool;
}
export async function closeAnalystPool() {
  await pool?.end();
  pool = null;
}

const describeSchema = defineTool({
  name: "describe_schema",
  description: "List every table in the shop database with its columns, types and documentation. Call this first.",
  effect: "read",
  input: z.object({}),
  async run({ db }) {
    const { rows } = await db.query(
      `SELECT c.table_name, c.column_name, c.data_type,
              col_description(format('shop.%I', c.table_name)::regclass, c.ordinal_position) AS col_doc,
              obj_description(format('shop.%I', c.table_name)::regclass, 'pg_class') AS table_doc
         FROM information_schema.columns c
        WHERE c.table_schema = 'shop'
        ORDER BY c.table_name, c.ordinal_position`,
    );
    const tables = new Map<string, { doc: string | null; cols: string[] }>();
    for (const r of rows) {
      const t = tables.get(r.table_name) ?? { doc: r.table_doc as string | null, cols: [] as string[] };
      t.cols.push(`${r.column_name} ${r.data_type}${r.col_doc ? ` -- ${r.col_doc}` : ""}`);
      tables.set(r.table_name, t);
    }
    return [...tables].map(([name, t]) => `${name}${t.doc ? `: ${t.doc}` : ""}\n  ${t.cols.join("\n  ")}`).join("\n");
  },
});

/**
 * Read-only SQL. Three independent guards: a READ ONLY transaction, a role
 * with SELECT-only grants on one schema, and the extended query protocol,
 * which rejects multi-statement strings such as "COMMIT; DROP ...".
 */
const runSql = defineTool({
  name: "run_sql",
  description: `Run one read-only PostgreSQL SELECT against the shop schema. Returns at most ${MAX_ROWS} rows. On an error you get the database message; fix the query and retry.`,
  effect: "read",
  maxOutputChars: 4000,
  input: z.object({ sql: z.string().min(1).max(4000).describe("A single SELECT (or WITH ... SELECT) statement") }),
  async run(_ctx, { sql }) {
    const client = await analystPool().connect();
    try {
      await client.query("BEGIN READ ONLY");
      await client.query("SET LOCAL statement_timeout = '5s'");
      const res = await client.query({ text: sql, rowMode: "array", queryMode: "extended" } as any);
      const cols = res.fields.map((f: { name: string }) => f.name);
      const rows = (res.rows as unknown[][]).slice(0, MAX_ROWS).map((r) => r.map(cell).join(" | "));
      const more = (res.rowCount ?? 0) > MAX_ROWS ? `\n(${res.rowCount} rows total, showing ${MAX_ROWS})` : "";
      return `${cols.join(" | ")}\n${rows.join("\n")}${rows.length ? "" : "(no rows)"}${more}`;
    } catch (err) {
      const e = err as { message?: string; hint?: string; position?: string };
      throw new ToolError(`SQL error: ${e.message}${e.hint ? ` (hint: ${e.hint})` : ""}${e.position ? ` at character ${e.position}` : ""}`);
    } finally {
      await client.query("ROLLBACK").catch(() => {});
      client.release();
    }
  },
});

function cell(v: unknown): string {
  return v === null ? "NULL" : String(v);
}

const shareReport = defineTool({
  name: "share_report",
  description: "Post a short report to the team's #analytics channel. A human must approve it before it is posted.",
  effect: "write",
  needsApproval: true,
  input: z.object({ title: z.string().min(1).max(120), body: z.string().min(1).max(3000) }),
  async run({ tx, runId, callId }, { title, body }) {
    await tx!.query(`INSERT INTO outbox (run_id, call_id, channel, payload) VALUES ($1, $2, 'slack:#analytics', $3)`, [runId, callId, JSON.stringify({ title, body })]);
    return { posted: true, channel: "#analytics" };
  },
});

export const analyst = defineAgent({
  name: "analyst",
  description: "Answers business questions by writing SQL against the shop database; can share reports after approval.",
  model: "groq:openai/gpt-oss-120b",
  reasoningEffort: "low",
  maxTurns: 12,
  maxTokens: 60_000,
  maxOutputTokens: 2048,
  system: `You are a careful data analyst with read-only SQL access to a PostgreSQL database (schema "shop").
Call describe_schema first and follow the business definitions in its documentation exactly.
Answer with SQL, never by guessing. If a query fails, read the error, fix the query and run it again.
Before answering, check the result makes sense (e.g. ties, NULLs, date ranges). Only share a report when the user asks for one.
Reply in one or two sentences, and end with a final line in the form "Answer: <value>".`,
  tools: [describeSchema, runSql, shareReport],
});
