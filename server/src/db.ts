import pg from "pg";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { createHash } from "node:crypto";

export type Db = pg.Pool;
export type Tx = pg.PoolClient;

export function createPool(url = process.env.DATABASE_URL): Db {
  if (!url) throw new Error("DATABASE_URL is not set");
  return new pg.Pool({ connectionString: url, max: Number(process.env.PG_POOL_MAX ?? 20) });
}

const here = path.dirname(fileURLToPath(import.meta.url));

/**
 * Apply schema.sql if it changed since the last migration. Booting a worker
 * against an up-to-date database runs no DDL at all, so it never takes table
 * locks that would conflict with live agent transactions.
 */
export async function migrate(db: Db): Promise<void> {
  const sql = readFileSync(path.join(here, "schema.sql"), "utf8");
  const hash = createHash("sha256").update(sql).digest("hex");
  const client = await db.connect();
  try {
    // Serialize concurrent migrations (several workers booting at once).
    await client.query("SELECT pg_advisory_lock(727101)");
    await client.query("CREATE TABLE IF NOT EXISTS schema_meta (id int PRIMARY KEY CHECK (id = 1), hash text NOT NULL)");
    const current = await client.query("SELECT hash FROM schema_meta WHERE id = 1");
    if (current.rows[0]?.hash === hash) return;
    await client.query("BEGIN");
    await client.query(sql);
    await client.query("INSERT INTO schema_meta (id, hash) VALUES (1, $1) ON CONFLICT (id) DO UPDATE SET hash = EXCLUDED.hash", [hash]);
    await client.query("COMMIT");
  } catch (err) {
    await client.query("ROLLBACK").catch(() => {});
    throw err;
  } finally {
    await client.query("SELECT pg_advisory_unlock(727101)").catch(() => {});
    client.release();
  }
}

const TRANSIENT_PG_CODES = new Set(["40P01", "40001", "55P03", "57P01", "57P02", "57P03", "53300"]);

/** Deadlocks, serialization failures, dropped connections: safe to retry. */
export function isTransientDbError(err: unknown): boolean {
  const e = err as { code?: string; message?: string } | null;
  if (!e) return false;
  if (e.code && (TRANSIENT_PG_CODES.has(e.code) || e.code.startsWith("08"))) return true;
  if (e.code === "ECONNRESET" || e.code === "ECONNREFUSED" || e.code === "EPIPE") return true;
  return /Connection terminated|connection.*(closed|ended)/i.test(e.message ?? "");
}

/** Run fn inside a transaction. A thrown error rolls back. */
export async function withTx<T>(db: Db, fn: (tx: Tx) => Promise<T>): Promise<T> {
  const tx = await db.connect();
  try {
    await tx.query("BEGIN");
    const out = await fn(tx);
    await tx.query("COMMIT");
    return out;
  } catch (err) {
    await tx.query("ROLLBACK").catch(() => {});
    throw err;
  } finally {
    tx.release();
  }
}
