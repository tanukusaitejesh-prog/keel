import { withTx, type Db } from "../db.js";

/**
 * Token bucket in Postgres, shared by every worker process, one per
 * provider:model. Workers reserve an estimate before calling the model and
 * reconcile with the real usage afterwards. When the bucket is short, the
 * caller parks the run (not_before) instead of sleeping, so no worker slot is
 * held while waiting.
 */
export interface Limits {
  /** Tokens per minute for a provider:model key, or undefined for unlimited. */
  tpm(key: string): number | undefined;
}

export function envLimits(): Limits {
  return {
    tpm(key) {
      const provider = key.split(":")[0]!.toUpperCase();
      const specific = process.env[`TPM_${key.toUpperCase().replace(/[^A-Z0-9]/g, "_")}`];
      const v = Number(specific ?? process.env[`${provider}_TPM`]);
      return Number.isFinite(v) && v > 0 ? v : undefined;
    },
  };
}

/**
 * Try to take `tokens` from the bucket. Returns how many tokens were reserved
 * (to reconcile later) and, if none could be, how long to wait.
 */
export async function reserve(db: Db, key: string, tpm: number, tokens: number): Promise<{ reserved: number; waitMs: number }> {
  // A request larger than the whole bucket could never run; let it through when the bucket is full.
  const want = Math.min(tokens, tpm);
  const refill = tpm / 60;
  return withTx(db, async (tx) => {
    await tx.query(
      `INSERT INTO rate_buckets (key, capacity, tokens, refill_per_sec) VALUES ($1, $2, $2, $3) ON CONFLICT (key) DO NOTHING`,
      [key, tpm, refill],
    );
    const { rows } = await tx.query<{ available: number }>(
      `SELECT least($2::float8, tokens + extract(epoch FROM clock_timestamp() - updated_at) * $3::float8) AS available
         FROM rate_buckets WHERE key = $1 FOR UPDATE`,
      [key, tpm, refill],
    );
    const available = Number(rows[0]!.available);
    const ok = available >= want;
    await tx.query(
      `UPDATE rate_buckets SET tokens = $2, capacity = $3, refill_per_sec = $4, updated_at = clock_timestamp() WHERE key = $1`,
      [key, ok ? available - want : available, tpm, refill],
    );
    return ok ? { reserved: want, waitMs: 0 } : { reserved: 0, waitMs: Math.ceil(((want - available) / refill) * 1000) };
  });
}

/**
 * Correct the bucket once the real usage is known: delta = actual - reserved.
 * It may go negative (the debt is paid by waiting) but never above capacity.
 */
export async function reconcile(db: Db, key: string, delta: number): Promise<void> {
  if (!delta) return;
  await db.query(`UPDATE rate_buckets SET tokens = least(capacity, tokens - $2) WHERE key = $1`, [key, delta]);
}

/** Drain the bucket after the provider itself said 429, so other workers back off too. */
export async function drain(db: Db, key: string): Promise<void> {
  await db.query(`UPDATE rate_buckets SET tokens = least(tokens, 0), updated_at = clock_timestamp() WHERE key = $1`, [key]);
}

/** Rough token estimate for a request: ~4 characters per token plus the output allowance. */
export function estimateTokens(payload: unknown, maxOutput: number): number {
  return Math.ceil(JSON.stringify(payload).length / 4) + Math.min(maxOutput, 1024);
}
