import { pathToFileURL } from "node:url";
import { createPool, migrate, type Db } from "./db.js";
import { registry } from "./agents/index.js";
import { envProviders } from "./core/providers/index.js";
import { envLimits } from "./core/ratelimit.js";
import { startWorker } from "./core/worker.js";

/** Deliver outbox messages. At least once: real channels should receive outbox.id as an idempotency key. */
export async function dispatchOutbox(db: Db, send: (m: { channel: string; payload: unknown }) => Promise<void>, batch = 20): Promise<number> {
  const client = await db.connect();
  try {
    await client.query("BEGIN");
    const { rows } = await client.query(`SELECT id, channel, payload FROM outbox WHERE sent_at IS NULL ORDER BY id FOR UPDATE SKIP LOCKED LIMIT $1`, [batch]);
    for (const r of rows) {
      await send(r);
      await client.query(`UPDATE outbox SET sent_at = now() WHERE id = $1`, [r.id]);
    }
    await client.query("COMMIT");
    return rows.length;
  } catch (err) {
    await client.query("ROLLBACK").catch(() => {});
    throw err;
  } finally {
    client.release();
  }
}

async function main() {
  const db = createPool();
  await migrate(db);
  const leaseMs = Number(process.env.LEASE_MS ?? 30000);
  const concurrency = Number(process.env.WORKER_CONCURRENCY ?? 4);
  const worker = startWorker({
    db,
    agents: registry(),
    providers: envProviders(),
    limits: envLimits(),
    leaseMs,
    concurrency,
    onOutcome: (id, o) => console.log(`[worker] ${id} -> ${o}`),
  });
  let stopping = false;
  const dispatcher = (async () => {
    while (!stopping) {
      try {
        const n = await dispatchOutbox(db, async (m) => console.log(`[outbox] ${m.channel}: ${JSON.stringify(m.payload).slice(0, 160)}`));
        if (!n) await new Promise((r) => setTimeout(r, 1000));
      } catch (err) {
        console.error("[outbox]", (err as Error).message);
        await new Promise((r) => setTimeout(r, 1000));
      }
    }
  })();
  console.log(`[worker] pid=${process.pid} concurrency=${concurrency} leaseMs=${leaseMs}`);
  const shutdown = async () => {
    stopping = true;
    await worker.stop();
    await dispatcher;
    await db.end();
    process.exit(0);
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) void main();
