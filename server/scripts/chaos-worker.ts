// Worker process for the chaos test: runs the scripted test agents.
import { createPool } from "../src/db.js";
import { startWorker } from "../src/core/worker.js";
import { agents, remote, scripted, TEST_DB_URL } from "../test/helpers.js";

const db = createPool(TEST_DB_URL);
remote.record = async (key: string) => {
  await db.query(`INSERT INTO test_remote (key, deliveries) VALUES ($1, 1) ON CONFLICT (key) DO UPDATE SET deliveries = test_remote.deliveries + 1`, [key]);
};
startWorker({ db, agents: agents(), providers: scripted(Number(process.env.LATENCY_MS ?? 200)), leaseMs: Number(process.env.LEASE_MS ?? 1500), concurrency: 4 });
