import { hostname } from "node:os";
import { randomUUID } from "node:crypto";
import { claimRun, executeRun, type RunOutcome, type RuntimeOptions } from "./runtime.js";

export interface WorkerHandle {
  stop(): Promise<void>;
}

/** Start `concurrency` claim loops. Each drives one run at a time until it ends, pauses or parks. */
export function startWorker(
  opts: RuntimeOptions & { concurrency: number; pollMs?: number; owner?: string; onOutcome?: (runId: string, o: RunOutcome) => void },
): WorkerHandle {
  const owner = opts.owner ?? `${hostname()}:${process.pid}:${randomUUID().slice(0, 6)}`;
  let stopping = false;
  const loops = Array.from({ length: opts.concurrency }, async (_, i) => {
    const slot = `${owner}#${i}`;
    while (!stopping) {
      let run = null;
      try {
        run = await claimRun(opts.db, slot, opts.leaseMs);
      } catch (err) {
        if (!stopping) console.error("[worker] claim failed:", (err as Error).message);
      }
      if (!run) {
        await new Promise((r) => setTimeout(r, opts.pollMs ?? 250));
        continue;
      }
      try {
        // Not `onOutcome?.(id, await executeRun(...))`: optional chaining would skip
        // evaluating the arguments, so the run would never execute without a callback.
        const outcome = await executeRun(run, slot, opts);
        opts.onOutcome?.(run.id, outcome);
      } catch (err) {
        console.error(`[worker] run ${run.id} crashed the loop:`, err);
      }
    }
  });
  return {
    async stop() {
      stopping = true;
      await Promise.all(loops);
    },
  };
}
