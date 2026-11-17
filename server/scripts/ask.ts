// Ask an agent one question with an in-process worker and print the trace.
//   npm run ask -w server -- docs "What happens when a worker crashes?"
//   npm run ask -w server -- docs "..." --model langchain:groq/openai/gpt-oss-20b
import "../src/env.js";
import { registry } from "../src/agents/index.js";
import { envProviders } from "../src/core/providers/index.js";
import { envLimits } from "../src/core/ratelimit.js";
import { startRun, TERMINAL } from "../src/core/runtime.js";
import { startWorker } from "../src/core/worker.js";
import { createPool, migrate } from "../src/db.js";
import { langChainResolver } from "../src/langchain/model.js";

const args = process.argv.slice(2);
const mi = args.indexOf("--model");
const model = mi >= 0 ? args.splice(mi, 2)[1] : undefined;
const [agent, ...q] = args;
if (!agent || !q.length) {
  console.error('usage: ask <agent> "<question>" [--model provider:model]');
  process.exit(2);
}

const db = createPool();
await migrate(db);
const agents = registry();
const id = await startRun(db, agents, { agent, input: q.join(" "), model });
const worker = startWorker({ db, agents, providers: envProviders({ langchain: langChainResolver() }), limits: envLimits(), leaseMs: 60_000, concurrency: 4 });
const t0 = Date.now();
for (;;) {
  const { rows } = await db.query(`SELECT status FROM runs WHERE id = $1`, [id]);
  if ((TERMINAL as readonly string[]).includes(rows[0].status)) break;
  if (rows[0].status === "awaiting_approval") {
    console.log("(waiting for approval in the dashboard)");
  }
  await new Promise((r) => setTimeout(r, 500));
}
await worker.stop();
const run = (await db.query(`SELECT * FROM runs WHERE id = $1`, [id])).rows[0];
const steps = (await db.query(`SELECT kind, name, payload, latency_ms FROM steps WHERE run_id = $1 ORDER BY seq`, [id])).rows;
for (const s of steps) {
  if (s.kind === "model") {
    const t = s.payload.turn;
    console.log(`model  ${t.provider}/${t.model} ${s.latency_ms}ms  ${t.toolCalls.map((c: { name: string; args: unknown }) => `${c.name}(${JSON.stringify(c.args)})`).join(", ") || "(answer)"}`);
  } else console.log(`tool   ${s.name}${s.payload.isError ? " ERROR" : ""} ${s.latency_ms ?? "-"}ms  ${String(s.payload.content).slice(0, 140).replace(/\s+/g, " ")}`);
}
console.log(`\n${run.status} in ${((Date.now() - t0) / 1000).toFixed(1)}s, ${run.input_tokens + run.output_tokens} tokens\n`);
console.log(run.result ?? run.error);
await db.end();
