import "./env.js";
import Fastify, { type FastifyInstance } from "fastify";
import pg from "pg";
import { pathToFileURL } from "node:url";
import { z } from "zod";
import { createPool, migrate, type Db } from "./db.js";
import { registry } from "./agents/index.js";
import type { AgentRegistry } from "./core/define.js";
import { cancelRun, ConflictError, decide, forkRun, startRun } from "./core/runtime.js";

const uuid = z.string().uuid();

export async function buildApi(db: Db, agents: AgentRegistry, opts: { databaseUrl?: string } = {}): Promise<FastifyInstance> {
  const app = Fastify({ logger: false });

  // Live updates: one LISTEN connection fanned out to Server-Sent Event clients.
  const subscribers = new Set<(id: string) => void>();
  let listener: pg.Client | null = null;
  if (opts.databaseUrl) {
    listener = new pg.Client({ connectionString: opts.databaseUrl });
    await listener.connect();
    await listener.query("LISTEN keel_events");
    listener.on("notification", (n) => n.payload && subscribers.forEach((fn) => fn(n.payload!)));
  }
  app.addHook("onClose", async () => {
    await listener?.end();
  });

  app.get("/api/events", (req, reply) => {
    reply.hijack();
    reply.raw.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache", connection: "keep-alive" });
    reply.raw.write(": connected\n\n");
    const send = (id: string) => reply.raw.write(`data: ${id}\n\n`);
    const ping = setInterval(() => reply.raw.write(": ping\n\n"), 15000);
    subscribers.add(send);
    req.raw.on("close", () => {
      clearInterval(ping);
      subscribers.delete(send);
    });
  });

  app.get("/api/agents", async () =>
    agents.list().map((a) => ({ name: a.name, description: a.description, model: a.model, tools: a.tools.map((t) => ({ name: t.name, effect: t.effect, approval: !!t.needsApproval })) })),
  );

  app.get("/api/runs", async () => {
    const { rows } = await db.query(
      `SELECT r.id, r.agent, left(r.input, 140) AS input, r.model, r.status, r.input_tokens, r.output_tokens, r.created_at, r.finished_at,
              r.forked_from, (SELECT count(*)::int FROM runs c WHERE c.parent_id = r.id) AS children,
              (SELECT count(*)::int FROM approvals a WHERE a.run_id = r.id AND a.status = 'pending') AS pending_approvals
         FROM runs r WHERE r.parent_id IS NULL ORDER BY r.created_at DESC LIMIT 100`,
    );
    return rows;
  });

  app.get<{ Params: { id: string } }>("/api/runs/:id", async (req, reply) => {
    if (!uuid.safeParse(req.params.id).success) return reply.code(400).send({ error: "invalid id" });
    const run = (await db.query(`SELECT * FROM runs WHERE id = $1`, [req.params.id])).rows[0];
    if (!run) return reply.code(404).send({ error: "not found" });
    const [steps, approvals, children, external, outbox] = await Promise.all([
      db.query(`SELECT seq, kind, call_id, name, payload, latency_ms, created_at FROM steps WHERE run_id = $1 ORDER BY seq`, [run.id]),
      db.query(`SELECT * FROM approvals WHERE run_id = $1 ORDER BY created_at`, [run.id]),
      db.query(`SELECT id, agent, input, status, parent_call_id, parent_task_index, input_tokens, output_tokens, result, error FROM runs WHERE parent_id = $1 ORDER BY created_at, parent_task_index`, [run.id]),
      db.query(`SELECT idempotency_key, tool_name, attempts FROM external_calls WHERE run_id = $1`, [run.id]),
      db.query(`SELECT call_id, channel, payload, sent_at FROM outbox WHERE run_id = $1`, [run.id]),
    ]);
    return { run, steps: steps.rows, approvals: approvals.rows, children: children.rows, external: external.rows, outbox: outbox.rows };
  });

  const createBody = z.object({ agent: z.string(), input: z.string().min(1).max(8000), model: z.string().max(200).optional() });
  app.post("/api/runs", async (req, reply) => {
    const body = createBody.safeParse(req.body);
    if (!body.success) return reply.code(400).send({ error: body.error.issues });
    if (!agents.has(body.data.agent)) return reply.code(404).send({ error: "unknown agent" });
    return reply.code(201).send({ id: await startRun(db, agents, body.data) });
  });

  const decisionBody = z.object({ decision: z.enum(["approved", "rejected"]), note: z.string().max(1000).optional() });
  app.post<{ Params: { id: string; callId: string } }>("/api/runs/:id/approvals/:callId", async (req, reply) => {
    if (!uuid.safeParse(req.params.id).success) return reply.code(400).send({ error: "invalid id" });
    const body = decisionBody.safeParse(req.body);
    if (!body.success) return reply.code(400).send({ error: body.error.issues });
    try {
      await decide(db, req.params.id, req.params.callId, body.data.decision, body.data.note);
    } catch (err) {
      if (err instanceof ConflictError) return reply.code(409).send({ error: err.message });
      throw err;
    }
    return { ok: true };
  });

  app.post<{ Params: { id: string } }>("/api/runs/:id/cancel", async (req, reply) => {
    if (!uuid.safeParse(req.params.id).success) return reply.code(400).send({ error: "invalid id" });
    return { cancelled: await cancelRun(db, req.params.id) };
  });

  const forkBody = z.object({ atSeq: z.number().int().min(0), model: z.string().max(200).optional() });
  app.post<{ Params: { id: string } }>("/api/runs/:id/fork", async (req, reply) => {
    if (!uuid.safeParse(req.params.id).success) return reply.code(400).send({ error: "invalid id" });
    const body = forkBody.safeParse(req.body);
    if (!body.success) return reply.code(400).send({ error: body.error.issues });
    try {
      return reply.code(201).send({ id: await forkRun(db, req.params.id, body.data.atSeq, { model: body.data.model }) });
    } catch (err) {
      if (err instanceof ConflictError) return reply.code(409).send({ error: err.message });
      throw err;
    }
  });

  app.get("/api/stats", async () => {
    const { rows } = await db.query(
      `SELECT agent, count(*)::int AS runs,
              count(*) FILTER (WHERE status = 'succeeded')::int AS succeeded,
              count(*) FILTER (WHERE status = 'failed')::int AS failed,
              coalesce(sum(input_tokens + output_tokens), 0)::int AS tokens,
              coalesce(sum(rate_limit_waits), 0)::int AS rate_limit_waits
         FROM runs GROUP BY agent ORDER BY agent`,
    );
    return rows;
  });

  return app;
}

async function main() {
  const url = process.env.DATABASE_URL;
  const db = createPool(url);
  await migrate(db);
  const app = await buildApi(db, registry(), { databaseUrl: url });
  const port = Number(process.env.PORT ?? 3002);
  await app.listen({ port, host: "127.0.0.1" });
  console.log(`[api] http://127.0.0.1:${port}`);
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) void main();
