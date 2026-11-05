import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import type { FastifyInstance } from "fastify";
import type { Db } from "../src/db.js";
import { buildApi } from "../src/api.js";
import { claimRun, executeRun } from "../src/core/runtime.js";
import { agents, freshDb, reset, scripted, settle } from "./helpers.js";

let db: Db;
let app: FastifyInstance;
const reg = agents();
beforeAll(async () => {
  db = await freshDb();
  app = await buildApi(db, reg);
});
beforeEach(async () => reset(db));
afterAll(async () => {
  await app.close();
  await db.end();
});

const post = (url: string, payload: unknown) => app.inject({ method: "POST", url, payload: payload as object });

describe("HTTP API", () => {
  it("validates input", async () => {
    expect((await post("/api/runs", { agent: "ledger" })).statusCode).toBe(400);
    expect((await post("/api/runs", { agent: "nope", input: "x" })).statusCode).toBe(404);
    expect((await app.inject({ method: "GET", url: "/api/runs/not-a-uuid" })).statusCode).toBe(400);
    expect((await app.inject({ method: "GET", url: "/api/runs/00000000-0000-0000-0000-000000000000" })).statusCode).toBe(404);
  });

  it("starts a run, exposes the approval, and accepts exactly one decision", async () => {
    const res = await post("/api/runs", { agent: "ledger", input: "go" });
    expect(res.statusCode).toBe(201);
    const { id } = res.json();
    const run = (await claimRun(db, "w", 5000))!;
    expect(await executeRun(run, "w", { db, agents: reg, providers: scripted(), leaseMs: 5000 })).toBe("paused");

    const detail = (await app.inject({ method: "GET", url: `/api/runs/${id}` })).json();
    expect(detail.run.status).toBe("awaiting_approval");
    expect(detail.external).toHaveLength(1);
    const pending = detail.approvals.filter((a: { status: string }) => a.status === "pending");
    expect(pending).toHaveLength(1);
    const url = `/api/runs/${id}/approvals/${pending[0].call_id}`;
    expect((await post(url, { decision: "maybe" })).statusCode).toBe(400);
    expect((await post(url, { decision: "approved" })).statusCode).toBe(200);
    expect((await post(url, { decision: "rejected" })).statusCode).toBe(409);
  });

  it("lists sub-agents, forks at model turns only, and cancels", async () => {
    const { id } = (await post("/api/runs", { agent: "boss", input: "go" })).json();
    await settle(db, [id]);
    const detail = (await app.inject({ method: "GET", url: `/api/runs/${id}` })).json();
    expect(detail.children).toHaveLength(3);
    const list = (await app.inject({ method: "GET", url: "/api/runs" })).json();
    expect(list).toHaveLength(1); // sub-agent runs are not listed at the top level
    expect(list[0].children).toBe(3);

    const toolSeq = detail.steps.find((s: { kind: string }) => s.kind === "tool").seq;
    expect((await post(`/api/runs/${id}/fork`, { atSeq: toolSeq })).statusCode).toBe(409);
    const fork = await post(`/api/runs/${id}/fork`, { atSeq: 0 });
    expect(fork.statusCode).toBe(201);
    expect((await post(`/api/runs/${fork.json().id}/cancel`, {})).json()).toEqual({ cancelled: 1 });
  });
});
