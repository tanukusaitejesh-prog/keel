import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import type { Db } from "../src/db.js";
import { conv, envProviders, ScriptedProvider } from "../src/core/providers/index.js";
import { cancelRun, claimRun, executeRun, forkRun, replay, SimulatedCrash, startRun, type FaultPoint } from "../src/core/runtime.js";
import { RateLimited, TransientModelError, type ModelProvider } from "../src/core/types.js";
import { __test } from "../src/core/providers/openai.js";
import { agents, freshDb, remote, reset, runRow, scripted, settle, stepsOf } from "./helpers.js";

let db: Db;
beforeAll(async () => {
  db = await freshDb();
});
beforeEach(async () => reset(db));
afterAll(async () => db.end());

const reg = agents();
const start = (agent: string, input = "go") => startRun(db, reg, { agent, input });
const ledger = async () => (await db.query(`SELECT amount FROM test_ledger ORDER BY amount`)).rows.map((r) => r.amount);

describe("single agent", () => {
  it("runs read, write and external tools, pausing for approval only when the predicate says so", async () => {
    const id = await start("ledger");
    const run = (await claimRun(db, "w", 5000))!;
    expect(await executeRun(run, "w", { db, agents: reg, providers: scripted(), leaseMs: 5000 })).toBe("paused");
    // The two small entries and the notification already ran; the 150 entry waits for a human.
    expect(await ledger()).toEqual([10, 20]);
    expect((await runRow(db, id)).status).toBe("awaiting_approval");
    const pending = (await db.query(`SELECT input FROM approvals WHERE status = 'pending'`)).rows;
    expect(pending).toEqual([{ input: { amount: 150 } }]);

    await settle(db, [id]);
    const r = await runRow(db, id);
    expect(r.status).toBe("succeeded");
    expect(r.result).toBe("done: ok,ok,ok,ok,ok");
    expect(await ledger()).toEqual([10, 20, 150]);
    expect([...remote.calls.values()]).toEqual([1]);
  });

  it("a rejected call becomes an error result with the reviewer's note", async () => {
    const id = await start("ledger");
    await settle(db, [id], { approver: () => ({ decision: "rejected", note: "too large" }) });
    const steps = await stepsOf(db, id);
    const rejected = steps.find((s) => s.payload.args?.amount === 150)!;
    expect(rejected.payload).toMatchObject({ isError: true });
    expect(rejected.payload.content).toContain("too large");
    expect(await ledger()).toEqual([10, 20]);
    expect((await runRow(db, id)).result).toBe("done: ok,ok,ok,ok,E");
  });

  it("returns unknown tools, invalid JSON and schema violations to the model instead of crashing", async () => {
    const providers = envProviders({
      scripted: (model) =>
        new ScriptedProvider(model, (m) =>
          conv.turns(m) === 0
            ? {
                calls: [
                  { name: "nope", args: {} },
                  { name: "add_entry", args: { __invalid_json: "{amount:" } },
                  { name: "add_entry", args: { amount: "ten" } },
                  { name: "add_entry", args: { amount: -5 } },
                ],
              }
            : { text: conv.lastResults(m).map((r) => r.content.slice(0, 20)).join(" | ") },
        ),
    });
    const id = await start("ledger");
    await settle(db, [id], { providers });
    const r = await runRow(db, id);
    expect(r.status).toBe("succeeded");
    expect(r.result).toBe("Unknown tool \"nope\". | Your arguments were  | Invalid arguments: a | amount must be posit");
    expect(await ledger()).toEqual([]);
  });

  it("enforces turn and token budgets", async () => {
    const id = await start("ledger");
    await db.query(`UPDATE runs SET max_turns = 2 WHERE id = $1`, [id]);
    const id2 = await start("ledger");
    await db.query(`UPDATE runs SET max_tokens = 10 WHERE id = $1`, [id2]);
    await settle(db, [id, id2]);
    expect((await runRow(db, id)).error).toContain("turn budget exhausted");
    expect((await runRow(db, id2)).error).toContain("token budget exhausted");
  });
});

describe("multi-agent", () => {
  it("fans out sub-agents in parallel, sleeps without a worker, and resumes with their results", async () => {
    const id = await start("boss");
    const run = (await claimRun(db, "w", 5000))!;
    expect(await executeRun(run, "w", { db, agents: reg, providers: scripted(), leaseMs: 5000 })).toBe("waiting");
    const parent = await runRow(db, id);
    expect(parent.status).toBe("waiting");
    expect(parent.lease_owner).toBeNull();
    const kids = (await db.query(`SELECT input, depth, status FROM runs WHERE parent_id = $1 ORDER BY input`, [id])).rows;
    expect(kids).toEqual(["a", "b", "c"].map((input) => ({ input, depth: 1, status: "queued" }))); // all spawned at once

    await settle(db, [id], { workers: 2, concurrency: 2 });
    const done = await runRow(db, id);
    expect(done.status).toBe("succeeded");
    expect(done.result).toBe("combined: did a + did b + did c");
    expect(await ledger()).toEqual([1, 1, 1]);
  });

  it("a failed sub-agent becomes an error result and the parent carries on", async () => {
    const providers = envProviders({
      scripted: (model) =>
        new ScriptedProvider(model, (m) => {
          if (model === "boss") return conv.turns(m) === 0 ? { calls: [{ name: "delegate", args: { task: "ok" } }, { name: "delegate", args: { task: "fail" } }] } : { text: conv.toolResults(m).map((r) => `${r.isError ? "ERR" : "OK"}`).join(",") };
          const task = (m[0] as { content: string }).content;
          return task === "fail" ? { calls: [{ name: "add_entry", args: { amount: 0.5 } }] } : conv.turns(m) === 0 ? { calls: [{ name: "add_entry", args: { amount: 1 } }] } : { text: "fine" };
        }),
    });
    const id = await start("boss");
    await settle(db, [id], { providers });
    expect((await runRow(db, id)).result).toBe("OK,ERR");
    const failed = (await db.query(`SELECT error FROM runs WHERE parent_id = $1 AND status = 'failed'`, [id])).rows;
    expect(failed[0].error).toContain("turn budget exhausted");
  });

  it("stops runaway recursion at the depth limit", async () => {
    const id = await start("recursive");
    await settle(db, [id], { maxDepth: 2 });
    const depths = (await db.query(`SELECT depth, status FROM runs ORDER BY depth`)).rows;
    expect(depths).toEqual([0, 1, 2].map((depth) => ({ depth, status: "succeeded" })));
    const deepest = (await db.query(`SELECT result FROM runs WHERE depth = 2`)).rows[0].result;
    expect(deepest).toContain("maximum agent nesting depth");
  });

  it("cancelling a parent cancels its whole tree, and a worker holding a child stops writing", async () => {
    const id = await start("boss");
    const parentRun = (await claimRun(db, "w", 5000))!;
    await executeRun(parentRun, "w", { db, agents: reg, providers: scripted(), leaseMs: 5000 });
    const child = (await claimRun(db, "w2", 5000))!;
    expect(child.depth).toBe(1);
    expect(await cancelRun(db, id)).toBe(4);
    expect(await executeRun(child, "w2", { db, agents: reg, providers: scripted(), leaseMs: 5000 })).toBe("abandoned");
    expect((await db.query(`SELECT DISTINCT status FROM runs`)).rows).toEqual([{ status: "cancelled" }]);
    expect(await ledger()).toEqual([]);
  });
});

describe("time travel", () => {
  it("forks a run at a model turn on another model, reusing recorded results without re-running side effects", async () => {
    const id = await start("ledger");
    await settle(db, [id]);
    const steps = await stepsOf(db, id);
    const thirdTurn = steps.filter((s) => s.kind === "model")[2]!.seq; // the turn that asked for the 150 entry
    const forkId = await forkRun(db, id, thirdTurn, { model: "scripted:ledger" });
    expect((await stepsOf(db, forkId)).length).toBe(thirdTurn);
    await settle(db, [forkId]);
    const fork = await runRow(db, forkId);
    expect(fork.status).toBe("succeeded");
    expect(fork.forked_from).toBe(id);
    // Only the call made after the fork point ran again: one more 150 entry, the 10/20 entries and the notify did not repeat.
    expect(await ledger()).toEqual([10, 20, 150, 150]);
    expect([...remote.calls.values()]).toEqual([1]);
    await expect(forkRun(db, id, 1)).rejects.toThrow("model turn");
  });
});

describe("rate limits and transient failures", () => {
  it("parks a run when the shared token bucket is empty instead of holding a worker", async () => {
    const id = await start("ledger");
    const limits = { tpm: () => 600 }; // ~10 tokens per second; each scripted turn needs more than that
    const run = (await claimRun(db, "w", 5000))!;
    expect(await executeRun(run, "w", { db, agents: reg, providers: scripted(), leaseMs: 5000, limits })).toBe("deferred");
    const parked = await runRow(db, id);
    expect(parked.status).toBe("queued");
    expect(parked.rate_limit_waits).toBe(1);
    expect(new Date(parked.not_before).getTime()).toBeGreaterThan(Date.now());
    expect(await claimRun(db, "w", 5000)).toBeNull(); // not claimable until not_before
  });

  it("drains the bucket and parks on a provider 429, then finishes", async () => {
    let calls = 0;
    const flaky: ModelProvider = { provider: "scripted", model: "ledger", complete: async (req) => {
      if (calls++ === 1) throw new RateLimited(300, "429 too many tokens");
      return new ScriptedProvider("ledger", (await import("./helpers.js")).policies.ledger!).complete(req);
    } };
    const id = await start("ledger");
    await settle(db, [id], { providers: () => flaky, limits: { tpm: () => 1_000_000 } });
    const r = await runRow(db, id);
    expect(r.status).toBe("succeeded");
    expect(r.rate_limit_waits).toBe(1);
    expect((await db.query(`SELECT tokens FROM rate_buckets`)).rows[0].tokens).toBeLessThan(1_000_000);
  });

  it("retries transient model errors with backoff, then gives up after five", async () => {
    const make = (failures: number): ModelProvider => {
      let n = 0;
      const inner = new ScriptedProvider("ledger", (m) => (conv.turns(m) === 0 ? { text: "only text" } : { text: "x" }));
      return { provider: "scripted", model: "ledger", complete: async (req) => {
        if (n++ < failures) throw new TransientModelError("tool_use_failed");
        return inner.complete(req);
      } };
    };
    const ok = await start("ledger");
    const p1 = make(1);
    await settle(db, [ok], { providers: () => p1 });
    expect((await runRow(db, ok)).status).toBe("succeeded");
    expect((await runRow(db, ok)).transient_errors).toBe(1);

    const bad = await start("ledger");
    await db.query(`UPDATE runs SET transient_errors = 4 WHERE id = $1`, [bad]); // skip the backoff waits
    const p2 = make(100);
    await settle(db, [bad], { providers: () => p2 });
    expect((await runRow(db, bad)).status).toBe("failed");
    expect((await runRow(db, bad)).transient_errors).toBe(5);
  });
});

describe("durability", () => {
  it("a worker that lost its lease cannot write anything", async () => {
    const id = await start("ledger");
    const stale = (await claimRun(db, "slow", 100))!;
    await new Promise((r) => setTimeout(r, 200));
    const fresh = (await claimRun(db, "new", 5000))!;
    expect(fresh.lease_token).toBe(stale.lease_token + 1);
    expect(await executeRun(stale, "slow", { db, agents: reg, providers: scripted(), leaseMs: 100 })).toBe("abandoned");
    expect(await stepsOf(db, id)).toEqual([]);
  });

  it("crashing at every fault point of a multi-agent workflow always ends in the clean-run state", async () => {
    const snapshot = async () => ({
      runs: (await db.query(`SELECT agent, input, depth, status, result FROM runs ORDER BY depth, agent, input`)).rows,
      steps: (await db.query(`SELECT r.agent, r.input, s.seq, s.kind, s.name, s.payload->'content' AS content, s.payload->'turn'->'toolCalls' AS calls FROM steps s JOIN runs r ON r.id = s.run_id ORDER BY r.depth, r.agent, r.input, s.seq`)).rows,
      ledger: await ledger(),
      remoteKeys: remote.calls.size,
    });
    const scenario = async () => [await start("ledger"), await start("boss")];

    let points = 0;
    await settle(db, await scenario(), { fault: () => void points++ });
    const expected = await snapshot();
    expect(expected.runs.every((r: { status: string }) => r.status === "succeeded")).toBe(true);
    expect(points).toBeGreaterThan(40);

    const kinds = new Set<FaultPoint>();
    let externalRetries = 0;
    for (let k = 1; k <= points; k++) {
      await reset(db);
      let n = 0;
      await settle(db, await scenario(), {
        leaseMs: 120,
        fault: (p) => {
          if (++n === k) {
            kinds.add(p);
            throw new SimulatedCrash(`#${k} ${p}`);
          }
        },
      });
      expect(await snapshot(), `crash at fault point #${k}`).toEqual(expected);
      externalRetries += [...remote.calls.values()].filter((c) => c > 1).length;
    }
    expect([...kinds].sort()).toEqual([
      "agent:before-commit", "approval:before-commit", "model:after-commit", "model:before-call", "model:before-commit",
      "tool:after-commit", "tool:before-commit", "tool:before-run",
    ]);
    // External calls are at-least-once: some crashes forced a retry, always with the same key.
    expect(externalRetries).toBeGreaterThan(0);
  });

  it("many workers, random crashes, mixed agents: every run finishes and writes happen exactly once", async () => {
    const ids: string[] = [];
    for (let i = 0; i < 12; i++) ids.push(await start("ledger"), await start("boss"));
    let s = 99;
    const rand = () => (s = (s * 1103515245 + 12345) >>> 0) / 2 ** 32;
    let crashes = 0;
    await settle(db, ids, {
      workers: 4,
      concurrency: 3,
      leaseMs: 150,
      timeoutMs: 110_000,
      fault: () => {
        if (rand() < 0.06) {
          crashes++;
          throw new SimulatedCrash("random");
        }
      },
    });
    expect(crashes).toBeGreaterThan(10);
    const statuses = (await db.query(`SELECT agent, status, count(*)::int AS n FROM runs GROUP BY 1, 2 ORDER BY 1`)).rows;
    expect(statuses).toEqual([
      { agent: "boss", status: "succeeded", n: 12 },
      { agent: "helper", status: "succeeded", n: 36 },
      { agent: "ledger", status: "succeeded", n: 12 },
    ]);
    // Each ledger run writes 10+20+150, each helper writes 1: exactly once each.
    expect((await db.query(`SELECT count(*)::int AS n, sum(amount)::int AS total FROM test_ledger`)).rows[0]).toEqual({ n: 12 * 3 + 36, total: 12 * 180 + 36 });
    expect(remote.calls.size).toBe(12); // one idempotency key per notify call
    const gaps = await db.query(`SELECT run_id FROM steps GROUP BY run_id HAVING max(seq) + 1 <> count(*)`);
    expect(gaps.rowCount).toBe(0);
  });
});

describe("units", () => {
  it("replays a log into a conversation and finds unanswered calls", () => {
    const st = replay("hi", [
      { seq: 0, kind: "model", call_id: null, name: null, payload: { turn: { text: "", stop: "tool_calls", toolCalls: [{ id: "a", name: "x", args: {} }, { id: "b", name: "y", args: {} }], usage: { input: 1, output: 1 }, model: "m", provider: "p" } } },
      { seq: 1, kind: "tool", call_id: "a", name: "x", payload: { content: "r", isError: false } },
    ]);
    expect(st.pending.map((c) => c.id)).toEqual(["b"]);
    expect(st.messages.map((m) => m.role)).toEqual(["user", "assistant", "tool"]);
    expect(st.nextSeq).toBe(2);
  });

  it("parses Groq-style retry headers", () => {
    const h = (o: Record<string, string>) => new Headers(o);
    expect(__test.retryAfterMs(h({ "retry-after": "2" }))).toBe(2000);
    expect(__test.retryAfterMs(h({ "x-ratelimit-reset-tokens": "7.66s" }))).toBe(7660);
    expect(__test.retryAfterMs(h({ "x-ratelimit-reset-tokens": "1m26.4s" }))).toBe(86400);
    expect(__test.retryAfterMs(h({ "x-ratelimit-reset-requests": "2h" }))).toBe(7_200_000);
    expect(__test.retryAfterMs(h({}))).toBeUndefined();
    expect(__test.parseArgs("{bad")).toEqual({ __invalid_json: "{bad" });
  });
});
