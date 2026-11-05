# Keel

**A durable runtime for AI agents.** Define tools and agents in TypeScript; Keel runs them on PostgreSQL so that a worker can be killed at any moment and another one picks the run up without repeating a model call or a side effect. Agents can delegate to sub-agents in parallel, pause for human approval, share a rate limit across every worker, and be forked at any turn to replay them on a different model.

**Stack:** TypeScript, Node.js, PostgreSQL, Fastify, React, Vite, Zod, OpenAI-compatible APIs (Groq) and the Claude API, Vitest, GitHub Actions.

```ts
const runSql = defineTool({
  name: "run_sql",
  description: "Run one read-only SELECT against the shop schema.",
  effect: "read",                       // read | write | external | agent
  input: z.object({ sql: z.string() }),
  run: async (ctx, { sql }) => query(sql),
});

const shareReport = defineTool({
  name: "share_report",
  effect: "write",                      // commits with the step: exactly once
  needsApproval: true,                  // run pauses until a human decides
  input: z.object({ title: z.string(), body: z.string() }),
  run: async ({ tx }, r) => tx!.query("INSERT INTO outbox ...", [r]),
});

export const analyst = defineAgent({
  name: "analyst",
  model: "groq:openai/gpt-oss-120b",    // or anthropic:claude-opus-5-5
  system: "You are a careful data analyst...",
  tools: [describeSchema, runSql, shareReport],
  maxTurns: 12, maxTokens: 60_000,
});

// Multi-agent: another agent exposed as a tool. Calls fan out as child runs.
agentTool({ name: "delegate_research", agent: "wiki_researcher", description: "..." });
```

## Architecture

```
 React dashboard ──REST──► Fastify API ──► PostgreSQL ◄──── worker processes (any number)
   live trace, approvals,       │          runs        (queue, leases, budgets, parent/child)
   fork, cancel                 │          steps       (append-only event log per run)
        ▲                       │          approvals, external_calls, outbox, rate_buckets
        └──── Server-Sent Events ◄── LISTEN/NOTIFY
```

A worker claims a run (`FOR UPDATE SKIP LOCKED`, with a lease renewed by a heartbeat), rebuilds the conversation from the run's step log, and takes exactly one step: a model turn or one tool call. Each step commits in a single transaction together with its effects, so the log is always a faithful record of what happened.

| Guarantee | How |
|---|---|
| **Crash recovery** | If a worker dies, its lease expires and another worker reclaims the run, replaying recorded model turns instead of calling the model again. |
| **Fencing** | Every claim bumps `lease_token`. Every write first locks the run row and checks the token, so a stalled worker that lost its lease can never commit. |
| **Side-effect semantics** | `read` tools may re-run after a crash. `write` tools run inside the step's transaction (savepoint per call), so they happen **exactly once**. `external` tools log their intent first and always receive the same idempotency key (`runId:callId`), so retries are **at least once** and safe to deduplicate remotely. |
| **Multi-agent** | An `agent` tool spawns child runs for every delegation in the turn in one transaction (parallel fan-out). The parent releases its worker and sleeps in `waiting`; a finishing child wakes it. Depth limit, failure propagation, and cancel-the-whole-tree are built in. |
| **Human in the loop** | `needsApproval` (a boolean or a per-call predicate) pauses the run in `awaiting_approval` without holding a worker. Rejection notes go back to the model as the tool result. |
| **Shared rate limits** | A token bucket per `provider:model` in Postgres, shared by all workers. If it is empty, the run is parked (`not_before`) rather than blocking a worker. A provider 429 drains the bucket and parks the run until `retry-after`. |
| **Retries and budgets** | Deadlocks, dropped connections, provider 5xx, malformed tool calls and empty completions are retried with exponential backoff (max 5); everything else fails the run with the message. Per-run turn and token budgets. |
| **Time travel** | `fork(run, atTurn, { model })` copies the log up to a model turn into a new run and continues from there, so you can replay a bad decision on another model or prompt without re-running earlier side effects. |
| **Model-agnostic** | A provider-neutral message format with adapters for OpenAI-compatible APIs (Groq, OpenAI, vLLM...) and Claude. Claude's own content blocks are replayed unchanged. |

## Example agents

**`analyst`: SQL data analyst.** Answers business questions over a synthetic e-commerce database (400 customers, 3,156 orders, 5,021 line items). Business definitions live in the database as column comments, which the agent reads with `describe_schema`. Queries run in a sandbox with three independent guards: a read-only transaction, a separate login role with `SELECT`-only grants on one schema, and the extended query protocol (no multi-statement strings), plus a 5 s statement timeout. `scripts/sandbox-check.ts` throws 12 attacks at it in CI, among them `COMMIT; DROP ...`, privilege escalation via `set_config('role', ...)`, file reads and password-table access.

**`researcher`: multi-agent Wikipedia research.** An orchestrator (gpt-oss-120b) splits a question into facts and delegates them to `wiki_researcher` sub-agents (gpt-oss-20b) that search and read Wikipedia and cite their sources; the orchestrator combines the findings and does the arithmetic.

<!-- EVALS -->

## Running it

Requires Node.js 22+ and PostgreSQL 14+ (`docker compose up -d` starts one).

```bash
npm install
cp .env.example .env          # set DATABASE_URL and GROQ_API_KEY (free at console.groq.com)
npm run seed                  # schema + sample shop database + analyst role
npm run api                   # http://127.0.0.1:3002
npm run worker                # start as many as you like
npm run web                   # http://localhost:5174
```

`npm test` runs the test suite against `TEST_DATABASE_URL`.
