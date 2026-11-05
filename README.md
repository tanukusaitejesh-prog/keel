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

## Results

All numbers come from `server/evals/results/*.json` and the scripts in this repo, run on Groq's free tier (8,000 tokens per minute per model).

### Agents on real models

Analyst answers are graded against **ground-truth SQL computed live from the same database**; argmax questions fail loudly if the ground truth has a tie. Reported across every valid run, not the best one:

| Suite | Model | Correct | Avg tokens / question | Avg turns |
|---|---|---|---|---|
| Analyst, basic (15 questions × 2 runs) | gpt-oss-120b | **30 / 30** | ~1,900 | 3.0 |
| Analyst, hard (8 questions × 2 runs): cohorts, window functions, percentiles, list-vs-charged price | gpt-oss-120b | **13 / 16** | ~2,100 | 3.0 |
| Research, multi-agent (8 questions × 2 runs) | 120b orchestrator, 20b sub-agents | **16 / 16** | ~5,700 | 2.4 to 3.0 |

The three hard-tier misses were genuine model errors, each checked by hand: a `LIMIT 1` inside a cohort CTE, a month-over-month calculation that filtered December out before applying `lag()`, and one empty completion (now resampled by the runtime; see below). The model fixed one failing SQL query from the database error message without help.

### What the evals found in the runtime

The evals were worth more for the bugs they surfaced than for the scores:

- **A tool fed the model wrong data.** `run_sql` formatted dates through JavaScript `Date`, so on a machine in IST, `2025-01-01` was shown as `2024-12-31`. The model read quarter labels shifted by one and answered −4.6% instead of +10.2%. Dates are now passed through exactly as Postgres prints them. (That run, 14/15, is excluded from the table above.)
- **Empty completions were accepted as answers.** The model occasionally returned no text and no tool calls; the runtime now treats that as transient and resamples.
- **gpt-oss on Groq emits one tool call per turn**, even when asked for three, so the orchestrator delegated sequentially in all 16 delegations of the first research run. Delegation tools now take a list of `tasks` and the runtime fans them out as parallel child runs, so fan-out no longer depends on the model supporting parallel tool calls. In the second run every question fanned out (8 batch calls, 21 sub-agents) and the orchestrator needed 2.4 turns instead of 3.0. Wall time did not improve on the free tier (206 s vs 168 s on average with all 8 questions running at once), because the shared 8k tokens-per-minute budget, not latency, is the bottleneck there.

### Durability

- **Test suite (22 tests, Vitest, real Postgres):** single-agent and multi-agent flows, approvals, all four tool effects, budgets, rate limiting, 429 handling, retries, cancellation, forking, fencing. The **crash sweep** runs a multi-agent workflow once to count its fault points (over 40), then crashes the worker at each one in turn and requires the final runs, logs and side effects to be identical to the clean run every time.
- **Chaos test with real processes** (`npm run chaos -w server`): worker processes are SIGKILLed every 500 ms while they run 100 top-level runs (half of them fanning out to 3 sub-agents). Two recorded runs:

| | Run 1 | Run 2 (batch fan-out) |
|---|---|---|
| Agent runs (incl. sub-agents) | 250 | 250 |
| Worker processes killed | 70 | 67 |
| Runs recovered after their worker died | 86 | 80 |
| Write-tool side effects (expected 300) | 300 | 300 |
| External calls / deliveries (one retry, same idempotency key) | 50 / 51 | 50 / 51 |
| Gaps in any step log | 0 | 0 |

- **SQL sandbox:** 12 of 12 escape attempts blocked; CI fails if any succeeds. One escape (switching back to the superuser with `set_config('role', ...)`) worked against the first design, which used `SET ROLE` from a privileged session; the analyst now connects as its own unprivileged login role.

Known limitation: the random-crash test uses a 150 ms lease, and once timed out (without any invariant failing) while the machine was also running an eval; it has passed on every run since.

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
