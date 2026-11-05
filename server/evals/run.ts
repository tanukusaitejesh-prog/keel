/**
 * Evaluate an agent against real models.
 *
 *   npm run eval -w server -- analyst [--limit 5] [--model groq:openai/gpt-oss-20b]
 *   npm run eval -w server -- research
 *
 * Analyst answers are graded against ground-truth SQL computed live from the
 * same database, so grading can never drift from the data. Results are saved
 * to evals/results/ as JSON.
 */
import "../src/env.js";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createPool, migrate, type Db } from "../src/db.js";
import { registry } from "../src/agents/index.js";
import { closeAnalystPool } from "../src/agents/analyst.js";
import { envProviders } from "../src/core/providers/index.js";
import { envLimits } from "../src/core/ratelimit.js";
import { startRun, TERMINAL } from "../src/core/runtime.js";
import { startWorker } from "../src/core/worker.js";

const here = path.dirname(fileURLToPath(import.meta.url));


type Expect = { kind: "number"; value: number; tol: number } | { kind: "text"; value: string };
interface Case {
  id: string;
  tier?: "basic" | "hard";
  question: string;
  expect: Expect | ((db: Db) => Promise<Expect>);
}

const num = (sql: string, tol = 0) => async (db: Db): Promise<Expect> => ({ kind: "number", value: Number((await db.query(sql)).rows[0].v), tol });
/** Argmax questions: fail loudly if the top two are tied, since then there is no single right answer. */
const top = (sql: string) => async (db: Db): Promise<Expect> => {
  const rows = (await db.query(sql)).rows;
  if (rows.length > 1 && Number(rows[0].score) === Number(rows[1].score)) throw new Error(`tie in ground truth: ${sql}`);
  return { kind: "text", value: String(rows[0].v) };
};
const REV = `sum(i.quantity * i.unit_price)`;
const COMPLETED = `shop.orders o JOIN shop.order_items i ON i.order_id = o.id WHERE o.status = 'completed'`;

const ANALYST: Case[] = [
  { id: "customers", question: "How many customers do we have?", expect: num(`SELECT count(*) v FROM shop.customers`) },
  { id: "orders-mar26", question: "How many orders were placed in March 2026, of any status?", expect: num(`SELECT count(*) v FROM shop.orders WHERE order_date >= '2026-03-01' AND order_date < '2026-04-01'`) },
  { id: "revenue-2025", question: "What was our total revenue in 2025? Round to 2 decimals.", expect: num(`SELECT round(${REV}, 2) v FROM ${COMPLETED} AND o.order_date >= '2025-01-01' AND o.order_date < '2026-01-01'`, 0.011) },
  { id: "top-category-q1", question: "Which product category generated the most revenue in Q1 2026?", expect: top(`SELECT p.category v, ${REV} score FROM ${COMPLETED} AND o.order_date >= '2026-01-01' AND o.order_date < '2026-04-01' JOIN shop.products p ON p.id = i.product_id GROUP BY 1 ORDER BY 2 DESC LIMIT 2`.replace(`WHERE o.status = 'completed' AND o.order_date >= '2026-01-01' AND o.order_date < '2026-04-01' JOIN shop.products p ON p.id = i.product_id`, `JOIN shop.products p ON p.id = i.product_id WHERE o.status = 'completed' AND o.order_date >= '2026-01-01' AND o.order_date < '2026-04-01'`)) },
  { id: "top-customer", question: "Who is our top customer by lifetime revenue? Give their name.", expect: top(`SELECT c.name v, ${REV} score FROM shop.orders o JOIN shop.order_items i ON i.order_id = o.id JOIN shop.customers c ON c.id = o.customer_id WHERE o.status = 'completed' GROUP BY c.id, c.name ORDER BY 2 DESC LIMIT 2`) },
  { id: "return-rate-2025", question: "What percentage of orders placed in 2025 were returned? Give one decimal place.", expect: num(`SELECT round(100.0 * count(*) FILTER (WHERE status = 'returned') / count(*), 1) v FROM shop.orders WHERE order_date >= '2025-01-01' AND order_date < '2026-01-01'`, 0.051) },
  { id: "top-country", question: "Which country has the most customers?", expect: top(`SELECT country v, count(*) score FROM shop.customers GROUP BY 1 ORDER BY 2 DESC LIMIT 2`) },
  { id: "aov-h1-2026", question: "What was the average order value of completed orders in the first half of 2026? Round to 2 decimals.", expect: num(`SELECT round(${REV} / count(DISTINCT o.id), 2) v FROM ${COMPLETED} AND o.order_date >= '2026-01-01' AND o.order_date < '2026-07-01'`, 0.011) },
  { id: "peak-month-2025", question: "Which month of 2025 had the most completed orders? Answer as YYYY-MM.", expect: top(`SELECT to_char(order_date, 'YYYY-MM') v, count(*) score FROM shop.orders WHERE status = 'completed' AND order_date >= '2025-01-01' AND order_date < '2026-01-01' GROUP BY 1 ORDER BY 2 DESC LIMIT 2`) },
  { id: "best-product-2025", question: "Which product sold the most units in completed orders during 2025? Give the product name.", expect: top(`SELECT p.name v, sum(i.quantity) score FROM shop.orders o JOIN shop.order_items i ON i.order_id = o.id JOIN shop.products p ON p.id = i.product_id WHERE o.status = 'completed' AND o.order_date >= '2025-01-01' AND o.order_date < '2026-01-01' GROUP BY p.id, p.name ORDER BY 2 DESC LIMIT 2`) },
  { id: "both-years", question: "How many customers placed at least one order in 2025 and at least one order in 2026 (any status)?", expect: num(`SELECT count(*) v FROM (SELECT customer_id FROM shop.orders GROUP BY 1 HAVING bool_or(extract(year FROM order_date) = 2025) AND bool_or(extract(year FROM order_date) = 2026)) t`) },
  { id: "business-channel", question: "Which sales channel brought in the most revenue from business-segment customers?", expect: top(`SELECT o.channel v, ${REV} score FROM shop.orders o JOIN shop.order_items i ON i.order_id = o.id JOIN shop.customers c ON c.id = o.customer_id WHERE o.status = 'completed' AND c.segment = 'business' GROUP BY 1 ORDER BY 2 DESC LIMIT 2`) },
  { id: "growth-q1", question: "By what percentage did revenue grow from Q1 2025 to Q1 2026? Give one decimal place.", expect: num(`SELECT round(100.0 * (sum(i.quantity * i.unit_price) FILTER (WHERE o.order_date >= '2026-01-01') / sum(i.quantity * i.unit_price) FILTER (WHERE o.order_date < '2025-04-01') - 1), 1) v FROM ${COMPLETED} AND ((o.order_date >= '2025-01-01' AND o.order_date < '2025-04-01') OR (o.order_date >= '2026-01-01' AND o.order_date < '2026-04-01'))`, 0.051) },
  { id: "never-ordered", question: "How many products have never been ordered?", expect: num(`SELECT count(*) v FROM shop.products p WHERE NOT EXISTS (SELECT 1 FROM shop.order_items i WHERE i.product_id = p.id)`) },
  { id: "signup-no-order", question: "How many customers who signed up in 2025 have never placed an order?", expect: num(`SELECT count(*) v FROM shop.customers c WHERE c.signup_date >= '2025-01-01' AND c.signup_date < '2026-01-01' AND NOT EXISTS (SELECT 1 FROM shop.orders o WHERE o.customer_id = c.id)`) },
];

const C25 = `o.status = 'completed' AND o.order_date >= '2025-01-01' AND o.order_date < '2026-01-01'`;
const HARD: Case[] = [
  { id: "top10-share", tier: "hard", question: "What share of 2025 revenue came from the top 10% of customers by their 2025 revenue (round the number of customers up to a whole number)? Give a percentage with one decimal place.",
    expect: num(`WITH c AS (SELECT o.customer_id, sum(i.quantity * i.unit_price) rev FROM shop.orders o JOIN shop.order_items i ON i.order_id = o.id WHERE ${C25} GROUP BY 1),
                 r AS (SELECT rev, row_number() OVER (ORDER BY rev DESC) rn, count(*) OVER () n FROM c)
            SELECT round(100 * sum(rev) FILTER (WHERE rn <= ceil(n * 0.1)) / sum(rev), 1) v FROM r`, 0.051) },
  { id: "q1-cohort-90d", tier: "hard", question: "Consider customers whose first order (any status) was placed in Q1 2025. What percentage of them placed another order (any status) on a later date at most 90 days after their first order? One decimal place.",
    expect: num(`WITH f AS (SELECT customer_id, min(order_date) AS first FROM shop.orders GROUP BY 1),
                 q AS (SELECT * FROM f WHERE first >= '2025-01-01' AND first < '2025-04-01')
            SELECT round(100.0 * count(*) FILTER (WHERE EXISTS (SELECT 1 FROM shop.orders o WHERE o.customer_id = q.customer_id AND o.order_date > q.first AND o.order_date <= q.first + 90)) / count(*), 1) v FROM q`, 0.051) },
  { id: "best-mom-2026", tier: "hard", question: "Between January and June 2026, which month had the largest percentage increase in revenue compared with the month before it? (January is compared with December 2025.) Answer as YYYY-MM.",
    expect: top(`WITH m AS (SELECT date_trunc('month', o.order_date) m, sum(i.quantity * i.unit_price) rev FROM shop.orders o JOIN shop.order_items i ON i.order_id = o.id
                           WHERE o.status = 'completed' AND o.order_date >= '2025-12-01' AND o.order_date < '2026-07-01' GROUP BY 1),
                 g AS (SELECT to_char(m, 'YYYY-MM') v, rev / lag(rev) OVER (ORDER BY m) - 1 score FROM m)
            SELECT v, score FROM g WHERE score IS NOT NULL ORDER BY score DESC LIMIT 2`) },
  { id: "median-order-2025", tier: "hard", question: "What was the median order value of completed orders placed in 2025? Round to 2 decimals.",
    expect: num(`SELECT round(percentile_cont(0.5) WITHIN GROUP (ORDER BY v)::numeric, 2) v FROM (SELECT sum(i.quantity * i.unit_price) v FROM shop.orders o JOIN shop.order_items i ON i.order_id = o.id WHERE ${C25} GROUP BY o.id) t`, 0.011) },
  { id: "discount-cost-2025", tier: "hard", question: "How much did discounts cost us in 2025? That is the total of (product list price minus the price actually charged) times quantity, over completed orders. Round to 2 decimals.",
    expect: num(`SELECT round(sum((p.price - i.unit_price) * i.quantity), 2) v FROM shop.orders o JOIN shop.order_items i ON i.order_id = o.id JOIN shop.products p ON p.id = i.product_id WHERE ${C25}`, 0.011) },
  { id: "country-rev-per-customer", tier: "hard", question: "Which country had the highest 2025 revenue per customer, counting only customers who placed at least one completed order in 2025?",
    expect: top(`SELECT c.country v, sum(i.quantity * i.unit_price) / count(DISTINCT c.id) score FROM shop.orders o JOIN shop.order_items i ON i.order_id = o.id JOIN shop.customers c ON c.id = o.customer_id WHERE ${C25} GROUP BY 1 ORDER BY 2 DESC LIMIT 2`) },
  { id: "longest-gap", tier: "hard", question: "Which customer had the longest gap in days between two consecutive orders of theirs (any status)? Give the customer's name.",
    expect: top(`WITH g AS (SELECT customer_id, order_date - lag(order_date) OVER (PARTITION BY customer_id ORDER BY order_date, id) gap FROM shop.orders)
            SELECT c.name v, max(g.gap) score FROM g JOIN shop.customers c ON c.id = g.customer_id WHERE g.gap IS NOT NULL GROUP BY c.id, c.name ORDER BY 2 DESC LIMIT 2`) },
  { id: "returned-electronics", tier: "hard", question: "What percentage of the orders returned in 2025 contained at least one Electronics item? One decimal place.",
    expect: num(`SELECT round(100.0 * count(*) FILTER (WHERE EXISTS (SELECT 1 FROM shop.order_items i JOIN shop.products p ON p.id = i.product_id WHERE i.order_id = o.id AND p.category = 'Electronics')) / count(*), 1) v
            FROM shop.orders o WHERE o.status = 'returned' AND o.order_date >= '2025-01-01' AND o.order_date < '2026-01-01'`, 0.051) },
];

const RESEARCH: Case[] = [
  { id: "lovelace-turing", question: "How many years apart were Ada Lovelace and Alan Turing born?", expect: { kind: "number", value: 97, tol: 0 } },
  { id: "kilimanjaro-montblanc", question: "Which is taller, Mount Kilimanjaro or Mont Blanc?", expect: { kind: "text", value: "Kilimanjaro" } },
  { id: "harvard-tokyo", question: "Which was founded earlier, Harvard University or the University of Tokyo? Give the founding year of the older one.", expect: { kind: "number", value: 1636, tol: 0 } },
  { id: "curie-einstein", question: "Who was older when they received their first Nobel Prize: Marie Curie or Albert Einstein?", expect: { kind: "text", value: "Einstein" } },
  { id: "moonwalker-death", question: "In what year did the first person to walk on the Moon die?", expect: { kind: "number", value: 2012, tol: 0 } },
  { id: "austen-melville", question: "How many years passed between the first publication of Pride and Prejudice and of Moby-Dick?", expect: { kind: "number", value: 38, tol: 0 } },
  { id: "best-picture-70", question: "Who directed the film that won Best Picture at the 70th Academy Awards?", expect: { kind: "text", value: "Cameron" } },
  { id: "lovelace-father", question: "Ada Lovelace's father was a famous poet. In what year did he die?", expect: { kind: "number", value: 1824, tol: 0 } },
];

function answerLine(text: string): string {
  const m = [...text.matchAll(/answer\s*:\s*(.+)/gi)].at(-1);
  return (m?.[1] ?? text).trim();
}

export function grade(text: string, e: Expect): boolean {
  const a = answerLine(text);
  if (e.kind === "text") return a.toLowerCase().includes(e.value.toLowerCase());
  const nums = [...a.replace(/(\d),(\d{3})/g, "$1$2").matchAll(/-?\d+(?:\.\d+)?/g)].map((m) => Number(m[0]));
  return nums.some((n) => Math.abs(n - e.value) <= e.tol);
}

async function main() {
  const which = process.argv[2] ?? "analyst";
  const limitArg = process.argv.indexOf("--limit");
  const modelArg = process.argv.indexOf("--model");
  const limit = limitArg > 0 ? Number(process.argv[limitArg + 1]) : Infinity;
  const model = modelArg > 0 ? process.argv[modelArg + 1] : undefined;
  const suites: Record<string, Case[]> = { analyst: [...ANALYST, ...HARD], "analyst-basic": ANALYST, "analyst-hard": HARD, research: RESEARCH };
  if (!suites[which]) throw new Error(`unknown suite ${which}; one of ${Object.keys(suites).join(", ")}`);
  const cases = suites[which]!.slice(0, limit);
  const agent = which === "research" ? "researcher" : "analyst";
  const tierOf = new Map(cases.map((c) => [c.id, c.tier ?? "basic"]));

  const db = createPool();
  await migrate(db);
  const expected = new Map<string, Expect>();
  for (const c of cases) expected.set(c.id, typeof c.expect === "function" ? await c.expect(db) : c.expect);

  const agents = registry();
  const ids = new Map<string, string>();
  for (const c of cases) ids.set(c.id, await startRun(db, agents, { agent, input: c.question, model }));
  const started = Date.now();
  const worker = startWorker({ db, agents, providers: envProviders(), limits: envLimits(), leaseMs: 60_000, concurrency: 3 });
  let lastOpen = -1;
  for (;;) {
    const open = (await db.query(
      `WITH RECURSIVE tree AS (SELECT id FROM runs WHERE id = ANY($1) UNION ALL SELECT r.id FROM runs r JOIN tree t ON r.parent_id = t.id)
       SELECT count(*)::int AS n FROM runs WHERE id IN (SELECT id FROM tree) AND status <> ALL($2)`,
      [[...ids.values()], TERMINAL],
    )).rows[0].n;
    if (open !== lastOpen) console.log(`[eval] ${open} run(s) open, ${Math.round((Date.now() - started) / 1000)}s`);
    lastOpen = open;
    if (open === 0) break;
    await new Promise((r) => setTimeout(r, 2000));
  }
  await worker.stop();

  const results: { id: string; question: string; expected: number | string; answer: string; pass: boolean; status: string; error: string | null; turns: number; toolCalls: number; toolErrors: number; selfCorrections: number; subAgents: number; tokens: number; seconds: number | null; rateLimitWaits: number; runId: string }[] = [];
  for (const c of cases) {
    const id = ids.get(c.id)!;
    const run = (await db.query(`SELECT * FROM runs WHERE id = $1`, [id])).rows[0];
    const steps = (await db.query(`SELECT kind, name, payload FROM steps WHERE run_id = $1 ORDER BY seq`, [id])).rows;
    const kids = (await db.query(`SELECT input_tokens + output_tokens AS t, (SELECT count(*) FROM steps s WHERE s.run_id = runs.id AND s.kind = 'model') AS turns FROM runs WHERE parent_id = $1`, [id])).rows;
    const toolSteps = steps.filter((s) => s.kind === "tool");
    // A self-correction: a failed run_sql later followed by a successful one.
    let selfCorrections = 0;
    let sawError = false;
    for (const s of toolSteps.filter((s) => s.name === "run_sql")) {
      if (s.payload.isError) sawError = true;
      else if (sawError) {
        selfCorrections++;
        sawError = false;
      }
    }
    const e = expected.get(c.id)!;
    const text = run.result ?? "";
    results.push({
      id: c.id,
      question: c.question,
      expected: e.value,
      answer: answerLine(text),
      pass: run.status === "succeeded" && grade(text, e),
      status: run.status,
      error: run.status === "succeeded" ? null : run.error,
      turns: steps.filter((s) => s.kind === "model").length,
      toolCalls: toolSteps.length,
      toolErrors: toolSteps.filter((s) => s.payload.isError).length,
      selfCorrections,
      subAgents: kids.length,
      tokens: run.input_tokens + run.output_tokens + kids.reduce((n: number, k: { t: number }) => n + Number(k.t), 0),
      seconds: run.finished_at ? (new Date(run.finished_at).getTime() - new Date(run.created_at).getTime()) / 1000 : null,
      rateLimitWaits: run.rate_limit_waits,
      runId: id,
    });
  }
  const passed = results.filter((r) => r.pass).length;
  const sum = (k: "tokens" | "turns" | "toolCalls" | "selfCorrections" | "toolErrors" | "rateLimitWaits") => results.reduce((n, r) => n + Number(r[k]), 0);
  const summary = {
    agent,
    model: model ?? agents.get(agent).model,
    cases: results.length,
    passed,
    passRate: Number((passed / results.length).toFixed(3)),
    totalTokens: sum("tokens"),
    avgTokens: Math.round(sum("tokens") / results.length),
    avgTurns: Number((sum("turns") / results.length).toFixed(2)),
    toolCalls: sum("toolCalls"),
    toolErrors: sum("toolErrors"),
    selfCorrections: sum("selfCorrections"),
    rateLimitWaits: sum("rateLimitWaits"),
    byTier: Object.fromEntries(
      [...new Set(results.map((r) => tierOf.get(r.id)!))].map((t) => {
        const rs = results.filter((r) => tierOf.get(r.id) === t);
        return [t, { passed: rs.filter((r) => r.pass).length, cases: rs.length }];
      }),
    ),
    wallSeconds: Math.round((Date.now() - started) / 1000),
    at: new Date().toISOString(),
  };
  console.table(results.map((r) => ({ id: r.id, pass: r.pass ? "PASS" : "FAIL", expected: r.expected, answer: r.answer.slice(0, 50), turns: r.turns, tools: r.toolCalls, fixes: r.selfCorrections, tokens: r.tokens })));
  console.log(JSON.stringify(summary, null, 2));
  const dir = path.join(here, "results");
  mkdirSync(dir, { recursive: true });
  const file = path.join(dir, `${agent}-${summary.at.replace(/[:.]/g, "-")}.json`);
  writeFileSync(file, JSON.stringify({ summary, results }, null, 2));
  console.log(`saved ${path.relative(process.cwd(), file)}`);
  await closeAnalystPool();
  await db.end();
}

if (import.meta.url === `file:///${process.argv[1]?.replace(/\\/g, "/")}` || process.argv[1]?.endsWith("run.ts")) void main();
