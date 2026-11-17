import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage, ToolMessage, type BaseMessage } from "@langchain/core/messages";
import type { ChatResult } from "@langchain/core/outputs";
import { tool } from "@langchain/core/tools";
import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { z } from "zod";
import { evaluate } from "../src/agents/calculator.js";
import { calculator } from "../src/agents/calculator.js";
import { docsAgent, docsStore, ingestRepoDocs } from "../src/agents/docs.js";
import { AgentRegistry, defineAgent, toolSpec } from "../src/core/define.js";
import { conv, envProviders, ScriptedProvider, type Policy } from "../src/core/providers/index.js";
import { SimulatedCrash, startRun } from "../src/core/runtime.js";
import type { Db } from "../src/db.js";
import { graphState, graphTool } from "../src/langchain/graph.js";
import { LangChainProvider, langChainResolver } from "../src/langchain/model.js";
import { fromLangChainTool, keelContext } from "../src/langchain/tools.js";
import { HashingEmbeddings } from "../src/rag/embeddings.js";
import { formatPassages, retrievalGraph } from "../src/rag/graph.js";
import { PgVectorStore, splitText } from "../src/rag/store.js";
import { freshDb, reset, settle } from "./helpers.js";

let db: Db;
beforeAll(async () => {
  db = await freshDb();
});
beforeEach(async () => reset(db));
afterAll(async () => db.end());

/** A LangChain chat model driven by a script, so tests need no API key. */
class ScriptedChat extends BaseChatModel {
  calls: { messages: BaseMessage[]; tools?: unknown[] }[] = [];
  boundTools?: unknown[];
  constructor(private script: (messages: BaseMessage[], n: number) => AIMessage | Error) {
    super({});
  }
  _llmType() {
    return "scripted-chat";
  }
  override bindTools(tools: unknown[]): any {
    const c = new ScriptedChat(this.script);
    c.calls = this.calls;
    c.boundTools = tools;
    return c;
  }
  async _generate(messages: BaseMessage[]): Promise<ChatResult> {
    this.calls.push({ messages, tools: this.boundTools });
    const out = this.script(messages, this.calls.length - 1);
    if (out instanceof Error) throw out;
    return { generations: [{ message: out, text: typeof out.content === "string" ? out.content : "" }] };
  }
}

const runOf = async (id: string) => (await db.query(`SELECT * FROM runs WHERE id = $1`, [id])).rows[0];
const stepsOf = async (id: string) => (await db.query(`SELECT kind, name, payload FROM steps WHERE run_id = $1 ORDER BY seq`, [id])).rows;

// ---------------------------------------------------------------------------
describe("LangChain tools in Keel", () => {
  const whoAmI = tool(async (_args, config) => `run ${keelContext(config).runId} key ${keelContext(config).idempotencyKey}`, {
    name: "who_am_i",
    description: "report the Keel run",
    schema: z.object({}),
  });

  it("exposes the LangChain tool's schema to the model", () => {
    const spec = toolSpec(fromLangChainTool(calculator));
    expect(spec.name).toBe("calculate");
    expect(spec.parameters).toMatchObject({ type: "object", properties: { expression: { type: "string" } }, required: ["expression"] });
  });

  it("runs LangChain tools as durable steps, passes Keel context, and reports bad arguments to the model", async () => {
    const policy: Policy = (m) => {
      const t = conv.turns(m);
      if (t === 0) return { calls: [{ name: "calculate", args: { expression: "(2 + 3) * 4" } }, { name: "who_am_i", args: {} }] };
      if (t === 1) return { calls: [{ name: "calculate", args: { nope: 1 } }] };
      return { text: conv.toolResults(m).map((r) => `${r.isError ? "ERR" : "ok"}: ${r.content}`).join(" | ") };
    };
    const agents = new AgentRegistry([
      defineAgent({ name: "lc_tools", description: "", model: "scripted:x", system: "s", tools: [fromLangChainTool(calculator), fromLangChainTool(whoAmI)] }),
    ]);
    const id = await startRun(db, agents, { agent: "lc_tools", input: "go" });
    await settle(db, [id], { agents, providers: envProviders({ scripted: (m) => new ScriptedProvider(m, policy) }) });
    const run = await runOf(id);
    expect(run.status).toBe("succeeded");
    expect(run.result).toContain("ok: (2 + 3) * 4 = 20");
    expect(run.result).toContain(`ok: run ${id} key ${id}:call_0_1`);
    expect(run.result).toMatch(/ERR: Invalid arguments/);
    expect((await stepsOf(id)).filter((s) => s.kind === "tool")).toHaveLength(3);
  });

  it("safe calculator handles precedence, unary minus and right-associative powers, and rejects code", () => {
    expect(evaluate("2 + 3 * 4")).toBe(14);
    expect(evaluate("-(2 + 3) ^ 2")).toBe(-25);
    expect(evaluate("2 ^ 3 ^ 2")).toBe(512);
    expect(evaluate("1,250 / 4 % 7")).toBe(4.5);
    expect(() => evaluate("process.exit(1)")).toThrow();
    expect(() => evaluate("1 / 0")).toThrow(/division by zero/);
  });
});

// ---------------------------------------------------------------------------
describe("LangChain chat models in Keel", () => {
  it("converts the conversation both ways and keeps Keel's loop, log and token accounting", async () => {
    const chat = new ScriptedChat((msgs, n) => {
      if (n === 0)
        return new AIMessage({ content: "", tool_calls: [{ id: "c1", name: "calculate", args: { expression: "6*7" }, type: "tool_call" }], usage_metadata: { input_tokens: 50, output_tokens: 5, total_tokens: 55 } });
      const tm = msgs.find((m) => m instanceof ToolMessage) as ToolMessage;
      return new AIMessage({ content: `answer from tool: ${tm.content}`, usage_metadata: { input_tokens: 80, output_tokens: 9, total_tokens: 89 } });
    });
    const agents = new AgentRegistry([defineAgent({ name: "lc_model", description: "", model: "langchain:fake/m1", system: "be exact", tools: [fromLangChainTool(calculator)] })]);
    const id = await startRun(db, agents, { agent: "lc_model", input: "what is 6*7?" });
    await settle(db, [id], { agents, providers: envProviders({ langchain: langChainResolver({ fake: () => chat }) }) });
    const run = await runOf(id);
    expect(run.status).toBe("succeeded");
    expect(run.result).toBe("answer from tool: 6*7 = 42");
    expect([run.input_tokens, run.output_tokens]).toEqual([130, 14]);
    // The model saw the system prompt, the question, its own tool call and the tool result, with tools bound.
    const second = chat.calls[1]!;
    expect(second.messages.map((m) => m.getType())).toEqual(["system", "human", "ai", "tool"]);
    expect(second.tools).toHaveLength(1);
    const turns = (await stepsOf(id)).filter((s) => s.kind === "model");
    expect(turns[0].payload.turn.provider).toBe("langchain:fake");
  });

  it("a 429 from a LangChain model parks the run instead of failing it", async () => {
    const chat = new ScriptedChat((_m, n) => (n === 0 ? Object.assign(new Error("Too Many Requests"), { status: 429, headers: { "retry-after": "0.3" } }) : new AIMessage("fine now")));
    const agents = new AgentRegistry([defineAgent({ name: "lc_429", description: "", model: "langchain:fake/m", system: "s", tools: [] })]);
    const id = await startRun(db, agents, { agent: "lc_429", input: "hi" });
    await settle(db, [id], { agents, providers: envProviders({ langchain: langChainResolver({ fake: () => chat }) }) });
    const run = await runOf(id);
    expect(run.status).toBe("succeeded");
    expect(run.rate_limit_waits).toBe(1);
  });

  it("maps invalid tool calls so the model gets a chance to fix its JSON", async () => {
    const p = new LangChainProvider("m", new ScriptedChat(() => new AIMessage({ content: "", invalid_tool_calls: [{ id: "x", name: "calculate", args: "{oops", type: "invalid_tool_call" }] })));
    const turn = await p.complete({ system: "s", messages: [{ role: "user", content: "q" }], tools: [], maxOutputTokens: 100 });
    expect(turn.stop).toBe("tool_calls");
    expect(turn.toolCalls[0]!.args).toEqual({ __invalid_json: "{oops" });
  });
});

// ---------------------------------------------------------------------------
describe("LangGraph graphs as durable Keel tools", () => {
  const counts = new Map<string, number>();
  let crashInB = false;
  const bump = (k: string) => counts.set(k, (counts.get(k) ?? 0) + 1);
  const State = Annotation.Root({ x: Annotation<number>({ reducer: (_a, b) => b, default: () => 0 }), path: Annotation<string[]>({ reducer: (a, b) => [...a, ...b], default: () => [] }) });
  const pipeline = new StateGraph(State)
    .addNode("a", async (s) => (bump("a"), { x: s.x + 1, path: ["a"] }))
    .addNode("b", async (s) => {
      bump("b");
      if (crashInB) {
        crashInB = false;
        throw new SimulatedCrash("worker died inside node b");
      }
      return { x: s.x * 10, path: ["b"] };
    })
    .addNode("c", async (s) => (bump("c"), { x: s.x + 2, path: ["c"] }))
    .addEdge(START, "a")
    .addEdge("a", "b")
    .addEdge("b", "c")
    .addEdge("c", END);
  const pipe = graphTool({
    name: "pipeline",
    description: "three-step pipeline",
    input: z.object({ x: z.number() }),
    graph: pipeline,
    output: (s) => `x=${s.x} via ${(s.path as string[]).join(">")}`,
  });
  const agents = new AgentRegistry([defineAgent({ name: "grapher", description: "", model: "scripted:g", system: "s", tools: [pipe] })]);
  const policy: Policy = (m) => (conv.turns(m) === 0 ? { calls: [{ name: "pipeline", args: { x: 4 } }] } : { text: conv.toolResults(m)[0]!.content });
  const providers = envProviders({ scripted: (m) => new ScriptedProvider(m, policy) });

  beforeEach(() => {
    counts.clear();
    crashInB = false;
  });

  it("runs the graph and returns its final state", async () => {
    const id = await startRun(db, agents, { agent: "grapher", input: "go" });
    await settle(db, [id], { agents, providers });
    expect((await runOf(id)).result).toBe("x=52 via a>b>c");
    expect(Object.fromEntries(counts)).toEqual({ a: 1, b: 1, c: 1 });
  });

  it("a worker dying mid-graph resumes after the last finished node: finished nodes never re-run", async () => {
    crashInB = true;
    const id = await startRun(db, agents, { agent: "grapher", input: "go" });
    await settle(db, [id], { agents, providers, leaseMs: 400 });
    const run = await runOf(id);
    expect(run.status).toBe("succeeded");
    expect(run.result).toBe("x=52 via a>b>c");
    expect(run.recoveries).toBeGreaterThanOrEqual(1);
    expect(Object.fromEntries(counts)).toEqual({ a: 1, b: 2, c: 1 }); // b was interrupted once, a never repeated
    const snap = await graphState(db, pipeline, `${id}:call_0_0`);
    expect(snap.next).toEqual([]);
  });

  it("a crash after the graph finished but before Keel recorded the step does not run the graph again", async () => {
    let crashed = false;
    const id = await startRun(db, agents, { agent: "grapher", input: "go" });
    await settle(db, [id], {
      agents,
      providers,
      leaseMs: 400,
      fault: (p) => {
        if (p === "tool:before-commit" && !crashed) {
          crashed = true;
          throw new SimulatedCrash("died before commit");
        }
      },
    });
    expect(crashed).toBe(true);
    expect((await runOf(id)).result).toBe("x=52 via a>b>c");
    expect(Object.fromEntries(counts)).toEqual({ a: 1, b: 1, c: 1 });
  });
});

// ---------------------------------------------------------------------------
describe("pgvector knowledge base", () => {
  const emb = new HashingEmbeddings(512);
  const store = () => new PgVectorStore(emb, { db, collection: "test-kb", dims: 512, embedder: "hashing-512" });

  beforeEach(async () => {
    await store().deleteCollection();
  });

  it("uses pgvector when the server has it (CI requires it), and the portable fallback otherwise", async () => {
    const backend = await store().ensureSchema();
    expect(["pgvector", "array"]).toContain(backend);
    if (process.env.EXPECT_PGVECTOR) expect(backend).toBe("pgvector");
  });

  it("splits markdown with headings and finds the right passage for a paraphrased question", async () => {
    const s = store();
    const docs = await splitText("README.md", (await import("node:fs")).readFileSync(new URL("../../README.md", import.meta.url), "utf8"));
    expect(docs.length).toBeGreaterThan(5);
    expect(docs.some((d) => d.metadata.heading === "Architecture")).toBe(true);
    await s.addDocuments(docs);
    const top = await s.similaritySearchWithScore("what happens when a worker process dies in the middle of a run", 3);
    expect(top.map(([d]) => d.pageContent).join(" ")).toMatch(/lease expires and another worker reclaims/);
    expect(top[0]![1]).toBeGreaterThan(top[2]![1] - 1e-9);
  });

  it("re-ingesting is idempotent and source filters work", async () => {
    const s = store();
    const docs = [
      { source: "a.md", text: "# Alpha\n\nPostgres row locks fence stale workers." },
      { source: "b.md", text: "# Beta\n\nToken buckets share a rate limit across workers." },
    ];
    for (let i = 0; i < 2; i++) for (const d of docs) await s.addDocuments(await splitText(d.source, d.text));
    expect(await s.count()).toBe(2);
    const hits = await s.similaritySearch("rate limit", 5, { source: "a.md" });
    expect(hits.every((h) => h.metadata.source === "a.md")).toBe(true);
  });

  it("corrective retrieval graph: answers well-posed questions first time, rewrites and finally gives up on hopeless ones", async () => {
    const s = store();
    await s.addDocuments(await splitText("README.md", (await import("node:fs")).readFileSync(new URL("../../README.md", import.meta.url), "utf8")));
    const app = retrievalGraph({ store: () => s }).compile();
    const good = await app.invoke({ question: "How does Keel handle a provider 429 rate limit?" });
    expect(good.verdict).toBe("good");
    expect(good.attempts).toBe(1);
    expect(formatPassages(good)).toMatch(/\[README\.md#\d+\]/);
    const bad = await app.invoke({ question: "zxqv quokka marmalade" });
    expect(bad.verdict).toBe("give_up");
    expect(bad.attempts).toBe(3);
    expect(new Set(bad.tried).size).toBe(3); // each retry searched with a different query
    expect(formatPassages(bad)).toMatch(/docs do not cover it/);
  });
});

// ---------------------------------------------------------------------------
describe("docs agent", () => {
  const providersFor = (policy: Policy) => envProviders({ scripted: (m) => new ScriptedProvider(m, policy) });
  const agents = new AgentRegistry([docsAgent]);

  beforeEach(async () => {
    await docsStore(db).deleteCollection();
    await ingestRepoDocs(db);
  });

  it("searches with the LangChain tool, falls back to the LangGraph pipeline, and cites sources", async () => {
    const policy: Policy = (m) => {
      const t = conv.turns(m);
      if (t === 0) return { calls: [{ name: "search_docs", args: { query: "zzz nothing" } }] };
      if (t === 1) return { calls: [{ name: "deep_search_docs", args: { question: "How many workers were killed in the chaos test?" } }] };
      const ref = /\[(README\.md#\d+)\]/.exec(conv.lastResults(m)[0]!.content)?.[1];
      return { text: `70 workers were killed [${ref}]` };
    };
    const id = await startRun(db, agents, { agent: "docs", input: "How many workers did the chaos test kill?", model: "scripted:docs" });
    await settle(db, [id], { agents, providers: providersFor(policy) });
    const run = await runOf(id);
    expect(run.status).toBe("succeeded");
    expect(run.result).toMatch(/^70 workers were killed \[README\.md#\d+\]$/);
    const deep = (await stepsOf(id)).find((s) => s.name === "deep_search_docs");
    expect(deep.payload.content).toMatch(/Worker processes killed/);
  });

  it("save_note needs approval and is stored exactly once, inside the step's transaction", async () => {
    const note = { title: "Deploy freeze", body: "No deploys on Fridays after 4pm, by order of the platform team." };
    const policy: Policy = (m) => {
      const t = conv.turns(m);
      if (t === 0) return { calls: [{ name: "save_note", args: note }] };
      if (t === 1) return { calls: [{ name: "search_docs", args: { query: "deploys on Fridays" } }] };
      return { text: conv.lastResults(m)[0]!.content.slice(0, 200) };
    };
    const approved = await startRun(db, agents, { agent: "docs", input: "remember this", model: "scripted:docs" });
    await settle(db, [approved], { agents, providers: providersFor(policy), approver: () => ({ decision: "approved" }) });
    expect((await runOf(approved)).result).toContain("[note:deploy-freeze#0]");
    const stored = async () => (await db.query(`SELECT count(*)::int AS n FROM ${docsStore(db).table} WHERE source = 'note:deploy-freeze'`)).rows[0].n;
    expect(await stored()).toBe(1);

    await docsStore(db).deleteCollection();
    const rejected = await startRun(db, agents, { agent: "docs", input: "remember this", model: "scripted:docs" });
    await settle(db, [rejected], { agents, providers: providersFor(policy), approver: () => ({ decision: "rejected", note: "not policy" }) });
    expect(await stored()).toBe(0);
  });
});
