import type { BaseCheckpointSaver } from "@langchain/langgraph";
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";
import type { z } from "zod";
import type { Db } from "../db.js";
import { defineTool } from "../core/define.js";
import { ToolError, type ToolContext, type ToolDef } from "../core/types.js";

/**
 * LangGraph checkpoints live in the same Postgres as Keel's runs, in their own
 * schema. Setup runs once per pool, serialized across processes.
 */
const savers = new WeakMap<Db, Promise<PostgresSaver>>();
export function checkpointer(db: Db): Promise<PostgresSaver> {
  let p = savers.get(db);
  if (!p) {
    p = (async () => {
      const saver = new PostgresSaver(db, undefined, { schema: "langgraph" });
      const client = await db.connect();
      try {
        await client.query("SELECT pg_advisory_lock(727102)");
        await client.query("CREATE SCHEMA IF NOT EXISTS langgraph");
        await saver.setup();
      } finally {
        await client.query("SELECT pg_advisory_unlock(727102)").catch(() => {});
        client.release();
      }
      return saver;
    })();
    p.catch(() => savers.delete(db));
    savers.set(db, p);
  }
  return p;
}

/** The parts of a LangGraph StateGraph (uncompiled) that Keel needs. */
export interface CompilableGraph {
  compile(args: { checkpointer: BaseCheckpointSaver; name?: string }): unknown;
}

interface GraphSnapshot {
  values: Record<string, unknown>;
  next: string[];
  metadata?: unknown;
  tasks?: { interrupts?: unknown[] }[];
}
interface CompiledGraph {
  invoke(input: unknown, config: Record<string, unknown>): Promise<Record<string, unknown>>;
  getState(config: Record<string, unknown>): Promise<GraphSnapshot>;
}

export interface GraphToolOptions<I> {
  name: string;
  description: string;
  input: z.ZodType<I>;
  /** A LangGraph StateGraph. Keel compiles it with a Postgres checkpointer. */
  graph: CompilableGraph;
  /** Initial graph state for a call (defaults to the tool arguments). */
  toState?: (input: I, ctx: ToolContext) => Record<string, unknown>;
  /** What the model sees (defaults to the final state). */
  output?: (state: Record<string, unknown>) => unknown;
  needsApproval?: boolean | ((input: I) => boolean);
  maxOutputChars?: number;
  recursionLimit?: number;
}

/**
 * Run a LangGraph graph as one durable Keel tool call.
 *
 * The graph's thread id is the call's idempotency key, and LangGraph saves a
 * checkpoint after every node (durability "sync"). So when a worker dies in the
 * middle of a graph, the worker that takes the run over resumes the graph after
 * its last finished node instead of starting again; finished nodes never re-run.
 * If the graph finished but the step was not yet recorded, the saved final state
 * is returned without running anything. Effect is "external": the node that was
 * interrupted runs at least once more, with the same thread id.
 */
export function graphTool<I>(opts: GraphToolOptions<I>): ToolDef<I> {
  const compiled = new WeakMap<BaseCheckpointSaver, CompiledGraph>();
  const app = (saver: BaseCheckpointSaver) => {
    let g = compiled.get(saver);
    if (!g) compiled.set(saver, (g = opts.graph.compile({ checkpointer: saver, name: opts.name }) as CompiledGraph));
    return g;
  };
  return defineTool<I>({
    name: opts.name,
    description: opts.description,
    effect: "external",
    input: opts.input,
    needsApproval: opts.needsApproval,
    maxOutputChars: opts.maxOutputChars,
    async run(ctx, input) {
      const graph = app(await checkpointer(ctx.db));
      const config = {
        configurable: { thread_id: `keel:${ctx.idempotencyKey}`, keel: ctx },
        signal: ctx.signal,
        durability: "sync",
        recursionLimit: opts.recursionLimit ?? 25,
      };
      const snap = await graph.getState(config);
      let state: Record<string, unknown>;
      if (snap.metadata === undefined) state = await graph.invoke(opts.toState ? opts.toState(input, ctx) : (input as Record<string, unknown>), config);
      else if (snap.next.length) state = await graph.invoke(null, config); // resume after the last checkpoint
      else state = snap.values; // already finished before a crash: do not run it again
      if (state && "__interrupt__" in state) {
        throw new ToolError("The graph paused with interrupt(); use the tool's needsApproval for human review in Keel.");
      }
      return opts.output ? opts.output(state) : state;
    },
  });
}

/** Inspect the checkpointed state of a graph call (for dashboards and tests). */
export async function graphState(db: Db, graph: CompilableGraph, idempotencyKey: string): Promise<GraphSnapshot> {
  const saver = await checkpointer(db);
  return (graph.compile({ checkpointer: saver }) as CompiledGraph).getState({ configurable: { thread_id: `keel:${idempotencyKey}` } });
}
