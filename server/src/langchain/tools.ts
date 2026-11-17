import type { StructuredToolInterface } from "@langchain/core/tools";
import { toJsonSchema } from "@langchain/core/utils/json_schema";
import { z } from "zod";
import { defineTool } from "../core/define.js";
import { ToolError, type ToolContext, type ToolDef } from "../core/types.js";

export interface FromLangChainOptions {
  /** Rename the tool for the model (defaults to the LangChain tool's name). */
  name?: string;
  description?: string;
  /**
   * How the tool touches the world. LangChain tools cannot join Keel's step
   * transaction, so only "read" (may re-run after a crash) and "external"
   * (at least once, with a stable idempotency key) are allowed.
   */
  effect?: "read" | "external";
  needsApproval?: boolean | ((input: Record<string, unknown>) => boolean);
  maxOutputChars?: number;
}

/**
 * The Keel context a LangChain tool receives, as `config.configurable.keel`.
 * Gives the tool the run id, the idempotency key for this call, and the pool.
 */
export type KeelToolConfig = { configurable?: { keel?: ToolContext } };

export function keelContext(config: unknown): ToolContext {
  const ctx = (config as KeelToolConfig | undefined)?.configurable?.keel;
  if (!ctx) throw new Error("this tool must run inside Keel (no config.configurable.keel)");
  return ctx;
}

/**
 * Use any LangChain tool (a `tool(...)`, a `DynamicStructuredTool`, a community
 * integration) as a durable Keel tool. Keel records its result in the run's step
 * log, so a crash after the call never runs it again unless its effect allows it,
 * and approvals, budgets and forking work exactly as for native tools.
 */
export function fromLangChainTool(tool: StructuredToolInterface, opts: FromLangChainOptions = {}): ToolDef<Record<string, unknown>> {
  const parameters = toJsonSchema(tool.schema as Parameters<typeof toJsonSchema>[0]) as Record<string, unknown>;
  delete parameters.$schema;
  return defineTool<Record<string, unknown>>({
    name: opts.name ?? tool.name,
    description: opts.description ?? tool.description,
    effect: opts.effect ?? "read",
    needsApproval: opts.needsApproval,
    maxOutputChars: opts.maxOutputChars,
    // The LangChain tool validates its own arguments against its schema.
    input: z.record(z.string(), z.unknown()),
    parameters,
    async run(ctx, input) {
      try {
        const out: unknown = await tool.invoke(input, {
          signal: ctx.signal,
          configurable: { keel: ctx },
          metadata: { keel_run_id: ctx.runId, keel_call_id: ctx.callId },
        });
        return toolOutput(out);
      } catch (err) {
        // Bad arguments are the model's mistake: show it the message so it can retry.
        const e = err as Error | undefined;
        if (e?.name === "ToolInputParsingException" || /did not match expected schema/.test(e?.message ?? "")) throw new ToolError(`Invalid arguments: ${(err as Error).message}`);
        throw err;
      }
    },
  });
}

function toolOutput(out: unknown): unknown {
  // Tools invoked with plain arguments return their content; tolerate a ToolMessage too.
  if (out && typeof out === "object" && "content" in out && (out as { _getType?: unknown })._getType) return (out as { content: unknown }).content;
  return out;
}
