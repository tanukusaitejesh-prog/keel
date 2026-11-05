import type { z } from "zod";
import type { Db, Tx } from "../db.js";

// ---------------------------------------------------------------------------
// Provider-neutral conversation. Adapters translate to each model API.
// ---------------------------------------------------------------------------
export interface ToolCall {
  id: string;
  name: string;
  /** Parsed JSON arguments, or { __invalid_json: raw } if the model emitted bad JSON. */
  args: unknown;
}

export type Message =
  | { role: "user"; content: string }
  | { role: "assistant"; content: string; toolCalls: ToolCall[]; provider?: string; raw?: unknown }
  | { role: "tool"; callId: string; name: string; content: string; isError: boolean };

export type StopReason = "end" | "tool_calls" | "length" | "refusal" | "other";

/** One model turn, stored verbatim in the step log and replayed on resume. */
export interface ModelTurn {
  text: string;
  toolCalls: ToolCall[];
  stop: StopReason;
  usage: { input: number; output: number };
  model: string;
  provider: string;
  /** Provider-native content, replayed unchanged to the same provider (e.g. Claude thinking blocks). */
  raw?: unknown;
}

export interface ToolSpec {
  name: string;
  description: string;
  parameters: Record<string, unknown>; // JSON Schema
}

export interface ModelRequest {
  system: string;
  messages: Message[];
  tools: ToolSpec[];
  maxOutputTokens: number;
  reasoningEffort?: "low" | "medium" | "high";
  signal?: AbortSignal;
}

export interface ModelProvider {
  readonly provider: string;
  readonly model: string;
  complete(req: ModelRequest): Promise<ModelTurn>;
}

/** Thrown by adapters when the provider says "slow down". The run is parked, not failed. */
export class RateLimited extends Error {
  constructor(public retryAfterMs: number, message: string) {
    super(message);
  }
}

/** A failure worth retrying later (overload, network, malformed tool call from the model). */
export class TransientModelError extends Error {}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

/**
 * How a tool touches the world decides what the runtime can guarantee:
 *  - read:     no side effects. May re-run after a crash.
 *  - write:    runs inside the transaction that records the step. Exactly once.
 *  - external: calls something outside Postgres. At least once, always with the
 *              same idempotency key, so the remote side can deduplicate.
 *  - agent:    delegates to a child agent run; the parent sleeps until it ends.
 */
export type Effect = "read" | "write" | "external" | "agent";

export interface ToolContext {
  runId: string;
  callId: string;
  /** Stable across retries of this call: `${runId}:${callId}`. */
  idempotencyKey: string;
  db: Db;
  /** Only for effect "write": the transaction that also records this step. */
  tx?: Tx;
  signal: AbortSignal;
}

export interface ToolDef<I = any> {
  name: string;
  description: string;
  input: z.ZodType<I>;
  effect: Effect;
  /** Pause for human approval before running. A function decides per call. */
  needsApproval?: boolean | ((input: I) => boolean);
  /** Results longer than this are truncated before the model sees them (saves tokens). */
  maxOutputChars?: number;
  run?: (ctx: ToolContext, input: I) => Promise<unknown>;
  /** For effect "agent": the agent to delegate to, and the tasks (one child run each) for a call. */
  agent?: string;
  toTasks?: (input: I) => string[];
}

/** A recoverable tool failure: the model sees the message and can try again. */
export class ToolError extends Error {}

// ---------------------------------------------------------------------------
// Agents
// ---------------------------------------------------------------------------
export interface AgentDef {
  name: string;
  description: string;
  /** provider:model, e.g. "groq:openai/gpt-oss-120b" or "anthropic:claude-opus-5-5". */
  model: string;
  system: string;
  tools: ToolDef[];
  maxTurns?: number;
  maxTokens?: number;
  maxOutputTokens?: number;
  reasoningEffort?: "low" | "medium" | "high";
}
