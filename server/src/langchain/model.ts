import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage, HumanMessage, SystemMessage, ToolMessage, type BaseMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { RateLimited, TransientModelError, type Message, type ModelProvider, type ModelRequest, type ModelTurn, type StopReason, type ToolCall } from "../core/types.js";

/**
 * Run a Keel agent on any LangChain chat model. The model only produces turns;
 * Keel still owns the loop, so every turn is recorded once, replayed after a
 * crash instead of re-requested, rate-limited across workers and forkable.
 *
 * Configure the model's own retries off (`maxRetries: 0`): Keel parks a
 * rate-limited run in Postgres instead of blocking a worker on a backoff.
 */
export class LangChainProvider implements ModelProvider {
  constructor(
    readonly model: string,
    private chat: BaseChatModel,
    readonly provider = "langchain",
  ) {}

  async complete(req: ModelRequest): Promise<ModelTurn> {
    const tools = req.tools.map((t) => ({ type: "function" as const, function: { name: t.name, description: t.description, parameters: t.parameters } }));
    if (tools.length && !this.chat.bindTools) throw new Error(`LangChain model ${this.model} does not support tool calling`);
    const runnable = tools.length ? this.chat.bindTools!(tools) : this.chat;
    let ai: AIMessage;
    try {
      ai = (await runnable.invoke(toLangChain(req), { signal: req.signal })) as AIMessage;
    } catch (err) {
      throw classify(err);
    }
    // Ids must be unique within the run: they key approvals and external idempotency.
    const turn = req.messages.filter((m) => m.role === "assistant").length;
    const toolCalls: ToolCall[] = [
      ...(ai.tool_calls ?? []).map((c, i) => ({ id: c.id || `lc_${turn}_${i}`, name: c.name, args: c.args ?? {} })),
      ...(ai.invalid_tool_calls ?? []).map((c, i) => ({ id: c.id || `lc_${turn}_bad_${i}`, name: c.name ?? "unknown", args: { __invalid_json: String(c.args ?? "").slice(0, 500) } })),
    ];
    const meta = (ai.response_metadata ?? {}) as Record<string, unknown>;
    const finish = String(meta.finish_reason ?? meta.stop_reason ?? "");
    const stop: StopReason = toolCalls.length
      ? "tool_calls"
      : /^(length|max_tokens)$/.test(finish)
        ? "length"
        : /^(content_filter|refusal)$/.test(finish)
          ? "refusal"
          : "end";
    return {
      text: textOf(ai.content),
      toolCalls,
      stop,
      usage: { input: ai.usage_metadata?.input_tokens ?? 0, output: ai.usage_metadata?.output_tokens ?? 0 },
      model: String(meta.model_name ?? meta.model ?? this.model),
      provider: this.provider,
    };
  }
}

export function toLangChain(req: Pick<ModelRequest, "system" | "messages">): BaseMessage[] {
  const out: BaseMessage[] = [new SystemMessage(req.system)];
  for (const m of req.messages) out.push(convert(m));
  return out;
}

function convert(m: Message): BaseMessage {
  if (m.role === "user") return new HumanMessage(m.content);
  if (m.role === "tool") return new ToolMessage({ content: m.content, tool_call_id: m.callId, name: m.name, status: m.isError ? "error" : "success" });
  return new AIMessage({
    content: m.content,
    tool_calls: m.toolCalls.map((c) => ({
      id: c.id,
      name: c.name,
      args: c.args && typeof c.args === "object" && !("__invalid_json" in c.args) ? (c.args as Record<string, unknown>) : {},
      type: "tool_call" as const,
    })),
  });
}

function textOf(content: AIMessage["content"]): string {
  if (typeof content === "string") return content;
  return content
    .map((b) => (typeof b === "string" ? b : b.type === "text" && "text" in b ? String(b.text) : ""))
    .join("");
}

function classify(err: unknown): unknown {
  const e = err as { status?: number; code?: string; name?: string; message?: string; headers?: Record<string, string> | Headers; lc_error_code?: string };
  if (!e || e.name === "AbortError") return err;
  const status = e.status ?? (e as { response?: { status?: number } }).response?.status;
  if (status === 429 || e.lc_error_code === "MODEL_RATE_LIMIT") return new RateLimited(retryAfter(e.headers) ?? 20_000, e.message ?? "rate limited");
  if ((status && status >= 500) || status === 408) return new TransientModelError(e.message ?? `provider error ${status}`);
  if (["ECONNRESET", "ECONNREFUSED", "ETIMEDOUT", "EPIPE", "UND_ERR_SOCKET"].includes(e.code ?? "")) return new TransientModelError(e.message ?? e.code!);
  if (/tool call validation failed|failed to call a function|tool_use_failed/i.test(e.message ?? "")) return new TransientModelError(`model produced an invalid tool call: ${e.message}`);
  return err;
}

function retryAfter(headers: Record<string, string> | Headers | undefined): number | undefined {
  const v = headers instanceof Headers ? headers.get("retry-after") : headers?.["retry-after"];
  return v && !Number.isNaN(Number(v)) ? Math.ceil(Number(v) * 1000) : undefined;
}

/**
 * Chat model factories for `langchain:<backend>/<model>` ids, e.g.
 * `langchain:groq/openai/gpt-oss-120b` or `langchain:openai/gpt-5-mini`.
 * Pass your own to use any LangChain integration (Ollama, Bedrock, Gemini...).
 */
const OPENAI_COMPATIBLE: Record<string, { baseURL?: string; keyEnv: string }> = {
  groq: { baseURL: "https://api.groq.com/openai/v1", keyEnv: "GROQ_API_KEY" },
  openai: { keyEnv: "OPENAI_API_KEY" },
};

/** Resolver for envProviders({ langchain: langChainResolver() }). */
export function langChainResolver(factories: Record<string, (model: string) => BaseChatModel> = {}): (model: string) => ModelProvider {
  return (id) => {
    const i = id.indexOf("/");
    if (i < 0) throw new Error(`langchain model must be "langchain:<backend>/<model>", got "langchain:${id}"`);
    const backend = id.slice(0, i);
    const model = id.slice(i + 1);
    const make = factories[backend] ?? openAICompatible(backend);
    return new LangChainProvider(model, make(model), `langchain:${backend}`);
  };
}

function openAICompatible(backend: string): (model: string) => BaseChatModel {
  const cfg = OPENAI_COMPATIBLE[backend];
  if (!cfg) throw new Error(`no LangChain chat model registered for backend "${backend}"`);
  return (model) => {
    const apiKey = process.env[cfg.keyEnv];
    if (!apiKey) throw new Error(`${cfg.keyEnv} is not set (needed for langchain:${backend}/${model})`);
    return new ChatOpenAI({ model, apiKey, maxRetries: 0, configuration: cfg.baseURL ? { baseURL: cfg.baseURL } : undefined });
  };
}
