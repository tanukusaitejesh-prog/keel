import OpenAI from "openai";
import type { ChatCompletionMessageParam, ChatCompletionTool } from "openai/resources/chat/completions";
import { RateLimited, TransientModelError, type Message, type ModelProvider, type ModelRequest, type ModelTurn, type StopReason, type ToolCall } from "../types.js";

/**
 * Any OpenAI-compatible chat completions endpoint (Groq, OpenAI, Together,
 * vLLM, Ollama...). SDK retries are disabled: the runtime parks the run in
 * Postgres instead, so a worker never sits blocked on a backoff.
 */
export class OpenAICompatibleProvider implements ModelProvider {
  private client: OpenAI;
  constructor(
    readonly provider: string,
    readonly model: string,
    opts: { apiKey: string; baseURL?: string; timeoutMs?: number },
  ) {
    this.client = new OpenAI({ apiKey: opts.apiKey, baseURL: opts.baseURL, maxRetries: 0, timeout: opts.timeoutMs ?? 120_000 });
  }

  async complete(req: ModelRequest): Promise<ModelTurn> {
    const tools: ChatCompletionTool[] = req.tools.map((t) => ({
      type: "function",
      function: { name: t.name, description: t.description, parameters: t.parameters },
    }));
    let res;
    try {
      res = await this.client.chat.completions.create(
        {
          model: this.model,
          messages: [{ role: "system", content: req.system }, ...req.messages.map(toOpenAI)],
          ...(tools.length ? { tools, tool_choice: "auto" as const } : {}),
          max_completion_tokens: req.maxOutputTokens,
          ...(req.reasoningEffort ? { reasoning_effort: req.reasoningEffort } : {}),
        },
        { signal: req.signal },
      );
    } catch (err) {
      throw classify(err);
    }
    const choice = res.choices[0];
    if (!choice) throw new TransientModelError("provider returned no choices");
    const toolCalls: ToolCall[] = (choice.message.tool_calls ?? []).flatMap((c) =>
      c.type === "function" ? [{ id: c.id, name: c.function.name, args: parseArgs(c.function.arguments) }] : [],
    );
    const stop: StopReason =
      choice.finish_reason === "tool_calls" || toolCalls.length
        ? "tool_calls"
        : choice.finish_reason === "stop"
          ? "end"
          : choice.finish_reason === "length"
            ? "length"
            : choice.finish_reason === "content_filter"
              ? "refusal"
              : "other";
    return {
      text: choice.message.content ?? "",
      toolCalls,
      stop,
      usage: { input: res.usage?.prompt_tokens ?? 0, output: res.usage?.completion_tokens ?? 0 },
      model: res.model ?? this.model,
      provider: this.provider,
    };
  }
}

function toOpenAI(m: Message): ChatCompletionMessageParam {
  if (m.role === "user") return { role: "user", content: m.content };
  if (m.role === "tool") return { role: "tool", tool_call_id: m.callId, content: m.content };
  return {
    role: "assistant",
    content: m.content || null,
    ...(m.toolCalls.length
      ? {
          tool_calls: m.toolCalls.map((c) => ({
            id: c.id,
            type: "function" as const,
            function: { name: c.name, arguments: JSON.stringify(c.args ?? {}) },
          })),
        }
      : {}),
  };
}

function parseArgs(raw: string): unknown {
  if (!raw?.trim()) return {};
  try {
    return JSON.parse(raw);
  } catch {
    return { __invalid_json: raw.slice(0, 500) };
  }
}

function classify(err: unknown): unknown {
  if (err instanceof OpenAI.RateLimitError) {
    return new RateLimited(retryAfterMs(err.headers) ?? 20_000, err.message);
  }
  if (err instanceof OpenAI.APIConnectionError) return new TransientModelError(err.message); // includes timeouts
  if (err instanceof OpenAI.InternalServerError) return new TransientModelError(err.message);
  if (err instanceof OpenAI.BadRequestError) {
    // Groq rejects a turn when the model emits a malformed tool call; resampling usually fixes it.
    const code = (err.error as { code?: string } | undefined)?.code ?? (err as { code?: string }).code;
    if (code === "tool_use_failed" || /tool call validation failed|failed to call a function/i.test(err.message)) {
      return new TransientModelError(`model produced an invalid tool call: ${err.message}`);
    }
  }
  return err;
}

function retryAfterMs(headers: Headers | undefined): number | undefined {
  const ra = headers?.get("retry-after");
  if (ra && !Number.isNaN(Number(ra))) return Math.ceil(Number(ra) * 1000);
  // Groq also reports when the token window resets, e.g. "7.66s" or "1m26.4s".
  const reset = headers?.get("x-ratelimit-reset-tokens") ?? headers?.get("x-ratelimit-reset-requests");
  if (reset) {
    const m = /^(?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?(?:([\d.]+)ms)?$/.exec(reset);
    if (m) return Math.ceil(((Number(m[1] ?? 0) * 60 + Number(m[2] ?? 0)) * 60 + Number(m[3] ?? 0)) * 1000 + Number(m[4] ?? 0));
  }
  return undefined;
}

export const __test = { retryAfterMs, parseArgs };
