import Anthropic from "@anthropic-ai/sdk";
import { RateLimited, TransientModelError, type Message, type ModelProvider, type ModelRequest, type ModelTurn, type StopReason } from "../types.js";

/** Claude via the official SDK. Assistant turns are replayed with their original content blocks. */
export class AnthropicProvider implements ModelProvider {
  readonly provider = "anthropic";
  private client = new Anthropic({ maxRetries: 0 });
  constructor(readonly model: string) {}

  async complete(req: ModelRequest): Promise<ModelTurn> {
    let res: Anthropic.Message;
    try {
      res = await this.client.messages.create(
        {
          model: this.model,
          max_tokens: Math.max(req.maxOutputTokens, 4096),
          output_config: { effort: req.reasoningEffort ?? "medium" },
          cache_control: { type: "ephemeral" },
          system: req.system,
          tools: req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters as Anthropic.Tool.InputSchema })),
          messages: toAnthropic(req.messages),
        },
        { signal: req.signal },
      );
    } catch (err) {
      if (err instanceof Anthropic.RateLimitError) {
        const ra = Number(err.headers?.get("retry-after"));
        throw new RateLimited(Number.isFinite(ra) && ra > 0 ? ra * 1000 : 20_000, err.message);
      }
      if (err instanceof Anthropic.APIConnectionError) throw new TransientModelError(err.message);
      if (err instanceof Anthropic.APIError && typeof err.status === "number" && err.status >= 500) throw new TransientModelError(err.message);
      throw err;
    }
    const stop: StopReason =
      res.stop_reason === "tool_use" ? "tool_calls" : res.stop_reason === "end_turn" || res.stop_reason === "stop_sequence" ? "end" : res.stop_reason === "max_tokens" ? "length" : res.stop_reason === "refusal" ? "refusal" : "other";
    return {
      text: res.content.flatMap((b) => (b.type === "text" ? [b.text] : [])).join("\n"),
      toolCalls: res.content.flatMap((b) => (b.type === "tool_use" ? [{ id: b.id, name: b.name, args: b.input }] : [])),
      stop,
      usage: { input: res.usage.input_tokens, output: res.usage.output_tokens },
      model: res.model,
      provider: this.provider,
      raw: res.content,
    };
  }
}

function toAnthropic(messages: Message[]): Anthropic.MessageParam[] {
  const out: Anthropic.MessageParam[] = [];
  for (const m of messages) {
    if (m.role === "user") out.push({ role: "user", content: m.content });
    else if (m.role === "assistant") {
      // Same provider: pass the original blocks back unchanged (keeps thinking blocks valid).
      const content =
        m.provider === "anthropic" && Array.isArray(m.raw)
          ? (m.raw as Anthropic.ContentBlockParam[])
          : [
              ...(m.content ? [{ type: "text" as const, text: m.content }] : []),
              ...m.toolCalls.map((c) => ({ type: "tool_use" as const, id: c.id, name: c.name, input: c.args as Record<string, unknown> })),
            ];
      out.push({ role: "assistant", content });
    } else {
      const block: Anthropic.ToolResultBlockParam = { type: "tool_result", tool_use_id: m.callId, content: m.content, ...(m.isError ? { is_error: true } : {}) };
      const prev = out.at(-1);
      // All results for one assistant turn go back in a single user message.
      if (prev?.role === "user" && Array.isArray(prev.content) && prev.content.every((b) => b.type === "tool_result")) prev.content.push(block);
      else out.push({ role: "user", content: [block] });
    }
  }
  return out;
}
