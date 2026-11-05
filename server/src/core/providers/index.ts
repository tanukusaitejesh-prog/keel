import type { Message, ModelProvider, ModelRequest, ModelTurn, ToolCall } from "../types.js";
import { AnthropicProvider } from "./anthropic.js";
import { OpenAICompatibleProvider } from "./openai.js";

export type ProviderResolver = (modelId: string) => ModelProvider;

const OPENAI_COMPATIBLE: Record<string, { baseURL?: string; keyEnv: string }> = {
  groq: { baseURL: "https://api.groq.com/openai/v1", keyEnv: "GROQ_API_KEY" },
  openai: { keyEnv: "OPENAI_API_KEY" },
};

/** Resolve "provider:model" ids from environment configuration. */
export function envProviders(extra: Record<string, ProviderResolver> = {}): ProviderResolver {
  const cache = new Map<string, ModelProvider>();
  return (id) => {
    const hit = cache.get(id);
    if (hit) return hit;
    const i = id.indexOf(":");
    const provider = id.slice(0, i);
    const model = id.slice(i + 1);
    let p: ModelProvider;
    if (extra[provider]) p = extra[provider](model);
    else if (provider === "anthropic") p = new AnthropicProvider(model);
    else if (OPENAI_COMPATIBLE[provider]) {
      const cfg = OPENAI_COMPATIBLE[provider];
      const apiKey = process.env[cfg.keyEnv];
      if (!apiKey) throw new Error(`${cfg.keyEnv} is not set (needed for ${id})`);
      p = new OpenAICompatibleProvider(provider, model, { apiKey, baseURL: cfg.baseURL });
    } else throw new Error(`unknown model provider "${provider}" in "${id}"`);
    cache.set(id, p);
    return p;
  };
}

/**
 * Deterministic provider for tests and offline demos: a policy function maps
 * the conversation so far to the next turn.
 */
export type Policy = (messages: Message[], req: ModelRequest) => { text?: string; calls?: { name: string; args: unknown }[] } | Promise<{ text?: string; calls?: { name: string; args: unknown }[] }>;

export class ScriptedProvider implements ModelProvider {
  readonly provider = "scripted";
  constructor(
    readonly model: string,
    private policy: Policy,
    private latencyMs = 0,
  ) {}
  async complete(req: ModelRequest): Promise<ModelTurn> {
    if (this.latencyMs) await new Promise((r) => setTimeout(r, this.latencyMs));
    const out = await this.policy(req.messages, req);
    const turn = req.messages.filter((m) => m.role === "assistant").length;
    const toolCalls: ToolCall[] = (out.calls ?? []).map((c, i) => ({ id: `call_${turn}_${i}`, name: c.name, args: c.args }));
    const tokens = Math.ceil(JSON.stringify(req.messages).length / 4);
    return {
      text: out.text ?? "",
      toolCalls,
      stop: toolCalls.length ? "tool_calls" : "end",
      usage: { input: tokens, output: 20 },
      model: this.model,
      provider: this.provider,
    };
  }
}

/** Helpers for writing policies. */
export const conv = {
  toolResults(messages: Message[]) {
    return messages.filter((m): m is Extract<Message, { role: "tool" }> => m.role === "tool");
  },
  lastResults(messages: Message[]) {
    const out: Extract<Message, { role: "tool" }>[] = [];
    for (let i = messages.length - 1; i >= 0 && messages[i]!.role === "tool"; i--) out.unshift(messages[i] as Extract<Message, { role: "tool" }>);
    return out;
  },
  turns(messages: Message[]) {
    return messages.filter((m) => m.role === "assistant").length;
  },
};
