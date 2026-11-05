import { z } from "zod";
import type { AgentDef, ToolDef, ToolSpec } from "./types.js";

export function defineTool<I>(def: ToolDef<I>): ToolDef<I> {
  if (def.effect === "agent") {
    if (!def.agent) throw new Error(`tool ${def.name}: effect "agent" needs an agent name`);
  } else if (!def.run) {
    throw new Error(`tool ${def.name}: missing run()`);
  }
  if (!/^[a-zA-Z0-9_-]{1,64}$/.test(def.name)) throw new Error(`tool name "${def.name}" is not a valid function name`);
  return def;
}

/**
 * Expose another agent as a tool. One call takes a list of tasks and fans them
 * out as parallel, durable child runs, so fan-out does not depend on the model
 * emitting several tool calls in one turn (many models, e.g. gpt-oss on Groq, emit one).
 */
export function agentTool(opts: { name: string; description: string; agent: string; maxTasks?: number }): ToolDef<{ tasks: string[] }> {
  return defineTool({
    name: opts.name,
    description: opts.description,
    effect: "agent",
    agent: opts.agent,
    input: z.object({
      tasks: z
        .array(z.string().min(1).max(2000))
        .min(1)
        .max(opts.maxTasks ?? 5)
        .describe("Independent, self-contained tasks. Each runs as its own sub-agent, all in parallel."),
    }),
    toTasks: (i) => i.tasks,
  });
}

export function defineAgent(def: AgentDef): AgentDef {
  const names = new Set<string>();
  for (const t of def.tools) {
    if (names.has(t.name)) throw new Error(`agent ${def.name}: duplicate tool ${t.name}`);
    names.add(t.name);
  }
  if (!def.model.includes(":")) throw new Error(`agent ${def.name}: model must be "provider:model"`);
  return def;
}

export function toolSpec(t: ToolDef): ToolSpec {
  const schema = z.toJSONSchema(t.input, { target: "draft-7", io: "input" }) as Record<string, unknown>;
  delete schema.$schema;
  return { name: t.name, description: t.description, parameters: schema };
}

export class AgentRegistry {
  private agents = new Map<string, AgentDef>();
  constructor(agents: AgentDef[] = []) {
    agents.forEach((a) => this.add(a));
  }
  add(a: AgentDef) {
    if (this.agents.has(a.name)) throw new Error(`duplicate agent ${a.name}`);
    this.agents.set(a.name, a);
    return this;
  }
  get(name: string): AgentDef {
    const a = this.agents.get(name);
    if (!a) throw new Error(`unknown agent "${name}"`);
    return a;
  }
  has(name: string) {
    return this.agents.has(name);
  }
  list(): AgentDef[] {
    return [...this.agents.values()];
  }
}
