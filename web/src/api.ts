export type Status = "queued" | "running" | "awaiting_approval" | "waiting" | "succeeded" | "failed" | "cancelled";
export interface AgentInfo { name: string; description: string; model: string; tools: { name: string; effect: string; approval: boolean }[] }
export interface RunSummary {
  id: string; agent: string; input: string; model: string; status: Status; input_tokens: number; output_tokens: number;
  created_at: string; finished_at: string | null; forked_from: string | null; children: number; pending_approvals: number;
}
export interface ToolCall { id: string; name: string; args: unknown }
export interface Step { seq: number; kind: "model" | "tool"; call_id: string | null; name: string | null; latency_ms: number | null; created_at: string; payload: any }
export interface Approval { run_id: string; call_id: string; tool_name: string; input: unknown; status: "pending" | "approved" | "rejected"; note: string | null }
export interface Child { id: string; agent: string; input: string; status: Status; parent_call_id: string; input_tokens: number; output_tokens: number; result: string | null; error: string | null }
export interface RunDetail {
  run: RunSummary & { result: string | null; error: string | null; parent_id: string | null; depth: number; forked_at_seq: number | null;
    max_turns: number; max_tokens: number; transient_errors: number; rate_limit_waits: number; lease_token: number; not_before: string };
  steps: Step[]; approvals: Approval[]; children: Child[];
  external: { idempotency_key: string; tool_name: string; attempts: number }[];
  outbox: { call_id: string; channel: string; payload: unknown; sent_at: string | null }[];
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { ...init, headers: init?.body ? { "content-type": "application/json" } : undefined });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json() as Promise<T>;
}
const post = <T,>(path: string, body: unknown) => req<T>(path, { method: "POST", body: JSON.stringify(body) });

export const api = {
  agents: () => req<AgentInfo[]>("/api/agents"),
  runs: () => req<RunSummary[]>("/api/runs"),
  run: (id: string) => req<RunDetail>(`/api/runs/${id}`),
  start: (agent: string, input: string, model?: string) => post<{ id: string }>("/api/runs", { agent, input, model: model || undefined }),
  decide: (id: string, callId: string, decision: "approved" | "rejected", note?: string) =>
    post<{ ok: true }>(`/api/runs/${id}/approvals/${encodeURIComponent(callId)}`, { decision, note: note || undefined }),
  cancel: (id: string) => post<{ cancelled: number }>(`/api/runs/${id}/cancel`, {}),
  fork: (id: string, atSeq: number, model?: string) => post<{ id: string }>(`/api/runs/${id}/fork`, { atSeq, model: model || undefined }),
};
