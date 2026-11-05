import { useCallback, useEffect, useRef, useState } from "react";
import { api, type AgentInfo, type Approval, type Child, type RunDetail, type RunSummary, type Status, type Step, type ToolCall } from "./api";

const MODELS = ["groq:openai/gpt-oss-120b", "groq:openai/gpt-oss-20b", "anthropic:claude-opus-5-5"];
const LABEL: Record<Status, string> = {
  queued: "Queued",
  running: "Running",
  awaiting_approval: "Needs approval",
  waiting: "Waiting on sub-agents",
  succeeded: "Succeeded",
  failed: "Failed",
  cancelled: "Cancelled",
};
const EXAMPLES: Record<string, string> = {
  analyst: "Which product category grew fastest from Q1 2025 to Q1 2026, and by how much?",
  researcher: "How many years apart were Ada Lovelace and Alan Turing born?",
  wiki_researcher: "When was the Eiffel Tower completed?",
};

export function App() {
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const selectedRef = useRef(selected);
  selectedRef.current = selected;
  const fail = (e: unknown) => setError(String(e));

  const loadRuns = useCallback(() => api.runs().then(setRuns).catch(fail), []);
  const loadDetail = useCallback((id: string) => api.run(id).then((d) => selectedRef.current === id && setDetail(d)).catch(fail), []);

  useEffect(() => {
    api.agents().then(setAgents).catch(fail);
    void loadRuns();
  }, [loadRuns]);
  useEffect(() => {
    setDetail(null);
    if (selected) void loadDetail(selected);
  }, [selected, loadDetail]);

  // Live: Postgres NOTIFY -> SSE. Refresh whenever any run in view changes.
  useEffect(() => {
    let timer: number | undefined;
    const es = new EventSource("/api/events");
    es.onmessage = () => {
      clearTimeout(timer);
      timer = window.setTimeout(() => {
        void loadRuns();
        if (selectedRef.current) void loadDetail(selectedRef.current);
      }, 150);
    };
    return () => {
      clearTimeout(timer);
      es.close();
    };
  }, [loadRuns, loadDetail]);

  return (
    <div className="app">
      <header>
        <h1>Keel</h1>
        <span className="muted">Durable runtime for AI agents</span>
      </header>
      {error && (
        <div className="banner" role="alert">
          <span>{error}</span>
          <button onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}
      <main>
        <aside>
          <NewRun agents={agents} onStarted={(id) => { setSelected(id); void loadRuns(); }} onError={fail} />
          <h2>Runs</h2>
          {runs.length === 0 && <p className="muted">No runs yet.</p>}
          <ul className="runs">
            {runs.map((r) => (
              <li key={r.id}>
                <button className={r.id === selected ? "run active" : "run"} onClick={() => setSelected(r.id)}>
                  <span className="run-title">{r.input}</span>
                  <span className="run-meta">
                    <Badge status={r.status} />
                    <span className="muted">
                      {r.agent}
                      {r.children ? ` · ${r.children} sub-agents` : ""}
                      {r.forked_from ? " · fork" : ""}
                    </span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </aside>
        <section className="detail">
          {!selected && <p className="muted empty">Start a run, or pick one from the list.</p>}
          {selected && !detail && <p className="muted empty">Loading…</p>}
          {detail && <RunView d={detail} open={setSelected} onError={fail} />}
        </section>
      </main>
    </div>
  );
}

function NewRun({ agents, onStarted, onError }: { agents: AgentInfo[]; onStarted: (id: string) => void; onError: (e: unknown) => void }) {
  const [agent, setAgent] = useState("");
  const [input, setInput] = useState("");
  const [model, setModel] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    if (!agent && agents[0]) {
      setAgent(agents[0].name);
      setInput(EXAMPLES[agents[0].name] ?? "");
    }
  }, [agents, agent]);
  const info = agents.find((a) => a.name === agent);
  return (
    <form
      className="card form"
      onSubmit={async (e) => {
        e.preventDefault();
        setBusy(true);
        try {
          onStarted((await api.start(agent, input, model)).id);
        } catch (err) {
          onError(err);
        } finally {
          setBusy(false);
        }
      }}
    >
      <label>
        Agent
        <select value={agent} onChange={(e) => { setAgent(e.target.value); setInput(EXAMPLES[e.target.value] ?? ""); }}>
          {agents.map((a) => <option key={a.name} value={a.name}>{a.name}</option>)}
        </select>
      </label>
      {info && <p className="muted small">{info.description}</p>}
      <label>
        Task
        <textarea rows={3} value={input} onChange={(e) => setInput(e.target.value)} required />
      </label>
      <label>
        Model
        <select value={model} onChange={(e) => setModel(e.target.value)}>
          <option value="">Agent default{info ? ` (${info.model})` : ""}</option>
          {MODELS.map((m) => <option key={m} value={m}>{m}</option>)}
        </select>
      </label>
      <button className="primary" disabled={busy || !agent || !input.trim()}>{busy ? "Starting…" : "Start run"}</button>
    </form>
  );
}

function Badge({ status }: { status: Status }) {
  return <span className={`badge ${status}`}>{LABEL[status]}</span>;
}

function RunView({ d, open, onError }: { d: RunDetail; open: (id: string) => void; onError: (e: unknown) => void }) {
  const { run, steps, approvals, children, external } = d;
  const turns = steps.filter((s) => s.kind === "model");
  const modelMs = turns.reduce((n, s) => n + (s.latency_ms ?? 0), 0);
  const childByCall = new Map(children.map((c) => [c.parent_call_id, c]));
  const decided = new Map(approvals.map((a) => [a.call_id, a]));
  const attempts = new Map(external.map((e) => [e.idempotency_key.split(":").slice(1).join(":"), e.attempts]));
  const live = !["succeeded", "failed", "cancelled"].includes(run.status);
  return (
    <>
      <div className="run-head">
        <div>
          <p className="muted small">
            {run.agent} · {run.model}
            {run.parent_id && <> · <a onClick={() => open(run.parent_id!)}>parent run</a></>}
            {run.forked_from && <> · forked from <a onClick={() => open(run.forked_from!)}>this run</a> at step {run.forked_at_seq}</>}
          </p>
          <h2>{run.input}</h2>
        </div>
        <div className="head-actions">
          <Badge status={run.status} />
          {live && <button onClick={() => api.cancel(run.id).catch(onError)}>Cancel</button>}
        </div>
      </div>

      <dl className="stats">
        <Stat label="Model turns" value={`${turns.length} / ${run.max_turns}`} />
        <Stat label="Tool calls" value={steps.length - turns.length} />
        <Stat label="Tokens" value={`${(run.input_tokens + run.output_tokens).toLocaleString()}`} hint={`${run.input_tokens.toLocaleString()} in · ${run.output_tokens.toLocaleString()} out`} />
        <Stat label="Model time" value={`${(modelMs / 1000).toFixed(1)}s`} />
        <Stat label="Rate-limit waits" value={run.rate_limit_waits} hint="Times the run was parked because the shared token budget was empty" />
        <Stat label="Claims" value={run.lease_token} hint="Each claim bumps the fencing token: start, resume after approval or sub-agents, or recovery after a crash" />
      </dl>

      {approvals.filter((a) => a.status === "pending").map((a) => <ApprovalCard key={a.call_id} a={a} onError={onError} />)}
      {run.status === "succeeded" && run.result && <div className="result ok"><h3>Answer</h3><p className="pre">{run.result}</p></div>}
      {run.error && <div className="result bad"><h3>{run.status === "failed" ? "Failed" : "Last error"}</h3><p>{run.error}</p></div>}

      <h3>Trace</h3>
      <ol className="trace">
        <li className="step user"><span className="who">Task</span><p className="pre">{run.input}</p></li>
        {steps.map((s) =>
          s.kind === "model" ? (
            <ModelStep key={s.seq} s={s} run={d.run} onForked={open} onError={onError} />
          ) : (
            <ToolStep key={s.seq} s={s} child={childByCall.get(s.call_id!)} approval={decided.get(s.call_id!)} attempts={attempts.get(s.call_id!)} open={open} />
          ),
        )}
        {children.filter((c) => !steps.some((s) => s.call_id === c.parent_call_id)).map((c) => (
          <li key={c.id} className="step tool pending-child">
            <div className="step-head"><code>sub-agent</code><Badge status={c.status} /></div>
            <ChildLink c={c} open={open} />
          </li>
        ))}
        {live && <li className="muted small">{run.status === "queued" && new Date(run.not_before) > new Date() ? `Parked until ${new Date(run.not_before).toLocaleTimeString()} (rate limit)` : "…"}</li>}
      </ol>
    </>
  );
}

function Stat({ label, value, hint }: { label: string; value: string | number; hint?: string }) {
  return (
    <div title={hint}>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function ModelStep({ s, run, onForked, onError }: { s: Step; run: RunDetail["run"]; onForked: (id: string) => void; onError: (e: unknown) => void }) {
  const turn = s.payload.turn as { text: string; toolCalls: ToolCall[]; stop: string; usage: { input: number; output: number }; model: string };
  const [forking, setForking] = useState(false);
  const [model, setModel] = useState(run.model);
  return (
    <li className="step model">
      <div className="step-head">
        <span className="who">Model</span>
        <span className="muted small">{turn.model} · {s.latency_ms ?? "–"} ms · {turn.usage.input + turn.usage.output} tokens · {turn.stop}</span>
        <button className="link" onClick={() => setForking((f) => !f)} title="Replay this run from here, reusing every earlier result">Fork from here</button>
      </div>
      {turn.text && <p className="pre">{turn.text}</p>}
      {turn.toolCalls.length > 0 && <p className="muted small">Calls {turn.toolCalls.map((c) => c.name).join(", ")}</p>}
      {forking && (
        <div className="fork">
          <span className="small">Re-run from this turn on</span>
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            {[...new Set([run.model, ...MODELS])].map((m) => <option key={m}>{m}</option>)}
          </select>
          <button className="primary" onClick={() => api.fork(run.id, s.seq, model).then((r) => onForked(r.id), onError)}>Fork</button>
        </div>
      )}
    </li>
  );
}

function ToolStep({ s, child, approval, attempts, open }: { s: Step; child?: Child; approval?: Approval; attempts?: number; open: (id: string) => void }) {
  const { args, content, isError } = s.payload as { args: unknown; content: string; isError: boolean };
  return (
    <li className={`step tool${isError ? " err" : ""}`}>
      <div className="step-head">
        <code>{s.name}</code>
        {child && <span className="tag">sub-agent</span>}
        {approval && <span className={`tag ${approval.status}`}>{approval.status}</span>}
        {attempts && attempts > 1 && <span className="tag" title="External call retried after a crash with the same idempotency key">{attempts} attempts</span>}
        {isError && <span className="tag rejected">error</span>}
        {s.latency_ms != null && <span className="muted small">{s.latency_ms} ms</span>}
      </div>
      {child && <ChildLink c={child} open={open} />}
      <details>
        <summary>Arguments and result</summary>
        <pre>{JSON.stringify(args, null, 2)}</pre>
        <pre>{content}</pre>
      </details>
    </li>
  );
}

function ChildLink({ c, open }: { c: Child; open: (id: string) => void }) {
  return (
    <p className="small">
      <a onClick={() => open(c.id)}>{c.agent}</a>: {c.input} <span className="muted">({(c.input_tokens + c.output_tokens).toLocaleString()} tokens)</span>
    </p>
  );
}

function ApprovalCard({ a, onError }: { a: Approval; onError: (e: unknown) => void }) {
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const decide = async (d: "approved" | "rejected") => {
    setBusy(true);
    try {
      await api.decide(a.run_id, a.call_id, d, note);
    } catch (e) {
      onError(e);
      setBusy(false);
    }
  };
  return (
    <div className="approval">
      <h3>Approve <code>{a.tool_name}</code>?</h3>
      <p className="muted small">The run is paused and holds no worker until you decide.</p>
      <pre>{JSON.stringify(a.input, null, 2)}</pre>
      <label>
        Note for the agent (optional)
        <input value={note} onChange={(e) => setNote(e.target.value)} placeholder="e.g. shorten the summary" />
      </label>
      <div className="actions">
        <button className="primary" disabled={busy} onClick={() => decide("approved")}>Approve</button>
        <button disabled={busy} onClick={() => decide("rejected")}>Reject</button>
      </div>
    </div>
  );
}
