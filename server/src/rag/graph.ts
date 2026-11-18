import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { keelContext } from "../langchain/tools.js";
import { terms } from "./embeddings.js";
import type { PgVectorStore } from "./store.js";

export interface Passage {
  ref: string;
  heading: string;
  text: string;
  score: number;
}

const last = <T>(fallback: () => T) => Annotation<T>({ reducer: (_a: T, b: T) => b, default: fallback });

export const RetrievalState = Annotation.Root({
  question: last<string>(() => ""),
  query: last<string>(() => ""),
  attempts: last<number>(() => 0),
  candidates: last<Passage[]>(() => []),
  passages: last<Passage[]>(() => []),
  tried: last<string[]>(() => []),
  verdict: last<"good" | "retry" | "give_up" | "">(() => ""),
});
type S = typeof RetrievalState.State;

export interface RetrievalGraphOptions {
  /** The store for a call; receives the Keel tool context (so it can use ctx.db). */
  store: (config: unknown) => PgVectorStore;
  k?: number;
  /** Minimum cosine score for a passage to count as relevant. */
  minScore?: number;
  /** Passages needed before we stop searching. */
  enough?: number;
  maxAttempts?: number;
}

/**
 * Corrective retrieval as a LangGraph state machine:
 *
 *   retrieve -> grade --good/give_up--> END
 *                 \--retry--> rewrite -> retrieve
 *
 * `grade` keeps passages that clear a score threshold AND share a content word
 * with the question (so a high score on boilerplate does not count). If too few
 * survive, `rewrite` reformulates the query from the question's own keywords
 * plus vocabulary from the best near-miss passage, and searches again, up to
 * `maxAttempts`. Every node is deterministic and checkpointed by Keel.
 */
export function retrievalGraph(opts: RetrievalGraphOptions) {
  const k = opts.k ?? 6;
  const minScore = opts.minScore ?? 0.12;
  const enough = opts.enough ?? 2;
  const maxAttempts = opts.maxAttempts ?? 3;

  const retrieve = async (s: S, config: unknown): Promise<Partial<S>> => {
    const query = s.query || s.question;
    const hits = await opts.store(config).similaritySearchWithScore(query, k);
    const candidates = hits.map(([d, score]) => ({
      ref: `${d.metadata.source}#${d.metadata.chunk}`,
      heading: String(d.metadata.heading ?? ""),
      text: d.pageContent,
      score: Math.round(score * 1000) / 1000,
    }));
    return { query, candidates, attempts: s.attempts + 1, tried: [...s.tried, query] };
  };

  const grade = async (s: S): Promise<Partial<S>> => {
    const want = new Set(terms(s.question));
    const kept = s.candidates.filter((p) => p.score >= minScore && terms(p.text).some((t) => want.has(t)));
    // Keep the best passages seen across attempts, not only the last search.
    const byRef = new Map([...s.passages, ...kept].map((p) => [p.ref, p]));
    const passages = [...byRef.values()].sort((a, b) => b.score - a.score).slice(0, k);
    const verdict = passages.length >= enough ? "good" : s.attempts >= maxAttempts ? "give_up" : "retry";
    return { passages, verdict };
  };

  const rewrite = async (s: S): Promise<Partial<S>> => {
    const keywords = [...new Set(terms(s.question))];
    const nearMiss = s.candidates[0];
    const borrowed = nearMiss ? [...new Set(terms(`${nearMiss.heading} ${nearMiss.text}`))].filter((t) => !keywords.includes(t)).slice(0, 4 * s.attempts) : [];
    let query = [...keywords, ...borrowed].join(" ");
    if (s.tried.includes(query)) query = `${query} ${s.attempts}`;
    return { query };
  };

  return new StateGraph(RetrievalState)
    .addNode("retrieve", retrieve)
    .addNode("grade", grade)
    .addNode("rewrite", rewrite)
    .addEdge(START, "retrieve")
    .addEdge("retrieve", "grade")
    .addConditionalEdges("grade", (s: S) => (s.verdict === "retry" ? "rewrite" : END), ["rewrite", END])
    .addEdge("rewrite", "retrieve");
}

export function formatPassages(state: Record<string, unknown>): string {
  const s = state as Partial<S>;
  const passages = s.passages ?? [];
  if (!passages.length) return `No passage in the knowledge base answers this (searched ${s.attempts ?? 0} time(s): ${(s.tried ?? []).map((q) => `"${q}"`).join(", ")}). Say the docs do not cover it.`;
  return passages.map((p) => `[${p.ref}]${p.heading ? ` (${p.heading})` : ""} score ${p.score}\n${p.text}`).join("\n\n---\n\n");
}

/** Store accessor for graph nodes running inside a Keel tool call. */
export const storeFromKeel = (get: (db: import("../db.js").Db) => PgVectorStore) => (config: unknown) => get(keelContext(config).db);
