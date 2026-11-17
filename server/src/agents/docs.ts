import { tool } from "@langchain/core/tools";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import type { Db } from "../db.js";
import { defineAgent, defineTool } from "../core/define.js";
import { ToolError } from "../core/types.js";
import { graphTool } from "../langchain/graph.js";
import { fromLangChainTool, keelContext } from "../langchain/tools.js";
import { embeddingsFromEnv } from "../rag/embeddings.js";
import { formatPassages, retrievalGraph, storeFromKeel } from "../rag/graph.js";
import { ingest, PgVectorStore } from "../rag/store.js";
import { calculator } from "./calculator.js";

export const DOCS_COLLECTION = "keel-docs";

const stores = new WeakMap<Db, PgVectorStore>();
/** The knowledge base the docs agent searches, on the same Postgres as the runs. */
export function docsStore(db: Db): PgVectorStore {
  let s = stores.get(db);
  if (!s) {
    const e = embeddingsFromEnv();
    stores.set(db, (s = new PgVectorStore(e.embeddings, { db, collection: DOCS_COLLECTION, dims: e.dims, embedder: e.id })));
  }
  return s;
}

/** Load the repository's own documentation into the knowledge base. */
export async function ingestRepoDocs(db: Db): Promise<number> {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
  return ingest(docsStore(db), [{ source: "README.md", text: await readFile(path.join(root, "README.md"), "utf8") }]);
}

// A plain LangChain tool. It reaches Keel's context through config.configurable.keel.
const searchDocsLc = tool(
  async ({ query, k }, config) => {
    const hits = await docsStore(keelContext(config).db).similaritySearchWithScore(query, k ?? 5);
    if (!hits.length) return "The knowledge base is empty.";
    return hits.map(([d, s]) => `[${d.metadata.source}#${d.metadata.chunk}] (${d.metadata.heading || "intro"}) score ${s.toFixed(3)}\n${d.pageContent}`).join("\n\n---\n\n");
  },
  {
    name: "search_docs",
    description: "Fast single vector search over the knowledge base. Returns passages tagged [source#chunk] with a similarity score.",
    schema: z.object({ query: z.string().min(2).max(300), k: z.number().int().min(1).max(10).optional() }),
  },
);

export const searchDocs = fromLangChainTool(searchDocsLc, { effect: "read", maxOutputChars: 7000 });

export const deepSearchDocs = graphTool({
  name: "deep_search_docs",
  description:
    "Thorough search: a LangGraph pipeline that retrieves, grades the passages for relevance, rewrites the query and searches again if too few are relevant. Use it when search_docs finds nothing useful or the question has several parts.",
  input: z.object({ question: z.string().min(2).max(500) }),
  graph: retrievalGraph({ store: storeFromKeel(docsStore) }),
  toState: ({ question }) => ({ question }),
  output: formatPassages,
  maxOutputChars: 7000,
});

export const saveNote = defineTool({
  name: "save_note",
  description: "Add a short note to the knowledge base so future questions can find it. A human must approve it first.",
  effect: "write",
  needsApproval: true,
  input: z.object({ title: z.string().min(3).max(120), body: z.string().min(10).max(4000) }),
  async run({ tx, db }, { title, body }) {
    const source = `note:${title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "")}`;
    if (source === "note:") throw new ToolError("title needs letters or digits");
    // Written through the step's transaction: the note commits exactly once, with the step.
    const chunks = await ingest(docsStore(db).using(tx!), [{ source, text: `# ${title}\n\n${body}` }]);
    return { saved: source, chunks };
  },
});

export const docsAgent = defineAgent({
  name: "docs",
  description: "Answers questions from the knowledge base (Keel's own docs by default) with citations, via LangChain tools and a LangGraph retrieval pipeline over pgvector.",
  model: "langchain:groq/openai/gpt-oss-120b",
  system: [
    "You answer questions using ONLY the knowledge base. Never answer from memory.",
    "Start with search_docs. If it returns nothing relevant, or the question has several parts, use deep_search_docs.",
    "Cite every claim with the [source#chunk] tag of the passage it came from.",
    "If the passages do not contain the answer, say plainly that the docs do not cover it. Do not guess.",
    "Use calculate for any arithmetic. Only call save_note when the user explicitly asks you to remember something.",
  ].join("\n"),
  tools: [searchDocs, deepSearchDocs, fromLangChainTool(calculator), saveNote],
  maxTurns: 8,
  maxTokens: 40_000,
  reasoningEffort: "low",
});
