import { z } from "zod";
import { agentTool, defineAgent, defineTool } from "../core/define.js";
import { ToolError } from "../core/types.js";

const API = "https://en.wikipedia.org/w/api.php";
const UA = "keel-agent/0.1 (open-source agent runtime demo)";

async function wiki(params: Record<string, string>, signal: AbortSignal): Promise<any> {
  const url = `${API}?${new URLSearchParams({ format: "json", formatversion: "2", ...params })}`;
  const res = await fetch(url, { headers: { "user-agent": UA }, signal: AbortSignal.any([signal, AbortSignal.timeout(15_000)]) });
  if (!res.ok) throw new ToolError(`Wikipedia returned HTTP ${res.status}`);
  return res.json();
}

const pageUrl = (title: string) => `https://en.wikipedia.org/wiki/${encodeURIComponent(title.replace(/ /g, "_"))}`;

const wikiSearch = defineTool({
  name: "wiki_search",
  description: "Search English Wikipedia. Returns the top article titles with a short snippet.",
  effect: "read",
  input: z.object({ query: z.string().min(1).max(200) }),
  async run({ signal }, { query }) {
    const j = await wiki({ action: "query", list: "search", srsearch: query, srlimit: "5" }, signal);
    const hits = (j.query?.search ?? []) as { title: string; snippet: string }[];
    if (!hits.length) return "No results. Try different keywords.";
    return hits.map((h) => `${h.title}: ${h.snippet.replace(/<[^>]+>/g, "").replace(/&quot;/g, '"').replace(/&amp;/g, "&")}`).join("\n");
  },
});

const wikiRead = defineTool({
  name: "wiki_read",
  description: "Read a Wikipedia article. Pass `focus` keywords to get the most relevant paragraphs instead of only the introduction.",
  effect: "read",
  maxOutputChars: 2600,
  input: z.object({
    title: z.string().min(1).max(200),
    focus: z.string().max(200).optional().describe("Keywords to look for, e.g. 'born 1815'"),
  }),
  async run({ signal }, { title, focus }) {
    const j = await wiki({ action: "query", prop: "extracts", explaintext: "1", redirects: "1", titles: title }, signal);
    const page = j.query?.pages?.[0];
    if (!page || page.missing || !page.extract) throw new ToolError(`No article titled "${title}". Use wiki_search to find the exact title.`);
    const paras: string[] = page.extract.split(/\n+/).map((p: string) => p.trim()).filter((p: string) => p.length > 40 && !p.startsWith("=="));
    const lead = (paras[0] ?? "").slice(0, 700);
    let body = paras.slice(1, 4).join("\n");
    if (focus) {
      const words = focus.toLowerCase().split(/\W+/).filter((w) => w.length > 2);
      const scored = paras.slice(1).map((p, i) => ({ i, p, s: words.reduce((n, w) => n + (p.toLowerCase().includes(w) ? 1 : 0), 0) }));
      body = scored
        .filter((x) => x.s > 0)
        .sort((a, b) => b.s - a.s || a.i - b.i)
        .slice(0, 4)
        .sort((a, b) => a.i - b.i)
        .map((x) => x.p.slice(0, 600))
        .join("\n");
    }
    return `${page.title} (${pageUrl(page.title)})\n${lead}\n${body || "(no paragraphs matched the focus keywords)"}`;
  },
});

export const wikiResearcher = defineAgent({
  name: "wiki_researcher",
  description: "Finds one specific fact on Wikipedia and reports it with its source.",
  model: "groq:openai/gpt-oss-20b",
  reasoningEffort: "low",
  maxTurns: 8,
  maxTokens: 30_000,
  maxOutputTokens: 1024,
  system: `You research one question using Wikipedia. Search, then read the most relevant article with focus keywords.
Report only what the article says. Reply in at most three sentences: the fact, and the source URL.
If you cannot find it, say so instead of guessing.`,
  tools: [wikiSearch, wikiRead],
});

export const researcher = defineAgent({
  name: "researcher",
  description: "Answers multi-part factual questions by delegating sub-questions to parallel research agents.",
  model: "groq:openai/gpt-oss-120b",
  reasoningEffort: "low",
  maxTurns: 6,
  maxTokens: 40_000,
  maxOutputTokens: 1536,
  system: `You coordinate research. Split the question into the independent facts you need, and call delegate_research once per fact, all in the same turn so they run in parallel.
Each sub-agent only sees its own task, so make every task self-contained.
Then combine the findings, doing any arithmetic yourself, and cite the source URLs. If findings conflict or are missing, say so.
End with a final line in the form "Answer: <value>".`,
  tools: [agentTool({ name: "delegate_research", description: "Ask a research sub-agent to find one fact on Wikipedia.", agent: "wiki_researcher" })],
});
