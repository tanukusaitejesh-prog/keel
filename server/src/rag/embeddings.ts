import { Embeddings } from "@langchain/core/embeddings";
import { OpenAIEmbeddings } from "@langchain/openai";

const STOP = new Set(
  "a an and are as at be but by can do does for from has have how i if in into is it its not of on or so than that the their then there these this to was were what when where which who why will with you your".split(" "),
);

/** Lowercase word tokens without stopwords, with a light suffix strip so "crashes"/"crashed" match "crash". */
export function terms(text: string): string[] {
  return (text.toLowerCase().match(/[a-z0-9]+/g) ?? [])
    .filter((w) => w.length > 1 && !STOP.has(w))
    .map(stem);
}

/** Tiny suffix stripper: "crashes"/"crashed"/"crashing" -> "crash", but "chaos", "status", "process" stay intact. */
function stem(w: string): string {
  if (w.length > 5 && /ing$/.test(w)) return w.slice(0, -3);
  if (w.length > 4 && /[^aeiou]ed$/.test(w)) return w.slice(0, -2);
  if (w.length > 4 && /(ch|sh|x|ss)es$/.test(w)) return w.slice(0, -2);
  if (w.length > 3 && /[^su]s$/.test(w) && !/(ss|us|is|os)$/.test(w)) return w.slice(0, -1);
  return w;
}

function fnv1a(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/**
 * Offline, deterministic embeddings: hashed unigrams and bigrams with
 * sublinear term frequency, L2-normalised. A lexical baseline, not a semantic
 * model, but it needs no API key, runs in CI and makes retrieval tests exact.
 * Set EMBEDDINGS_MODEL to use a real embedding model instead.
 */
export class HashingEmbeddings extends Embeddings {
  constructor(readonly dims = 512) {
    super({});
  }
  async embedDocuments(texts: string[]): Promise<number[][]> {
    return texts.map((t) => this.embed(t));
  }
  async embedQuery(text: string): Promise<number[]> {
    return this.embed(text);
  }
  embed(text: string): number[] {
    const v = new Array<number>(this.dims).fill(0);
    const t = terms(text);
    const counts = new Map<string, number>();
    for (let i = 0; i < t.length; i++) {
      counts.set(t[i]!, (counts.get(t[i]!) ?? 0) + 1);
      if (i + 1 < t.length) counts.set(`${t[i]} ${t[i + 1]}`, (counts.get(`${t[i]} ${t[i + 1]}`) ?? 0) + 0.5);
    }
    for (const [f, c] of counts) {
      const h = fnv1a(f);
      v[h % this.dims]! += (h & 0x80000000 ? -1 : 1) * (1 + Math.log(c));
    }
    const norm = Math.hypot(...v) || 1;
    return v.map((x) => x / norm);
  }
}

export interface EmbeddingConfig {
  embeddings: Embeddings;
  dims: number;
  /** Stored with each chunk so vectors from different models are never compared. */
  id: string;
}

/**
 * EMBEDDINGS_MODEL=openai:text-embedding-3-small (with OPENAI_API_KEY, and
 * OPENAI_BASE_URL for any OpenAI-compatible server such as Ollama or vLLM)
 * plus EMBEDDINGS_DIMS. Unset: the offline hashing baseline.
 */
export function embeddingsFromEnv(env = process.env): EmbeddingConfig {
  const spec = env.EMBEDDINGS_MODEL;
  if (!spec) return { embeddings: new HashingEmbeddings(512), dims: 512, id: "hashing-512" };
  const [provider, model] = [spec.slice(0, spec.indexOf(":")), spec.slice(spec.indexOf(":") + 1)];
  if (provider !== "openai" || !model) throw new Error(`EMBEDDINGS_MODEL must look like "openai:<model>", got "${spec}"`);
  const dims = Number(env.EMBEDDINGS_DIMS ?? 1536);
  return {
    embeddings: new OpenAIEmbeddings({ model, dimensions: dims, apiKey: env.OPENAI_API_KEY, configuration: env.OPENAI_BASE_URL ? { baseURL: env.OPENAI_BASE_URL } : undefined }),
    dims,
    id: `${spec}-${dims}`,
  };
}
