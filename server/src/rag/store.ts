import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import { Document, type DocumentInterface } from "@langchain/core/documents";
import { VectorStore } from "@langchain/core/vectorstores";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import type { Db, Tx } from "../db.js";

export type Backend = "pgvector" | "array";

export interface PgVectorStoreOptions {
  db: Db;
  /** Logical knowledge base, e.g. "keel-docs". */
  collection: string;
  dims: number;
  /** Identifies the embedding model; chunks embedded by another model are ignored. */
  embedder: string;
  /** Run writes on this client instead of the pool (e.g. a Keel step transaction). */
  client?: Tx;
}

/**
 * LangChain vector store on Keel's Postgres. Uses pgvector (cosine distance,
 * HNSW index) when the extension is available, and otherwise falls back to a
 * float4[] column with an exact cosine scan in SQL, so it works on any Postgres.
 * Chunks are upserted by (collection, source, chunk): re-ingesting is idempotent.
 */
export class PgVectorStore extends VectorStore {
  declare FilterType: { source?: string };
  backend?: Backend;
  private ready?: Promise<Backend>;

  constructor(
    embeddings: EmbeddingsInterface,
    readonly opts: PgVectorStoreOptions,
  ) {
    super(embeddings, {});
    if (!Number.isInteger(opts.dims) || opts.dims < 1 || opts.dims > 16000) throw new Error(`bad embedding dims ${opts.dims}`);
  }

  _vectorstoreType(): string {
    return "keel-pg";
  }

  get table(): string {
    return `rag_chunks_d${this.opts.dims}`;
  }

  /** The same store, writing through `client` (so ingestion commits with a Keel step). */
  using(client: Tx): PgVectorStore {
    const s = new PgVectorStore(this.embeddings, { ...this.opts, client });
    s.ready = this.ensureSchema();
    return s;
  }

  ensureSchema(): Promise<Backend> {
    this.ready ??= this.setup().then((b) => (this.backend = b));
    this.ready.catch(() => (this.ready = undefined));
    return this.ready;
  }

  private async setup(): Promise<Backend> {
    const c = await this.opts.db.connect();
    try {
      await c.query("SELECT pg_advisory_lock(727103)");
      const existing = await c.query<{ udt_name: string }>(
        `SELECT udt_name FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = $1 AND column_name = 'embedding'`,
        [this.table],
      );
      let backend: Backend;
      if (existing.rows[0]) backend = existing.rows[0].udt_name === "vector" ? "pgvector" : "array";
      else {
        backend = "array";
        try {
          await c.query("CREATE EXTENSION IF NOT EXISTS vector");
          backend = "pgvector";
        } catch {
          /* pgvector not installed on this server: use the portable fallback */
        }
        const col = backend === "pgvector" ? `vector(${this.opts.dims})` : "float4[]";
        await c.query(`CREATE TABLE IF NOT EXISTS ${this.table} (
          id bigserial PRIMARY KEY,
          collection text NOT NULL,
          source text NOT NULL,
          chunk int NOT NULL,
          embedder text NOT NULL,
          content text NOT NULL,
          metadata jsonb NOT NULL DEFAULT '{}',
          embedding ${col} NOT NULL,
          created_at timestamptz NOT NULL DEFAULT now(),
          UNIQUE (collection, source, chunk))`);
        if (backend === "pgvector" && this.opts.dims <= 2000) {
          await c.query(`CREATE INDEX IF NOT EXISTS ${this.table}_hnsw ON ${this.table} USING hnsw (embedding vector_cosine_ops)`);
        }
      }
      if (backend === "array") {
        await c.query(`CREATE OR REPLACE FUNCTION keel_cosine(a float4[], b float4[]) RETURNS float8
          LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
            SELECT coalesce(sum(x * y) / nullif(sqrt(sum(x * x)) * sqrt(sum(y * y)), 0), 0) FROM unnest(a, b) AS t(x, y) $$`);
      }
      return backend;
    } finally {
      await c.query("SELECT pg_advisory_unlock(727103)").catch(() => {});
      c.release();
    }
  }

  private vec(v: number[]): string | number[] {
    if (v.length !== this.opts.dims) throw new Error(`embedding has ${v.length} dims, store expects ${this.opts.dims}`);
    return this.backend === "pgvector" ? `[${v.join(",")}]` : v;
  }

  async addVectors(vectors: number[][], documents: DocumentInterface[]): Promise<string[]> {
    await this.ensureSchema();
    const q = this.opts.client ?? this.opts.db;
    const ids: string[] = [];
    for (const [i, doc] of documents.entries()) {
      const source = String(doc.metadata?.source ?? "unknown");
      const chunk = Number.isInteger(doc.metadata?.chunk) ? Number(doc.metadata.chunk) : i;
      const { rows } = await q.query<{ id: string }>(
        `INSERT INTO ${this.table} (collection, source, chunk, embedder, content, metadata, embedding)
         VALUES ($1, $2, $3, $4, $5, $6, $7${this.backend === "pgvector" ? "::vector" : "::float4[]"})
         ON CONFLICT (collection, source, chunk) DO UPDATE
           SET content = EXCLUDED.content, metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding, embedder = EXCLUDED.embedder
         RETURNING id`,
        [this.opts.collection, source, chunk, this.opts.embedder, doc.pageContent, JSON.stringify(doc.metadata ?? {}), this.vec(vectors[i]!)],
      );
      ids.push(String(rows[0]!.id));
    }
    return ids;
  }

  async addDocuments(documents: DocumentInterface[]): Promise<string[]> {
    return this.addVectors(await this.embeddings.embedDocuments(documents.map((d) => d.pageContent)), documents);
  }

  async similaritySearchVectorWithScore(query: number[], k: number, filter?: this["FilterType"]): Promise<[DocumentInterface, number][]> {
    await this.ensureSchema();
    const q = this.opts.client ?? this.opts.db;
    const score = this.backend === "pgvector" ? `1 - (embedding <=> $1::vector)` : `keel_cosine(embedding, $1::float4[])`;
    const order = this.backend === "pgvector" ? `embedding <=> $1::vector` : `${score} DESC`;
    const { rows } = await q.query<{ content: string; metadata: Record<string, unknown>; source: string; chunk: number; score: number }>(
      `SELECT content, metadata, source, chunk, ${score} AS score FROM ${this.table}
        WHERE collection = $2 AND embedder = $3 AND ($4::text IS NULL OR source = $4)
        ORDER BY ${order} LIMIT $5`,
      [this.vec(query), this.opts.collection, this.opts.embedder, filter?.source ?? null, k],
    );
    return rows.map((r) => [new Document({ pageContent: r.content, metadata: { ...r.metadata, source: r.source, chunk: r.chunk } }), Number(r.score)]);
  }

  async count(): Promise<number> {
    await this.ensureSchema();
    const { rows } = await (this.opts.client ?? this.opts.db).query<{ n: number }>(`SELECT count(*)::int AS n FROM ${this.table} WHERE collection = $1`, [this.opts.collection]);
    return rows[0]!.n;
  }

  async deleteCollection(): Promise<void> {
    await this.ensureSchema();
    await (this.opts.client ?? this.opts.db).query(`DELETE FROM ${this.table} WHERE collection = $1`, [this.opts.collection]);
  }
}

/** Split markdown/text into overlapping chunks, tagged with source, chunk number and nearest heading. */
export async function splitText(source: string, text: string, opts: { chunkSize?: number; chunkOverlap?: number } = {}): Promise<Document[]> {
  const splitter = RecursiveCharacterTextSplitter.fromLanguage("markdown", { chunkSize: opts.chunkSize ?? 900, chunkOverlap: opts.chunkOverlap ?? 120 });
  const parts = await splitter.splitText(text);
  let heading = "";
  let cursor = 0;
  return parts.map((content, chunk) => {
    // Track the most recent markdown heading at or before this chunk.
    const at = text.indexOf(content.slice(0, 80), Math.max(0, cursor - (opts.chunkOverlap ?? 120) - 1));
    if (at >= 0) {
      const before = text.slice(0, at + content.length);
      const hs = [...before.matchAll(/^#{1,4}\s+(.+)$/gm)];
      if (hs.length) heading = hs[hs.length - 1]![1]!.trim();
      cursor = at + content.length;
    }
    // A chunk that starts inside a table has lost the sentence saying what the table is.
    // Carry that sentence over, so the rows can be found (and understood) on their own.
    const lead = /^\s*\|/.test(content) && at > 0 ? tableLead(text, at) : "";
    return new Document({ pageContent: lead ? `(${lead})\n${content}` : content, metadata: { source, chunk, heading } });
  });
}

function tableLead(text: string, at: number): string {
  const lines = text.slice(0, at).split("\n");
  for (let i = lines.length - 1; i >= 0; i--) {
    const l = lines[i]!.trim();
    if (l && !l.startsWith("|")) return l.replace(/[*`]/g, "").slice(0, 240);
  }
  return "";
}

/** Chunk, embed and upsert documents. Removes stale chunks when a document gets shorter. */
export async function ingest(store: PgVectorStore, docs: { source: string; text: string }[]): Promise<number> {
  let n = 0;
  for (const d of docs) {
    const chunks = await splitText(d.source, d.text);
    await store.addDocuments(chunks);
    await (store.opts.client ?? store.opts.db).query(`DELETE FROM ${store.table} WHERE collection = $1 AND source = $2 AND chunk >= $3`, [
      store.opts.collection,
      d.source,
      chunks.length,
    ]);
    n += chunks.length;
  }
  return n;
}
