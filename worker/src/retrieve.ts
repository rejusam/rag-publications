import type { Chunk, Env } from "./types.ts";

export const EMBEDDING_MODEL = "@cf/baai/bge-small-en-v1.5";
export const RETRIEVER_K = 4;

export async function retrieve(
  env: Pick<Env, "AI" | "VECTORIZE">,
  question: string,
  topK: number = RETRIEVER_K,
): Promise<Chunk[]> {
  const embedded = (await env.AI.run(EMBEDDING_MODEL, { text: [question] })) as { data?: number[][] };
  const vector = embedded.data?.[0];
  if (!vector || vector.length === 0) throw new Error("embedding returned no vector");

  const result = await env.VECTORIZE.query(vector, { topK, returnMetadata: "all" });
  const chunks: Chunk[] = [];
  for (const match of result.matches) {
    const meta = (match.metadata ?? {}) as Record<string, unknown>;
    if (typeof meta.text !== "string" || !meta.text) continue;
    chunks.push({
      text: meta.text,
      source_file: typeof meta.source_file === "string" ? meta.source_file : null,
    });
  }
  return chunks;
}
