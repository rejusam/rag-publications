// One-off: embed worker/data/chunks.jsonl with Workers AI and write
// worker/data/vectors.ndjson for `wrangler vectorize insert`.
// Needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN (Workers AI read) in
// the environment. Safe to re-run: ids are stable, so inserts overwrite.
//
//   node scripts/ingest.ts

import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath, URL } from "node:url";

const MODEL = "@cf/baai/bge-small-en-v1.5";
const BATCH = 50;
const DIMENSIONS = 384;

interface ChunkRow { id: string; text: string; source_file: string | null; page: number | null }

const accountId = process.env.CLOUDFLARE_ACCOUNT_ID;
const token = process.env.CLOUDFLARE_API_TOKEN;
if (!accountId || !token) {
  console.error("Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN first.");
  process.exit(1);
}

const rows: ChunkRow[] = readFileSync(fileURLToPath(new URL("../data/chunks.jsonl", import.meta.url)), "utf8")
  .split("\n").filter(Boolean).map((line: string) => JSON.parse(line));

async function embed(texts: string[]): Promise<number[][]> {
  const res = await fetch(`https://api.cloudflare.com/client/v4/accounts/${accountId}/ai/run/${MODEL}`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify({ text: texts }),
  });
  if (!res.ok) throw new Error(`embedding failed: ${res.status} ${await res.text()}`);
  const body = (await res.json()) as { result: { data: number[][] } };
  return body.result.data;
}

const lines: string[] = [];
for (let i = 0; i < rows.length; i += BATCH) {
  const batch = rows.slice(i, i + BATCH);
  const vectors = await embed(batch.map((r) => r.text));
  if (vectors.length !== batch.length) throw new Error(`expected ${batch.length} vectors, got ${vectors.length}`);
  batch.forEach((row, j) => {
    const values = vectors[j]!;
    if (values.length !== DIMENSIONS) throw new Error(`unexpected dimension ${values.length}`);
    const metadata: Record<string, string | number> = { text: row.text };
    if (row.source_file) metadata.source_file = row.source_file;
    if (row.page !== null) metadata.page = row.page;
    lines.push(JSON.stringify({ id: row.id, values, metadata }));
  });
  console.log(`embedded ${Math.min(i + BATCH, rows.length)}/${rows.length}`);
}

writeFileSync(fileURLToPath(new URL("../data/vectors.ndjson", import.meta.url)), lines.join("\n") + "\n");
console.log("\nNext: npx wrangler vectorize insert research-chunks --file=data/vectors.ndjson");
