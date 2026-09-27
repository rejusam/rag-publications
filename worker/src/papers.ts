// Citations shown to users come from this lookup, not from model output,
// so a paper can only be cited if it was actually retrieved.

import rawPapers from "../../papers.json";
import type { Chunk, Source } from "./types.ts";

type RawPaper = Partial<Source> & { title: string };

export const PAPERS: Record<string, Source> = Object.fromEntries(
  Object.entries(rawPapers as Record<string, RawPaper>).map(([key, p]) => [
    key,
    {
      title: p.title,
      authors_short: p.authors_short ?? null,
      year: p.year ?? null,
      journal: p.journal ?? null,
      doi: p.doi ?? null,
    },
  ]),
);

/** Unique papers behind the retrieved chunks, in retrieval order. */
export function resolveSources(chunks: readonly Chunk[], papers: Record<string, Source>): Source[] {
  const seen = new Set<string>();
  const sources: Source[] = [];
  for (const chunk of chunks) {
    const key = chunk.source_file;
    if (!key || seen.has(key)) continue;
    seen.add(key);
    const source = papers[key];
    if (source === undefined) {
      console.warn(`No papers.json entry for source_file=${JSON.stringify(key)}`);
      sources.push({ title: key, authors_short: null, year: null, journal: null, doi: null });
    } else {
      sources.push(source);
    }
  }
  return sources;
}
