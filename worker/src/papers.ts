// Citations shown to users come from this lookup, not from model output,
// so a paper can only be cited if it was actually retrieved.

import rawPapers from "../../papers.json";
import type { Chunk, CitedSource, Source } from "./types.ts";

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

/** Unique papers behind the retrieved chunks, in retrieval order, with the pages read. */
export function resolveSources(chunks: readonly Chunk[], papers: Record<string, Source>): CitedSource[] {
  const byKey = new Map<string, { source: Source; pages: Set<number> }>();
  for (const chunk of chunks) {
    const key = chunk.source_file;
    if (!key) continue;
    let entry = byKey.get(key);
    if (!entry) {
      let source = papers[key];
      if (source === undefined) {
        console.warn(`No papers.json entry for source_file=${JSON.stringify(key)}`);
        source = { title: key, authors_short: null, year: null, journal: null, doi: null };
      }
      entry = { source, pages: new Set() };
      byKey.set(key, entry);
    }
    if (chunk.page !== null) entry.pages.add(chunk.page);
  }
  return [...byKey.values()].map(({ source, pages }) => ({ ...source, pages: [...pages].sort((a, b) => a - b) }));
}
