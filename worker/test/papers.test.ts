import { describe, expect, it, vi } from "vitest";
import { PAPERS, resolveSources } from "../src/papers.ts";
import type { Chunk, Source } from "../src/types.ts";

const s = (title: string, authors_short: string, year: number): Source => ({
  title, authors_short, year, journal: "J", doi: `10.1/${title}`,
});
const papers = { a: s("Paper A", "John et al.", 2024), b: s("Paper B", "Hayman et al.", 2022) };
const chunk = (source_file: string | null, page: number | null = null): Chunk => ({ text: "text", source_file, page });

describe("PAPERS", () => {
  it("loads all six papers with full metadata", () => {
    expect(Object.keys(PAPERS)).toHaveLength(6);
    for (const p of Object.values(PAPERS)) {
      expect(p.title && p.authors_short && p.journal).toBeTruthy();
      expect(Number.isInteger(p.year)).toBe(true);
      expect(p.doi?.startsWith("10.")).toBe(true);
    }
  });
});

describe("resolveSources", () => {
  it("de-duplicates, keeps retrieval order and collects sorted unique pages", () => {
    const out = resolveSources([chunk("b", 9), chunk("a", 2), chunk("b", 3), chunk("b", 9)], papers);
    expect(out.map((x) => [x.title, x.pages])).toEqual([["Paper B", [3, 9]], ["Paper A", [2]]]);
  });

  it("gives an empty page list when pages are unknown", () => {
    expect(resolveSources([chunk("a")], papers)[0]?.pages).toEqual([]);
  });

  it("does not mutate the shared papers lookup", () => {
    resolveSources([chunk("a", 4)], papers);
    expect("pages" in papers.a).toBe(false);
  });

  it("falls back to the raw file name and warns for unmapped files", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const out = resolveSources([chunk("mystery-file", 1)], papers);
    expect(out).toEqual([{ title: "mystery-file", authors_short: null, year: null, journal: null, doi: null, pages: [1] }]);
    expect(warn.mock.calls.flat().join(" ")).toContain("mystery-file");
    warn.mockRestore();
  });

  it("ignores chunks without a source file", () => {
    expect(resolveSources([chunk(null), chunk("")], papers)).toEqual([]);
  });
});
