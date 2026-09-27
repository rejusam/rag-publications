import { describe, expect, it, vi } from "vitest";
import { PAPERS, resolveSources } from "../src/papers.ts";
import type { Chunk, Source } from "../src/types.ts";

const s = (title: string, authors_short: string, year: number): Source => ({
  title, authors_short, year, journal: "J", doi: `10.1/${title}`,
});
const papers = { a: s("Paper A", "John et al.", 2024), b: s("Paper B", "Hayman et al.", 2022) };
const chunk = (source_file: string | null): Chunk => ({ text: "text", source_file });

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
  it("de-duplicates and keeps retrieval order", () => {
    const out = resolveSources([chunk("b"), chunk("a"), chunk("b")], papers);
    expect(out.map((x) => x.title)).toEqual(["Paper B", "Paper A"]);
  });

  it("falls back to the raw file name and warns for unmapped files", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const out = resolveSources([chunk("mystery-file")], papers);
    expect(out).toEqual([{ title: "mystery-file", authors_short: null, year: null, journal: null, doi: null }]);
    expect(warn.mock.calls.flat().join(" ")).toContain("mystery-file");
    warn.mockRestore();
  });

  it("ignores chunks without a source file", () => {
    expect(resolveSources([chunk(null), chunk("")], papers)).toEqual([]);
  });
});
