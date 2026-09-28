import { describe, expect, it, vi } from "vitest";
import { EMBEDDING_MODEL, retrieve } from "../src/retrieve.ts";

function env(matches: unknown[], data: number[][] = [[0.1, 0.2]]) {
  return {
    AI: { run: vi.fn(async () => ({ data })) } as unknown as Ai,
    VECTORIZE: { query: vi.fn(async () => ({ matches, count: matches.length })) } as unknown as Vectorize,
  };
}

describe("retrieve", () => {
  it("embeds the question and returns text, source and 1-based page in match order", async () => {
    const e = env([
      { id: "1", score: 0.9, metadata: { text: "First.", source_file: "a", page: 0 } },
      { id: "2", score: 0.8, metadata: { text: "Second.", source_file: "b", page: 6 } },
    ]);
    const chunks = await retrieve(e, "How does Lassa spread?");
    expect(chunks).toEqual([
      { text: "First.", source_file: "a", page: 1 },
      { text: "Second.", source_file: "b", page: 7 },
    ]);
    expect(e.AI.run).toHaveBeenCalledWith(EMBEDDING_MODEL, { text: ["How does Lassa spread?"] });
    expect(e.VECTORIZE.query).toHaveBeenCalledWith([0.1, 0.2], { topK: 4, returnMetadata: "all" });
  });

  it("drops matches without text and tolerates missing source file and page", async () => {
    const e = env([
      { id: "1", score: 0.9, metadata: {} },
      { id: "2", score: 0.8, metadata: { text: "Kept." } },
    ]);
    expect(await retrieve(e, "q")).toEqual([{ text: "Kept.", source_file: null, page: null }]);
  });

  it("ignores page metadata that is not a non-negative integer", async () => {
    const e = env([
      { id: "1", score: 0.9, metadata: { text: "a", page: -1 } },
      { id: "2", score: 0.9, metadata: { text: "b", page: 2.5 } },
      { id: "3", score: 0.9, metadata: { text: "c", page: "4" } },
    ]);
    expect((await retrieve(e, "q")).map((c) => c.page)).toEqual([null, null, null]);
  });

  it("returns an empty list when the index has no matches", async () => {
    expect(await retrieve(env([]), "q")).toEqual([]);
  });

  it("throws when the embedding comes back empty", async () => {
    await expect(retrieve(env([], []), "q")).rejects.toThrow("embedding");
  });
});
