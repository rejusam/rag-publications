import { afterEach, describe, expect, it, vi } from "vitest";
import type { Llm } from "../src/llm.ts";
import { ask, EmptyAnswerError, type Deps } from "../src/pipeline.ts";
import type { Chunk, Source } from "../src/types.ts";

afterEach(() => vi.restoreAllMocks());

const PAPERS: Record<string, Source> = {
  lassa: { title: "Lassa paper", authors_short: "John et al.", year: 2024, journal: null, doi: null },
};
const DOCS: Chunk[] = [
  { text: "Rodents shed virus.", source_file: "lassa", page: 3 },
  { text: "More on rodents.", source_file: "lassa", page: 5 },
];
const fixed = (text: string): Llm => ({ name: "fake", complete: async () => text });
const deps = (llms: Llm[], docs: Chunk[] = DOCS): Deps => ({ retrieve: async () => docs, llms, papers: PAPERS });

describe("ask", () => {
  it("returns a clean answer and resolved sources", async () => {
    const result = await ask(deps([fixed("<think>hmm</think>Via rodent contact.")]), "How does Lassa spread?");
    expect(result).toEqual({ answer: "Via rodent contact.", sources: [{ ...PAPERS.lassa, pages: [3, 5] }], status: "answered" });
  });

  it("normalises full-width citation brackets", async () => {
    const { answer } = await ask(deps([fixed("Answer text 【John et al., 2024】.")]), "q?");
    expect(answer).toBe("Answer text (John et al., 2024).");
  });

  it("strips Markdown", async () => {
    const { answer } = await ask(deps([fixed("**Bold** answer with *italic* text.")]), "q?");
    expect(answer).toBe("Bold answer with italic text.");
  });

  it("drops out-of-corpus citations and keeps in-corpus ones", async () => {
    const { answer } = await ask(
      deps([fixed("Spread happens via contact (John et al., 2024; Lo Iacono et al., 2015).")]),
      "q?",
    );
    expect(answer).toBe("Spread happens via contact (John et al., 2024).");
  });

  it("sends the formatted context and question to the model", async () => {
    const seen: string[] = [];
    const recorder: Llm = { name: "rec", complete: async (p) => { seen.push(p); return "ok"; } };
    await ask(deps([recorder]), "How does Lassa spread?");
    expect(seen[0]).toContain("[Source: John et al., 2024]");
    expect(seen[0]).toContain("How does Lassa spread?");
  });

  it("uses the fallback model when the primary fails", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const failing: Llm = { name: "down", complete: async () => { throw new Error("primary down"); } };
    expect((await ask(deps([failing, fixed("From fallback.")]), "q?")).answer).toBe("From fallback.");
  });

  it("raises EmptyAnswerError for reasoning-only output", async () => {
    await expect(ask(deps([fixed("<think>never finishes")]), "q?")).rejects.toBeInstanceOf(EmptyAnswerError);
  });

  it("still asks the model when retrieval finds nothing, and returns no sources", async () => {
    const result = await ask(deps([fixed("The papers do not cover that.")], []), "q?");
    expect(result).toEqual({ answer: "The papers do not cover that.", sources: [], status: "answered" });
  });

  it("reports status none and strips the STATUS line from the answer", async () => {
    const result = await ask(
      deps([fixed("STATUS: none\nThe papers do not cover treatment (John et al., 2024).")]),
      "q?",
    );
    expect(result.status).toBe("none");
    expect(result.answer.startsWith("STATUS")).toBe(false);
  });

  it("defaults to status answered when the model omits the STATUS line", async () => {
    const result = await ask(deps([fixed("Plain answer.")]), "q?");
    expect(result.status).toBe("answered");
  });
});
