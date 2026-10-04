import { describe, expect, it } from "vitest";
import { buildPrompt, formatDocs } from "../src/prompt.ts";
import type { Chunk, Source } from "../src/types.ts";

const PAPERS: Record<string, Source> = {
  lassa: { title: "Lassa paper", authors_short: "John et al.", year: 2024, journal: null, doi: null },
};
const DOCS: Chunk[] = [
  { text: "Rodents shed virus.", source_file: "lassa", page: null },
  { text: "More on rodents.", source_file: "lassa", page: null },
];

describe("formatDocs", () => {
  it("labels sources by author and year", () => {
    const text = formatDocs(DOCS, PAPERS);
    expect(text.split("[Source: John et al., 2024]").length - 1).toBe(2);
    expect(text).not.toContain("[Source: lassa]");
    expect(text).toContain("Rodents shed virus.");
    expect(text).toContain("\n\n---\n\n");
  });

  it("uses the raw source file for unmapped keys", () => {
    const text = formatDocs([{ text: "Unmapped text.", source_file: "unmapped-key", page: null }], PAPERS);
    expect(text).toContain("[Source: unmapped-key]");
  });

  it("uses 'unknown' when a chunk has no source file", () => {
    expect(formatDocs([{ text: "x", source_file: null, page: null }], PAPERS)).toContain("[Source: unknown]");
  });

  it("disambiguates shared author–year labels with six title words", () => {
    const papers: Record<string, Source> = {
      "john-a": { title: "Travel time and disease transmission across large connected populations", authors_short: "John et al.", year: 2024, journal: null, doi: null },
      "john-b": { title: "Modelling Lassa virus dynamics in rodents and human spillover risk", authors_short: "John et al.", year: 2024, journal: null, doi: null },
    };
    const text = formatDocs(
      [{ text: "A text.", source_file: "john-a", page: null }, { text: "B text.", source_file: "john-b", page: null }],
      papers,
    );
    expect(text).toContain("[Source: John et al., 2024 — Travel time and disease transmission across]");
    expect(text).toContain("[Source: John et al., 2024 — Modelling Lassa virus dynamics in rodents]");
  });
});

describe("buildPrompt", () => {
  it("contains context, question and the grounding rules", () => {
    const prompt = buildPrompt(formatDocs(DOCS, PAPERS), "How does Lassa spread?");
    expect(prompt).toContain("[Source: John et al., 2024]");
    expect(prompt).toContain("How does Lassa spread?");
    expect(prompt).toContain("ONLY the provided context");
    expect(prompt).toContain("Never cite references that appear inside the context text");
    expect(prompt).toContain("Write in plain prose without Markdown");
    expect(prompt).toContain("Use British/NZ spelling");
    expect(prompt).toContain("Begin your reply with exactly one line: STATUS: answered, STATUS: partial or STATUS: none.");
  });

  it("inserts text containing $ patterns literally", () => {
    const prompt = buildPrompt("cost $& and $1", "what about $'?");
    expect(prompt).toContain("cost $& and $1");
    expect(prompt).toContain("what about $'?");
  });
});

describe("no-answer instruction", () => {
  it("tells the model to open a none answer with the plain line", () => {
    const text = buildPrompt(formatDocs(DOCS, PAPERS), "Who won the 2023 Rugby World Cup?");
    expect(text).toContain("start the answer with exactly: These papers don’t cover that.");
    expect(text).not.toContain("say so honestly");
  });
});
