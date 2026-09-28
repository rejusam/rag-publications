// Retrieve relevant chunks, then ask the model to answer from them only.

import { generateWithFallback, type Llm } from "./llm.ts";
import { resolveSources } from "./papers.ts";
import { filterCitations, normaliseCitations, stripMarkdown, stripReasoning } from "./postprocess.ts";
import { buildPrompt, formatDocs } from "./prompt.ts";
import type { Chunk, CitedSource, Source } from "./types.ts";

export interface Deps {
  retrieve(question: string): Promise<Chunk[]>;
  llms: Llm[];
  papers: Record<string, Source>;
}

/** The model produced no usable answer text. */
export class EmptyAnswerError extends Error {
  constructor() {
    super("model returned no answer text");
    this.name = "EmptyAnswerError";
  }
}

export async function ask(deps: Deps, question: string): Promise<{ answer: string; sources: CitedSource[] }> {
  const chunks = await deps.retrieve(question);
  const prompt = buildPrompt(formatDocs(chunks, deps.papers), question);
  const raw = await generateWithFallback(deps.llms, prompt);
  let text = stripReasoning(raw);
  text = normaliseCitations(text);
  text = stripMarkdown(text);
  const sources = resolveSources(chunks, deps.papers);
  text = filterCitations(text, sources);
  if (!text) throw new EmptyAnswerError();
  return { answer: text, sources };
}
