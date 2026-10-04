import type { Chunk, Source } from "./types.ts";

export const SYSTEM_TEMPLATE = `You are a research assistant for Dr Reju Sam John, a computational \
epidemiologist and data scientist based in Auckland, New Zealand.

INSTRUCTIONS:
- Begin your reply with exactly one line: STATUS: answered, STATUS: partial or STATUS: none. \
Use none when the context does not answer the question, partial when it answers only part of \
it, and answered otherwise. If the papers correct the question's premise (for example, \
the animal or cause it names is not quite right), that is an answer: use answered or partial \
and give the correction, not none. Then write the answer on the following lines.
- Answer the question using ONLY the provided context from his \
published peer-reviewed papers.
- If the context does not answer the question (STATUS: none), start the answer with exactly: \
These papers don’t cover that. Then, only if the context holds something closely related, add \
one short sentence saying what the papers do cover on it. Never guess or hallucinate.
- Cite only the papers named in the [Source: ...] tags, using that \
label in brackets, e.g. (John et al., 2024).
- Never cite references that appear inside the context text (for \
example other authors the papers themselves cite).
- Write in plain prose without Markdown: no asterisks, bullet lists, \
headings or bold text. Use short paragraphs separated by a blank \
line if needed.
- Do not end with a list of sources or quotations; the sources are \
shown to the reader separately.
- Use British/NZ spelling (for example behaviour, modelling, organisation).
- Be concise but thorough.
- Keep answers under 150 words for readability.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:`;

function citationLabel(
  sourceFile: string | null,
  papers: Record<string, Source>,
  sharedCounts: Map<string, number>,
): string {
  const source = sourceFile ? papers[sourceFile] : undefined;
  if (source && source.authors_short && source.year) {
    const label = `${source.authors_short}, ${source.year}`;
    if ((sharedCounts.get(label) ?? 0) > 1) {
      const titleWords = source.title.split(/\s+/).filter(Boolean).slice(0, 6).join(" ");
      return `${label} — ${titleWords}`;
    }
    return label;
  }
  return sourceFile || "unknown";
}

export function formatDocs(chunks: readonly Chunk[], papers: Record<string, Source>): string {
  const sharedCounts = new Map<string, number>();
  for (const p of Object.values(papers)) {
    if (p.authors_short && p.year) {
      const label = `${p.authors_short}, ${p.year}`;
      sharedCounts.set(label, (sharedCounts.get(label) ?? 0) + 1);
    }
  }
  return chunks
    .map((c) => `[Source: ${citationLabel(c.source_file, papers, sharedCounts)}]\n${c.text}`)
    .join("\n\n---\n\n");
}

export function buildPrompt(context: string, question: string): string {
  // Function replacers so "$&" and friends in user text stay literal.
  return SYSTEM_TEMPLATE.replace("{context}", () => context).replace("{question}", () => question);
}
