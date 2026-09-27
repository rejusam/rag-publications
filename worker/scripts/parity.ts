// Compare the Render API and the Worker on eval/questions.json.
//   node scripts/parity.ts https://rag-publications-api.onrender.com https://ask-my-research.<sub>.workers.dev
// Prints a Markdown table: expected paper retrieved, citations, and any
// citation that names a paper outside papers.json.

import { readFileSync } from "node:fs";
import { fileURLToPath, URL } from "node:url";

interface Question { id: string; question: string; expected: string | null }
interface Paper { title: string; authors_short: string; year: number }

const [renderUrl, workerUrl] = process.argv.slice(2);
if (!renderUrl || !workerUrl) {
  console.error("usage: node scripts/parity.ts <render-url> <worker-url>");
  process.exit(1);
}

const questions: Question[] = JSON.parse(
  readFileSync(fileURLToPath(new URL("../../eval/questions.json", import.meta.url)), "utf8"),
);
const papers: Record<string, Paper> = JSON.parse(
  readFileSync(fileURLToPath(new URL("../../papers.json", import.meta.url)), "utf8"),
);
const knownLabels = new Set(Object.values(papers).map((p) => `${p.authors_short}, ${p.year}`));
const CITATION = /\(([^()]*\d{4}[^()]*)\)/g;
const US_SPELLINGS = /\b(behavior|modeling|modeled|organization|analyze|analyzed|center|color|favor)\b/gi;

async function askOnce(base: string, question: string) {
  const started = Date.now();
  const res = await fetch(`${base.replace(/\/+$/, "")}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Origin: "https://rejusamjohn.pages.dev" },
    body: JSON.stringify({ question }),
  });
  const ms = Date.now() - started;
  const body = (await res.json()) as { answer?: string; sources?: { title: string }[] };
  return { status: res.status, ms, answer: body.answer ?? "", titles: (body.sources ?? []).map((s) => s.title) };
}

function citations(answer: string): string[] {
  return [...answer.matchAll(CITATION)].flatMap((m) => m[1]!.split(";").map((s) => s.trim()));
}

// Collapses any run of whitespace, including U+202F (the narrow no-break
// space gpt-oss puts before "et al."), to a single space before comparing,
// so a citation like "John et al., 2024" is recognised as in-corpus.
function normaliseWhitespace(s: string): string {
  return s.replace(/\s+/gu, " ").trim();
}

console.log("| id | side | status | ms | expected hit | citations | out-of-corpus | US spelling |");
console.log("|---|---|---|---|---|---|---|---|");
for (const q of questions) {
  const expectedTitle = q.expected ? papers[q.expected]?.title : null;
  for (const [side, url] of [["render", renderUrl], ["worker", workerUrl]] as const) {
    const r = await askOnce(url, q.question);
    const cites = citations(r.answer);
    const outside = cites.filter(
      (c) => !Array.from(knownLabels).some((label) => normaliseWhitespace(c).startsWith(normaliseWhitespace(label))),
    );
    const us = r.answer.match(US_SPELLINGS) ?? [];
    const hit = expectedTitle == null ? "n/a" : r.titles.includes(expectedTitle) ? "yes" : "no";
    console.log(`| ${q.id} | ${side} | ${r.status} | ${r.ms} | ${hit} | ${cites.length} | ${outside.join("; ") || "—"} | ${us.join(", ") || "—"} |`);
    await new Promise((resolve) => setTimeout(resolve, 7000)); // stay under 10/min on both
  }
}
