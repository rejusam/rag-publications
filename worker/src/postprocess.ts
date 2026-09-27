// Deterministic clean-up of model output. Ported from rag_api/pipeline.py;
// behaviour must stay identical, so change both or neither.

import type { Source } from "./types.ts";

const THINK_BLOCK = /<think>[\s\S]*?<\/think>/gi;
const THINK_OPEN = /<think>/i;
const FULLWIDTH_CITATION = /【(.*?)】/g;
const BOLD = /\*\*([\s\S]+?)\*\*|__([\s\S]+?)__/g;
const ITALIC = /\*(\S(?:[^\n*]*\S)?)\*/gu;
const LEADING_MARKER = /^[ \t]*(?:[-*•][ \t]+(?![ \t\d])|#{1,6}\s+)/gmu;
const INNER_GROUP = /\(([^()]*)\)/g;
const BARE_YEAR = /^\d{4}[a-z]?$/;
const MONTH_NAMES = new Set([
  "january", "february", "march", "april", "may", "june", "july",
  "august", "september", "october", "november", "december",
]);
const MONTH_WORD = /\p{L}+/gu;

// Lowercase name particles that may precede a surname ("van der Berg").
const PARTICLES = [
  "van", "von", "der", "den", "de", "del", "della", "da", "di", "du",
  "le", "la", "dos", "das", "ter", "ten", "'d", "’d", "bin", "al-",
];
const PARTICLE_SET = new Set(PARTICLES);
// No "-" in this set: "\-" outside a character class is a syntax error under
// the `u` flag, and a bare "-" is already literal there ("al-").
const escapeRegex = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const PARTICLE = `(?:${PARTICLES.map(escapeRegex).join("|")})\\s+`;
// A surname word is letters plus apostrophes and hyphens. ‐ and ‑ are the
// hyphen and non-breaking hyphen gpt-oss often writes instead of "-".
const LETTER = "[\\p{L}\\p{Nl}\\p{No}]";
const SURNAME_WORD = `${LETTER}(?:${LETTER}|['’\\-‐‑])+`;
const SURNAME = `(?:${PARTICLE}){0,3}${SURNAME_WORD}(?:\\s+${SURNAME_WORD})?`;
const AUTHOR = `${SURNAME}(?:\\s+et\\s+al\\.?|\\s+(?:and|&)\\s+${SURNAME})?`;
const YEAR = "\\d{4}[a-z]?";
const CITATION_ENTRY = new RegExp(
  "^(?:(?:see|e\\.g\\.,?|cf\\.)\\s+)?" +
    `(?<author>${AUTHOR}),?[ \\t]{0,3}(?<year>${YEAR})` +
    "(?:,[ \\t]{0,3}\\d{4}[a-z]?)*" +
    "(?:,[ \\t]{0,3}(?:p|pp)\\.[ \\t]{0,3}[\\d\\-–]+" +
    "|[ \\t]{0,3}[—–-][ \\t]{0,3}.+)?$",
  "u",
);
const ET_AL_SUFFIX = /\s+et\s+al\.?$/;
const AUTHOR_JOINER = /\s+(?:and|&)\s+/;

/** Remove any reasoning markup so only the final answer reaches users. */
export function stripReasoning(text: string): string {
  text = text.replace(THINK_BLOCK, "");
  const open = text.search(THINK_OPEN);
  if (open !== -1) text = text.slice(0, open);
  return text.trim();
}

/** Replace the full-width brackets gpt-oss sometimes uses with plain parens. */
export function normaliseCitations(text: string): string {
  text = text.replace(FULLWIDTH_CITATION, (_m, inner: string) => `(${inner})`);
  return text.replaceAll("【", "(").replaceAll("】", ")");
}

/**
 * Undo the handful of Markdown patterns models emit despite the prompt
 * asking for plain prose. Not a Markdown parser, on purpose.
 */
export function stripMarkdown(text: string): string {
  text = text.replace(BOLD, (_m, a?: string, b?: string) => a ?? b ?? "");
  text = text.replace(ITALIC, (_m, inner: string) => inner);
  return text.replace(LEADING_MARKER, "");
}

function normaliseAuthor(author: string): string {
  // Fold ‐/‑ to "-" so hyphenated surnames match; key only, never output.
  author = author.normalize("NFC").replaceAll("‐", "-").replaceAll("‑", "-");
  return author.replaceAll(".", "").replaceAll(",", "").split(/\s+/).filter(Boolean).join(" ").toLowerCase();
}

/**
 * Each surname: at most 3 leading particles, then 1–2 capitalised words.
 * Rejects lowercase phrases the regex alone accepts ("in 2019", "de facto, 2019").
 */
function isSurnameAuthor(author: string): boolean {
  author = author.replace(ET_AL_SUFFIX, "");
  for (const surname of author.split(AUTHOR_JOINER)) {
    let words = surname.split(/\s+/).filter(Boolean);
    let particles = 0;
    while (particles < 3 && words.length > 1 && PARTICLE_SET.has(words[0]!)) {
      words = words.slice(1);
      particles += 1;
    }
    if (words.length < 1 || words.length > 2) return false;
    if (!words.every((w) => /^\p{Lu}/u.test(w))) return false;
  }
  return true;
}

function hasMonthToken(author: string): boolean {
  return (author.match(MONTH_WORD) ?? []).some((w) => MONTH_NAMES.has(w.toLowerCase()));
}

const keyOf = (author: string, year: number) => `${author}\u0000${year}`;

/**
 * Returns the entries to keep and whether anything was dropped. When nothing
 * was dropped the caller must leave the group text exactly as written.
 */
function splitGroupEntries(content: string, known: Set<string>): { kept: string[]; droppedAny: boolean } {
  if (!content.trim()) return { kept: [], droppedAny: true };
  if (!/\d{4}/.test(content)) return { kept: [content], droppedAny: false };

  const kept: string[] = [];
  let droppedAny = false;
  let prevDroppedCitation = false;
  for (const rawEntry of content.split(";")) {
    const entry = rawEntry.trim();
    if (!entry) continue;
    // NFC for matching only; a kept entry is appended exactly as written.
    const match = CITATION_ENTRY.exec(entry.normalize("NFC"));
    const author = match?.groups?.author;
    if (match && author && isSurnameAuthor(author) && !hasMonthToken(author)) {
      const year = Number.parseInt(match.groups!.year!.slice(0, 4), 10);
      if (known.has(keyOf(normaliseAuthor(author), year))) {
        kept.push(entry);
        prevDroppedCitation = false;
      } else {
        droppedAny = true;
        prevDroppedCitation = true;
      }
      continue;
    }
    if (prevDroppedCitation && BARE_YEAR.test(entry)) {
      droppedAny = true;
      continue; // orphan year left behind by a dropped citation
    }
    kept.push(entry);
    prevDroppedCitation = false;
  }
  return { kept, droppedAny };
}

/**
 * Scan backwards through out and prefixText to find two facts without
 * concatenating (which is quadratic): whether the accumulated result ends
 * with a space, and what the last non-space character is. Must walk back
 * through all pieces (including empty ones) to match Python's "".join(out).
 */
function scanSoFarBackwards(out: string[], prefixText: string): {
  endsWithSpace: boolean;
  lastNonSpaceChar: string | null;
} {
  let endsWithSpace = false;
  let lastNonSpaceChar: string | null = null;

  // First, find what the last character is (for endsWithSpace)
  if (prefixText.length > 0) {
    // If prefixText is non-empty, its last char is the last overall
    endsWithSpace = prefixText[prefixText.length - 1]! === " ";
  } else {
    // Otherwise walk back through out to find the first non-empty piece
    for (let j = out.length - 1; j >= 0; j--) {
      const piece = out[j]!;
      if (piece.length > 0) {
        endsWithSpace = piece[piece.length - 1]! === " ";
        break;
      }
    }
  }

  // Now find the last non-space character, scanning prefixText first
  for (let i = prefixText.length - 1; i >= 0; i--) {
    if (prefixText[i]! !== " ") {
      lastNonSpaceChar = prefixText[i]!;
      break;
    }
  }

  // If not found in prefixText, scan out backwards
  if (lastNonSpaceChar === null) {
    for (let j = out.length - 1; j >= 0; j--) {
      const piece = out[j]!;
      for (let i = piece.length - 1; i >= 0; i--) {
        if (piece[i]! !== " ") {
          lastNonSpaceChar = piece[i]!;
          break;
        }
      }
      if (lastNonSpaceChar !== null) break;
    }
  }

  return { endsWithSpace, lastNonSpaceChar };
}

function runFilterPass(text: string, known: Set<string>): string {
  const out: string[] = [];
  let lastEnd = 0;
  const n = text.length;
  for (const match of text.matchAll(INNER_GROUP)) {
    const start = match.index!;
    const end = start + match[0].length;
    const { kept, droppedAny } = splitGroupEntries(match[1]!, known);
    const prefixText = text.slice(lastEnd, start);

    if (!droppedAny) {
      out.push(prefixText, match[0]);
      lastEnd = end;
      continue;
    }
    if (kept.length > 0) {
      out.push(prefixText, `(${kept.join("; ")})`);
      lastEnd = end;
      continue;
    }

    // Whole group dropped: tidy locally only. The following "sentence" is
    // left alone; just the one letter that would now start it is capitalised.
    const { endsWithSpace, lastNonSpaceChar } = scanSoFarBackwards(out, prefixText);
    const isSentenceStart = lastNonSpaceChar === null || ".!?\n".includes(lastNonSpaceChar);

    const leftTrim = endsWithSpace ? 1 : 0;
    out.push(prefixText.slice(0, prefixText.length - leftTrim));
    const rightExtra =
      end < n && text[end] === " " &&
      (leftTrim === 0 || (end + 1 < n && ".,;".includes(text[end + 1]!)))
        ? 1
        : 0;
    lastEnd = end + rightExtra;

    if (isSentenceStart) {
      let peek = lastEnd;
      if (peek < n && text[peek] === " ") peek += 1;
      if (peek < n && /\p{Ll}/u.test(text[peek]!)) {
        out.push(text.slice(lastEnd, peek), text[peek]!.toUpperCase());
        lastEnd = peek + 1;
      }
    }
  }
  out.push(text.slice(lastEnd));
  return out.join("");
}

/**
 * Drop parenthesised citations that don't name a retrieved paper. Runs to a
 * fixed point so a nested group emptied by one pass is cleaned by the next.
 */
export function filterCitations(text: string, sources: readonly Source[]): string {
  const known = new Set(
    sources
      .filter((s) => s.authors_short && s.year)
      .map((s) => keyOf(normaliseAuthor(s.authors_short!), s.year!)),
  );
  let previous: string | null = null;
  while (previous !== text) {
    previous = text;
    text = runFilterPass(text, known);
  }
  return text;
}
