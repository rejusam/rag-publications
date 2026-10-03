// The model's first line says whether the papers answered the question. It is read only from
// the first non-empty line and removed; anything unexpected leaves the answer as it was.

export type AnswerStatus = "answered" | "partial" | "none";
export const NONE_FALLBACK = "These papers don’t answer this question.";
const LINE = /^[\s*_`]*STATUS:\s*(answered|partial|none)[\s*_`.]*$/i;

export function splitStatus(text: string): { status: AnswerStatus; text: string } {
  const lines = text.split("\n");
  const i = lines.findIndex((l) => l.trim() !== "");
  const m = i === -1 ? null : LINE.exec(lines[i]!);
  if (!m) return { status: "answered", text };
  const status = m[1]!.toLowerCase() as AnswerStatus;
  const rest = lines.slice(i + 1).join("\n").trim();
  return { status, text: rest || (status === "none" ? NONE_FALLBACK : "") };
}
