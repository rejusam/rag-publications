import type { Env } from "./types.ts";

const LLM_TIMEOUT_MS = 20_000;

export interface Llm {
  name: string;
  complete(prompt: string): Promise<string>;
}

/** A model call that failed; `status` decides what the browser sees. */
export class UpstreamError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = "UpstreamError";
  }
}

/**
 * Try each model in turn; if all fail, throw the last failure, so the status
 * the caller maps reflects the model that failed last (as the Python API did).
 */
export async function generateWithFallback(llms: readonly Llm[], prompt: string): Promise<string> {
  if (llms.length === 0) throw new Error("at least one model is required");
  let lastError: unknown;
  for (const llm of llms) {
    try {
      return await llm.complete(prompt);
    } catch (error) {
      console.error(`model ${llm.name} failed:`, error);
      lastError = error;
    }
  }
  throw lastError;
}

/** Pull the answer text out of whichever response shape a provider returned. */
export function extractResponseText(res: unknown): string | null {
  if (typeof res === "string") return res;
  if (res === null || typeof res !== "object") return null;
  const r = res as Record<string, any>;
  if (typeof r.output_text === "string") return r.output_text;
  if (typeof r.response === "string") return r.response;
  const content = r.choices?.[0]?.message?.content;
  if (typeof content === "string") return content;
  if (Array.isArray(r.output)) {
    const parts: string[] = [];
    for (const item of r.output) {
      if (item?.type !== "message" || !Array.isArray(item.content)) continue;
      for (const c of item.content) {
        if (c?.type === "output_text" && typeof c.text === "string") parts.push(c.text);
      }
    }
    if (parts.length > 0) return parts.join("");
  }
  return null;
}

export function groqLlm(env: Env, fetchFn: typeof fetch = fetch): Llm {
  return {
    name: `groq:${env.GROQ_MODEL}`,
    async complete(prompt) {
      const base = (await env.AI.gateway(env.GATEWAY_ID).getUrl("groq")).replace(/\/+$/, "");
      let body: unknown;
      try {
        const res = await fetchFn(`${base}/chat/completions`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${env.GROQ_API_KEY}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model: env.GROQ_MODEL,
            messages: [{ role: "user", content: prompt }],
            temperature: 0.1,
            reasoning_effort: "low",
          }),
          signal: AbortSignal.timeout(LLM_TIMEOUT_MS),
        });
        if (!res.ok) throw new UpstreamError(`groq returned ${res.status}`, res.status);
        body = await res.json();
      } catch (error) {
        if (error instanceof UpstreamError) throw error;
        // Network failure, the 20s timeout firing, or a non-JSON body all land
        // here; normalise to 503 the same way workersAiLlm does so the caller
        // always has a status to map, even for the last model in the chain.
        throw new UpstreamError(`groq request failed: ${String(error)}`, 503);
      }
      const text = extractResponseText(body);
      // A reasoning-only reply counts as a failure so the fallback gets a turn.
      if (!text || !text.trim()) throw new UpstreamError("groq returned no answer text", 502);
      return text;
    },
  };
}

export function workersAiLlm(env: Env): Llm {
  return {
    name: `workers-ai:${env.FALLBACK_MODEL}`,
    async complete(prompt) {
      let res: unknown;
      let timer: ReturnType<typeof setTimeout> | undefined;
      try {
        res = await Promise.race([
          env.AI.run(
            env.FALLBACK_MODEL as keyof AiModels,
            { input: prompt, reasoning: { effort: "low" } } as never,
            { gateway: { id: env.GATEWAY_ID } },
          ),
          new Promise<never>((_resolve, reject) => {
            timer = setTimeout(() => reject(new UpstreamError("workers ai timed out", 503)), LLM_TIMEOUT_MS);
          }),
        ]);
      } catch (error) {
        if (error instanceof UpstreamError) throw error;
        // Includes the daily free allocation running out; the widget shows
        // its "unavailable" message rather than a rate-limit one.
        throw new UpstreamError(`workers ai failed: ${String(error)}`, 503);
      } finally {
        clearTimeout(timer);
      }
      const text = extractResponseText(res);
      if (!text || !text.trim()) throw new UpstreamError("workers ai returned no answer text", 503);
      return text;
    },
  };
}
