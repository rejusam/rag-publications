// HTTP layer: validation, rate limiting, CORS, error mapping and health.

import { groqLlm, workersAiLlm } from "./llm.ts";
import { PAPERS } from "./papers.ts";
import { ask, EmptyAnswerError, type Deps } from "./pipeline.ts";
import { retrieve } from "./retrieve.ts";
import type { Env } from "./types.ts";

const MIN_QUESTION = 3;
const MAX_QUESTION = 500;

function newRequestId(): string {
  return crypto.randomUUID().replaceAll("-", "").slice(0, 12);
}

function corsHeaders(origin: string | null, env: Env): Headers {
  const headers = new Headers({ Vary: "Origin" });
  const allowed = env.ALLOWED_ORIGINS.split(",").map((o) => o.trim()).filter(Boolean);
  if (origin && allowed.includes(origin)) {
    headers.set("Access-Control-Allow-Origin", origin);
    headers.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    headers.set("Access-Control-Allow-Headers", "Content-Type");
    headers.set("Access-Control-Expose-Headers", "X-Request-ID");
    headers.set("Access-Control-Max-Age", "86400");
  }
  return headers;
}

/** Trimmed question if valid, else null. Length is counted in code points. */
async function readQuestion(request: Request): Promise<string | null> {
  let body: unknown;
  try {
    body = JSON.parse(await request.text());
  } catch {
    return null;
  }
  const question = (body as { question?: unknown } | null)?.question;
  if (typeof question !== "string") return null;
  const trimmed = question.trim();
  const length = [...trimmed].length;
  return length >= MIN_QUESTION && length <= MAX_QUESTION ? trimmed : null;
}

function defaultDeps(env: Env): Deps {
  return {
    retrieve: (question) => retrieve(env, question),
    llms: [groqLlm(env), workersAiLlm(env)],
    papers: PAPERS,
  };
}

export function makeHandler(buildDeps: (env: Env) => Deps) {
  return async function handle(request: Request, env: Env): Promise<Response> {
    const requestId = newRequestId();
    const headers = corsHeaders(request.headers.get("Origin"), env);
    headers.set("X-Request-ID", requestId);
    const json = (data: unknown, status: number) => {
      const h = new Headers(headers);
      h.set("Content-Type", "application/json");
      return new Response(JSON.stringify(data), { status, headers: h });
    };

    try {
      const { pathname } = new URL(request.url);
      if (pathname !== "/ask" && pathname !== "/health") return json({ error: "not_found" }, 404);
      if (request.method === "OPTIONS") return new Response(null, { status: 204, headers });

      if (pathname === "/health") {
        if (request.method !== "GET") return json({ error: "method_not_allowed" }, 405);
        return json({ status: "ok" }, 200);
      }

      if (request.method !== "POST") return json({ error: "method_not_allowed" }, 405);
      const question = await readQuestion(request);
      if (question === null) return json({ error: "invalid_question" }, 422);

      const ip = request.headers.get("CF-Connecting-IP")?.trim() || "unknown";
      const { success } = await env.RATE_LIMITER.limit({ key: ip });
      if (!success) return json({ error: "rate_limited" }, 429);

      try {
        return json(await ask(buildDeps(env), question), 200);
      } catch (error) {
        console.error(`ask failed (request_id=${requestId}):`, error);
        if (!(error instanceof EmptyAnswerError) && (error as { status?: number })?.status === 429) {
          return json({ error: "rate_limited" }, 429);
        }
        return json({ error: "unavailable" }, 503);
      }
    } catch (error) {
      console.error(`unhandled error (request_id=${requestId}):`, error);
      return json({ error: "unavailable" }, 503);
    }
  };
}

export default { fetch: makeHandler(defaultDeps) } satisfies ExportedHandler<Env>;
