import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { makeHandler } from "../src/index.ts";
import { UpstreamError, type Llm } from "../src/llm.ts";
import type { Deps } from "../src/pipeline.ts";
import type { Env, Source } from "../src/types.ts";

const ORIGIN = "https://rejusamjohn.pages.dev";
const PAPERS: Record<string, Source> = {
  lassa: { title: "Lassa paper", authors_short: "John et al.", year: 2024, journal: "J", doi: "10.1/x" },
};

let limiterSuccess = true;
let llms: Llm[] = [];
const retrieveFn = vi.fn(async () => [{ text: "Rodents shed virus.", source_file: "lassa", page: 3 }]);

function env(): Env {
  return {
    AI: {} as Ai,
    VECTORIZE: {} as Vectorize,
    RATE_LIMITER: { limit: vi.fn(async () => ({ success: limiterSuccess })) } as unknown as RateLimit,
    GROQ_API_KEY: "k",
    GROQ_MODEL: "m",
    FALLBACK_MODEL: "f",
    GATEWAY_ID: "g",
    ALLOWED_ORIGINS: `${ORIGIN}, http://localhost:5500, https://tasmanlab.com`,
  };
}
const handler = makeHandler((): Deps => ({ retrieve: retrieveFn, llms, papers: PAPERS }));

function post(body: string, headers: Record<string, string> = { "Content-Type": "application/json", Origin: ORIGIN }) {
  return new Request("https://w.example/ask", { method: "POST", body, headers });
}
const askBody = (question: unknown) => JSON.stringify({ question });

beforeEach(() => {
  limiterSuccess = true;
  llms = [{ name: "ok", complete: async () => "Via rodents (John et al., 2024)." }];
  retrieveFn.mockClear();
  vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("health", () => {
  it("returns ok", async () => {
    const res = await handler(new Request("https://w.example/health"), env());
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ status: "ok" });
    expect(res.headers.get("X-Request-ID")).toMatch(/^[0-9a-f]{12}$/);
  });
});

describe("ask", () => {
  it("answers with sources and CORS headers", async () => {
    const res = await handler(post(askBody("How does Lassa spread?")), env());
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({
      answer: "Via rodents (John et al., 2024).",
      sources: [{ ...PAPERS.lassa, pages: [3] }],
    });
    expect(res.headers.get("Access-Control-Allow-Origin")).toBe(ORIGIN);
    expect(res.headers.get("Access-Control-Expose-Headers")).toBe("X-Request-ID");
  });

  it.each([
    ["not json", "{"],
    ["missing question", "{}"],
    ["non-string", askBody(42)],
    ["too short after trim", askBody("  hi  ")],
    ["too long", askBody("a".repeat(501))],
  ])("rejects %s with 422", async (_name, body) => {
    const res = await handler(post(body), env());
    expect(res.status).toBe(422);
    expect(await res.json()).toEqual({ error: "invalid_question" });
    expect(res.headers.get("Access-Control-Allow-Origin")).toBe(ORIGIN);
  });

  it("counts code points, not UTF-16 units", async () => {
    expect((await handler(post(askBody("🦠".repeat(500))), env())).status).toBe(200);
    expect((await handler(post(askBody("🦠".repeat(501))), env())).status).toBe(422);
  });

  it("accepts a JSON body sent as text/plain", async () => {
    const res = await handler(post(askBody("How does Lassa spread?"), { "Content-Type": "text/plain" }), env());
    expect(res.status).toBe(200);
  });

  it("rate limits by CF-Connecting-IP after validation", async () => {
    limiterSuccess = false;
    const e = env();
    const res = await handler(
      post(askBody("How does Lassa spread?"), { "Content-Type": "application/json", "CF-Connecting-IP": "203.0.113.9" }),
      e,
    );
    expect(res.status).toBe(429);
    expect(await res.json()).toEqual({ error: "rate_limited" });
    expect(e.RATE_LIMITER.limit).toHaveBeenCalledWith({ key: "203.0.113.9" });
    expect(retrieveFn).not.toHaveBeenCalled();
  });

  it("does not consume rate limit for invalid questions", async () => {
    const e = env();
    await handler(post(askBody("x")), e);
    expect(e.RATE_LIMITER.limit).not.toHaveBeenCalled();
  });

  it("maps a last-model 429 to rate_limited", async () => {
    llms = [
      { name: "a", complete: async () => { throw new UpstreamError("a", 500); } },
      { name: "b", complete: async () => { throw new UpstreamError("b", 429); } },
    ];
    const res = await handler(post(askBody("How does Lassa spread?")), env());
    expect(res.status).toBe(429);
  });

  it("maps other model failures to unavailable", async () => {
    llms = [{ name: "a", complete: async () => { throw new UpstreamError("a", 500); } }];
    const res = await handler(post(askBody("How does Lassa spread?")), env());
    expect(res.status).toBe(503);
    expect(await res.json()).toEqual({ error: "unavailable" });
  });

  it("maps an empty answer to unavailable", async () => {
    llms = [{ name: "a", complete: async () => "<think>only" }];
    expect((await handler(post(askBody("How does Lassa spread?")), env())).status).toBe(503);
  });

  it("maps retrieval failure to unavailable without leaking the message", async () => {
    retrieveFn.mockRejectedValueOnce(new Error("vectorize exploded: secret detail"));
    const res = await handler(post(askBody("How does Lassa spread?")), env());
    expect(res.status).toBe(503);
    expect(await res.text()).not.toContain("secret detail");
  });
});

describe("CORS and routing", () => {
  it("answers preflight with 204", async () => {
    const res = await handler(new Request("https://w.example/ask", { method: "OPTIONS", headers: { Origin: ORIGIN } }), env());
    expect(res.status).toBe(204);
    expect(res.headers.get("Access-Control-Allow-Methods")).toBe("GET, POST, OPTIONS");
    expect(res.headers.get("Access-Control-Allow-Headers")).toBe("Content-Type");
  });

  it("gives disallowed origins no allow-origin header", async () => {
    const res = await handler(post(askBody("How does Lassa spread?"), { "Content-Type": "application/json", Origin: "https://evil.example" }), env());
    expect(res.headers.get("Access-Control-Allow-Origin")).toBeNull();
    expect(res.headers.get("Vary")).toBe("Origin");
  });

  it("serves requests with no Origin header (curl)", async () => {
    const res = await handler(post(askBody("How does Lassa spread?"), { "Content-Type": "application/json" }), env());
    expect(res.status).toBe(200);
    expect(res.headers.get("Access-Control-Allow-Origin")).toBeNull();
    expect(res.headers.get("X-Request-ID")).toMatch(/^[0-9a-f]{12}$/);
  });

  it("allows the second configured origin despite surrounding spaces", async () => {
    const res = await handler(post(askBody("How does Lassa spread?"), { "Content-Type": "application/json", Origin: "http://localhost:5500" }), env());
    expect(res.headers.get("Access-Control-Allow-Origin")).toBe("http://localhost:5500");
  });

  it("allows the tasmanlab.com origin", async () => {
    const res = await handler(post(askBody("How does Lassa spread?"), { "Content-Type": "application/json", Origin: "https://tasmanlab.com" }), env());
    expect(res.headers.get("Access-Control-Allow-Origin")).toBe("https://tasmanlab.com");
  });

  it("gives a tasmanlab.com subdomain no allow-origin header", async () => {
    const res = await handler(post(askBody("How does Lassa spread?"), { "Content-Type": "application/json", Origin: "https://evil.tasmanlab.com" }), env());
    expect(res.headers.get("Access-Control-Allow-Origin")).toBeNull();
  });

  it("returns 405 for the wrong method and 404 for unknown paths", async () => {
    expect((await handler(new Request("https://w.example/ask"), env())).status).toBe(405);
    expect((await handler(new Request("https://w.example/health", { method: "POST" }), env())).status).toBe(405);
    expect((await handler(new Request("https://w.example/nope"), env())).status).toBe(404);
  });
});
