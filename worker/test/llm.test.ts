import { afterEach, describe, expect, it, vi } from "vitest";
import { extractResponseText, generateWithFallback, groqLlm, UpstreamError, workersAiLlm, type Llm } from "../src/llm.ts";
import type { Env } from "../src/types.ts";

const ok = (text: string): Llm => ({ name: "ok", complete: async () => text });
const fail = (status: number, name = "fail"): Llm => ({
  name,
  complete: async () => { throw new UpstreamError(`${name} down`, status); },
});

function fakeEnv(overrides: Partial<Env> = {}): Env {
  return {
    AI: {
      gateway: () => ({ getUrl: async (p?: string) => `https://gw.example/acct/ask-my-research/${p ?? ""}` }),
      run: vi.fn(),
    } as unknown as Ai,
    VECTORIZE: {} as Vectorize,
    RATE_LIMITER: { limit: async () => ({ success: true }) } as unknown as RateLimit,
    GROQ_API_KEY: "test-key",
    GROQ_MODEL: "openai/gpt-oss-120b",
    FALLBACK_MODEL: "@cf/openai/gpt-oss-120b",
    GATEWAY_ID: "ask-my-research",
    ALLOWED_ORIGINS: "https://rejusamjohn.pages.dev",
    ...overrides,
  };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("generateWithFallback", () => {
  it("returns the primary answer", async () => {
    expect(await generateWithFallback([ok("A"), ok("B")], "p")).toBe("A");
  });
  it("uses the fallback when the primary fails", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    expect(await generateWithFallback([fail(500), ok("From fallback.")], "p")).toBe("From fallback.");
  });
  it("raises the last failure when all fail", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    await expect(generateWithFallback([fail(429, "first"), fail(503, "last")], "p")).rejects.toMatchObject({ status: 503, message: "last down" });
  });
});

describe("extractResponseText", () => {
  it.each([
    ["plain", "plain"],
    [{ output_text: "ot" }, "ot"],
    [{ response: "r" }, "r"],
    [{ choices: [{ message: { content: "cc" } }] }, "cc"],
    [{ output: [{ type: "reasoning", content: [{ type: "reasoning_text", text: "hmm" }] }, { type: "message", content: [{ type: "output_text", text: "Final " }, { type: "output_text", text: "answer." }] }] }, "Final answer."],
    [{}, null],
    [null, null],
  ])("%j", (input, expected) => {
    expect(extractResponseText(input)).toBe(expected);
  });
});

describe("groqLlm", () => {
  it("posts to the gateway Groq URL with the key and returns content", async () => {
    const fetchFn = vi.fn(async () => Response.json({ choices: [{ message: { content: "Answer." } }] }));
    const text = await groqLlm(fakeEnv(), fetchFn as unknown as typeof fetch).complete("PROMPT");
    expect(text).toBe("Answer.");
    const [url, init] = fetchFn.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("https://gw.example/acct/ask-my-research/groq/chat/completions");
    expect((init.headers as Record<string, string>).Authorization).toBe("Bearer test-key");
    const body = JSON.parse(init.body as string);
    expect(body.model).toBe("openai/gpt-oss-120b");
    expect(body.messages).toEqual([{ role: "user", content: "PROMPT" }]);
    expect(body.temperature).toBe(0.1);
    expect(body.reasoning_effort).toBe("low");
  });

  it("maps HTTP errors to UpstreamError with the status", async () => {
    const fetchFn = vi.fn(async () => new Response("slow down", { status: 429 }));
    await expect(groqLlm(fakeEnv(), fetchFn as unknown as typeof fetch).complete("p")).rejects.toMatchObject({ status: 429 });
  });

  it("treats a 200 with empty content as a failure so the fallback runs", async () => {
    const fetchFn = vi.fn(async () => Response.json({ choices: [{ message: { content: "" } }] }));
    await expect(groqLlm(fakeEnv(), fetchFn as unknown as typeof fetch).complete("p")).rejects.toBeInstanceOf(UpstreamError);
  });

  it("treats a 200 with non-string content as a failure", async () => {
    const fetchFn = vi.fn(async () => Response.json({ choices: [{ message: { content: null } }] }));
    await expect(groqLlm(fakeEnv(), fetchFn as unknown as typeof fetch).complete("p")).rejects.toBeInstanceOf(UpstreamError);
  });

  it("maps a network failure (or the request timing out) to a 503", async () => {
    const fetchFn = vi.fn(async () => { throw new TypeError("fetch failed"); });
    await expect(groqLlm(fakeEnv(), fetchFn as unknown as typeof fetch).complete("p")).rejects.toMatchObject({ status: 503 });
  });

  it("maps a non-JSON response body to a 503", async () => {
    const fetchFn = vi.fn(async () => new Response("not json", { status: 200 }));
    await expect(groqLlm(fakeEnv(), fetchFn as unknown as typeof fetch).complete("p")).rejects.toMatchObject({ status: 503 });
  });
});

describe("workersAiLlm", () => {
  it("runs the fallback model through the gateway with low reasoning effort", async () => {
    const env = fakeEnv();
    (env.AI.run as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({ output_text: "From Workers AI." });
    expect(await workersAiLlm(env).complete("PROMPT")).toBe("From Workers AI.");
    const [model, input, options] = (env.AI.run as unknown as ReturnType<typeof vi.fn>).mock.calls[0]!;
    expect(model).toBe("@cf/openai/gpt-oss-120b");
    expect(input).toMatchObject({ input: "PROMPT", reasoning: { effort: "low" } });
    expect(options).toEqual({ gateway: { id: "ask-my-research" } });
  });

  it("wraps binding errors as UpstreamError 503", async () => {
    const env = fakeEnv();
    (env.AI.run as unknown as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("4006: daily free allocation used"));
    await expect(workersAiLlm(env).complete("p")).rejects.toMatchObject({ status: 503 });
  });

  it("treats an empty result as a failure", async () => {
    const env = fakeEnv();
    (env.AI.run as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({ output: [] });
    await expect(workersAiLlm(env).complete("p")).rejects.toBeInstanceOf(UpstreamError);
  });

  it("times out after 20s and maps to 503, without waiting for a hung call", async () => {
    vi.useFakeTimers();
    try {
      const env = fakeEnv();
      (env.AI.run as unknown as ReturnType<typeof vi.fn>).mockReturnValue(new Promise(() => {}));
      const result = expect(workersAiLlm(env).complete("p")).rejects.toMatchObject({ status: 503 });
      await vi.advanceTimersByTimeAsync(20_000);
      await result;
    } finally {
      vi.useRealTimers();
    }
  });
});
