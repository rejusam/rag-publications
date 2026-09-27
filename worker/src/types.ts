export interface Source {
  title: string;
  authors_short: string | null;
  year: number | null;
  journal: string | null;
  doi: string | null;
}

export interface Chunk {
  text: string;
  source_file: string | null;
}

export interface Env {
  AI: Ai;
  VECTORIZE: Vectorize;
  RATE_LIMITER: RateLimit;
  GROQ_API_KEY: string;
  GROQ_MODEL: string;
  FALLBACK_MODEL: string;
  GATEWAY_ID: string;
  ALLOWED_ORIGINS: string;
}
