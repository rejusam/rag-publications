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
  /** 1-based PDF page of the passage, or null if unknown. */
  page: number | null;
}

export interface CitedSource extends Source {
  /** Sorted, unique 1-based PDF pages of the retrieved passages from this paper. */
  pages: number[];
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
