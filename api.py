"""FastAPI backend for the publications RAG chatbot.

Deployment stack:
    - Embeddings: sentence-transformers (all-MiniLM-L6-v2)
    - Vector store: ChromaDB (pre-built, bundled in repo)
    - LLM: OpenAI gpt-4o-mini (default) or Google Gemini

Run locally:
    GROQ_API_KEY=xxx uvicorn api:app --reload --port 8000

    Or with OpenAI:
    LLM_PROVIDER=openai OPENAI_API_KEY=xxx uvicorn api:app --reload --port 8000

Deploy on Render:
    See render.yaml
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────

CHROMA_DIR = Path(__file__).parent / "chroma_db_deploy"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # "groq", "openai", or "gemini"
RETRIEVER_K = 4

# ─── Prompt ───────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """\
You are a research assistant for Dr Reju Sam John, a computational \
epidemiologist and data scientist based in Auckland, New Zealand.

INSTRUCTIONS:
- Answer the question using ONLY the provided context from his \
published peer-reviewed papers.
- If the context does not contain enough information to answer, \
say so honestly — do not hallucinate.
- Cite papers using short names like (John et al., 2024) — derive \
the author and year from the filename in the [Source:] tags.
- Be concise but thorough.
- Keep answers under 150 words for readability.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

# ─── Globals ──────────────────────────────────────────────────────────

chain: Runnable | None = None


def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a labelled context string."""
    sections: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown")
        sections.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


# ─── App lifecycle ────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and vector store on startup."""
    global chain

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )

    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    else:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

    prompt = ChatPromptTemplate.from_template(SYSTEM_TEMPLATE)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    yield


# ─── FastAPI app ──────────────────────────────────────────────────────

app = FastAPI(title="Ask My Research — RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://rejusamjohn.pages.dev",
        "http://localhost:8080",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ─── Models ───────────────────────────────────────────────────────────


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


# ─── Routes ───────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        answer = chain.invoke(req.question)
        return AskResponse(answer=answer)
    except Exception as e:
        logger.exception("Chain invocation failed")
        raise HTTPException(status_code=500, detail=str(e))
