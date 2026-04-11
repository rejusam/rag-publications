"""RAG chain combining document retrieval with LLM generation.

Supports two backends:
    - **ollama** (default): Runs locally, no API key needed.
    - **huggingface**: Uses the free HF Inference API for deployment
      on Hugging Face Spaces where Ollama isn't available.

Switch via ``config.LLM_BACKEND``.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from src.config import (
    HF_MODEL_ID,
    LLM_BACKEND,
    LLM_MODEL,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
)
from src.retriever import get_retriever

# ─── Prompt ──────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """\
You are a research assistant for Dr Reju Sam John, a computational \
epidemiologist and data scientist based in Auckland, New Zealand.

INSTRUCTIONS:
- Answer the question using ONLY the provided context from his \
published peer-reviewed papers.
- If the context does not contain enough information to answer, \
say so honestly — do not hallucinate.
- Always cite which paper(s) your answer draws from using the \
[Source: filename] tags in the context.
- Be concise but thorough.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


# ─── Helpers ─────────────────────────────────────────────────────────


def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a labelled context string.

    Each chunk is prefixed with its source paper filename so the LLM
    can cite correctly.
    """
    sections: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown")
        sections.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


def _get_llm() -> Runnable:
    """Instantiate the LLM backend based on config."""
    if LLM_BACKEND == "huggingface":
        from langchain_huggingface import HuggingFaceEndpoint

        return HuggingFaceEndpoint(
            repo_id=HF_MODEL_ID,
            temperature=LLM_TEMPERATURE,
            max_new_tokens=1024,
        )

    # Default: Ollama local
    from langchain_community.llms import Ollama

    return Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=LLM_TEMPERATURE,
    )


# ─── Chain ───────────────────────────────────────────────────────────


def build_chain() -> Runnable:
    """Build the RAG chain: question → retrieve → prompt → LLM → text.

    Returns:
        A LangChain Runnable that accepts a string question and
        returns a string answer with citations.
    """
    retriever = get_retriever()
    llm = _get_llm()
    prompt = ChatPromptTemplate.from_template(SYSTEM_TEMPLATE)

    chain: Runnable = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def ask(question: str) -> str:
    """One-shot convenience: build chain and invoke.

    Useful for quick testing in a REPL or notebook:

        >>> from src.chain import ask
        >>> print(ask("What did you find about travel time and disease?"))
    """
    chain = build_chain()
    return chain.invoke(question)
