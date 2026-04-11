"""Streamlit chat interface for the publications RAG app.

Run with:
    streamlit run src/app.py
"""

from __future__ import annotations

import streamlit as st

from src.chain import build_chain
from src.retriever import search

# ─── Page config ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ask My Research — Dr Reju Sam John",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Sidebar ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "This app uses **Retrieval-Augmented Generation** to let you "
        "chat with Dr Reju Sam John's 11 peer-reviewed publications.\n\n"
        "Answers are grounded in the actual papers — the LLM only uses "
        "retrieved context, reducing hallucination."
    )
    st.markdown("---")
    st.markdown("### Architecture")
    st.markdown(
        "1. Your question is **embedded** into a vector\n"
        "2. **ChromaDB** finds the 4 most similar paper chunks\n"
        "3. Chunks are injected as **context** into a prompt\n"
        "4. The **LLM** generates a cited answer"
    )
    st.markdown("---")
    st.markdown(
        "[GitHub](https://github.com/rejusam) · "
        "[Google Scholar](https://scholar.google.com/citations?user=EeZSlzsAAAAJ) · "
        "[Portfolio](https://rejusamjohn.pages.dev)"
    )

    show_sources = st.toggle("Show retrieved chunks", value=False)

# ─── Header ──────────────────────────────────────────────────────────

st.title("📄 Ask my research")
st.caption(
    "Chat with 11 peer-reviewed publications by Dr Reju Sam John — "
    "computational epidemiologist & data scientist"
)

# ─── Suggested questions ─────────────────────────────────────────────

SUGGESTIONS = [
    "What is the connectivity paradox in pandemic spread?",
    "How does Lassa virus transmit from rodents to humans?",
    "What methods did you use for the sarbecovirus spillover study?",
    "What did you find about Ebola persistence in primates?",
]

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("**Try one of these:**")
    cols = st.columns(2)
    for i, suggestion in enumerate(SUGGESTIONS):
        if cols[i % 2].button(suggestion, key=f"sug_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": suggestion})
            st.rerun()


# ─── Chain (cached) ──────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading model and vector store...")
def get_chain():
    """Build and cache the RAG chain across reruns."""
    return build_chain()


chain = get_chain()

# ─── Chat history ────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── User input ──────────────────────────────────────────────────────

if question := st.chat_input("Ask a question about my research..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching papers..."):
            response = chain.invoke(question)

        st.markdown(response)

        # Optionally show the raw retrieved chunks
        if show_sources:
            with st.expander("Retrieved chunks", expanded=False):
                chunks = search(question, k=4)
                for i, chunk in enumerate(chunks):
                    source = chunk.metadata.get("source_file", "unknown")
                    st.markdown(f"**Chunk {i + 1}** — `{source}`")
                    st.text(chunk.page_content[:500])
                    st.markdown("---")

    st.session_state.messages.append({"role": "assistant", "content": response})
