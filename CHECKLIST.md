# Sprint 1 checklist

## Setup
- [x] Create venv with uv
- [x] Install Ollama + pull llama3.2 + nomic-embed-text
- [x] Install Python dependencies

## Data
- [x] Download paper PDFs into data/papers/
- [x] Verify PDFs are text-extractable (not scanned images)
- [x] Remove off-domain papers (astrophysics) to reduce retrieval noise

## Build
- [x] ingest.py — loads, chunks, embeds, stores in ChromaDB
- [x] retriever.py — similarity search returning top-k chunks
- [x] chain.py — prompt template + LLM + output parser
- [x] app.py — Streamlit chat interface
- [x] config.py — centralised configuration
- [x] Fix deprecation: migrate to langchain-ollama

## Test
- [x] Run ingest pipeline end-to-end (6 papers, 112 pages, 779 chunks)
- [x] Ask 5 test questions, verify cited answers
- [x] Test edge case: question with no relevant context
- [x] 11 unit tests passing (test_ingest.py + test_chain.py)

## Deploy
- [ ] Option A: Deploy to Hugging Face Spaces (swap Ollama for HF Inference API)
- [ ] Option B: Record a GIF demo + screenshots for README

## Polish
- [ ] Architecture diagram in README (Mermaid or image)
- [ ] "What I learned" section in README
- [x] .gitignore (exclude chroma_db/, .venv/, __pycache__/)
- [ ] Push to GitHub as public repo
