"""One-time ingestion for deployment.

Creates a ChromaDB vector store using fastembed (ONNX runtime) embeddings.
No PyTorch, no Ollama — lightweight enough to fit in 512MB RAM at query time.

Run this locally before deploying:

    python scripts/ingest_deploy.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
CHROMA_DIR = PROJECT_ROOT / "chroma_db_deploy"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    print("=" * 60)
    print("RAG Publications — Deploy Ingestion")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print("=" * 60)

    # Load PDFs
    print("\n[1/3] Loading papers...")
    pdf_paths = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {PAPERS_DIR}", file=sys.stderr)
        sys.exit(1)

    docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["source_file"] = pdf_path.stem
            page.metadata["source_path"] = str(pdf_path.name)
        docs.extend(pages)
        print(f"  {pdf_path.name} ({len(pages)} pages)")

    print(f"\nTotal: {len(docs)} pages from {len(pdf_paths)} papers")

    # Chunk
    print("\n[2/3] Chunking...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    # Embed + store
    print("\n[3/3] Embedding with fastembed (ONNX)...")
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"Vector store saved to {CHROMA_DIR}")
    print("\nDone. You can now deploy the API.")


if __name__ == "__main__":
    main()
