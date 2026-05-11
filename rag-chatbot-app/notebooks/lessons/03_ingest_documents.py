"""
LESSON 03: Document Ingestion — Loading Documents into ChromaDB
================================================================
CONCEPT: The one-time pipeline that makes your documents searchable

WHAT THIS DOES:
  1. Reads all .txt / .md / .pdf files from data/documents/
  2. Splits them into overlapping chunks (so context isn't lost at boundaries)
  3. Converts each chunk to a 768-dim vector via nomic-embed-text
  4. Stores vectors + text + metadata in ChromaDB on disk

WHY THIS MATTERS:
  You only run this once (or when documents change).
  After ingestion, every question can search all your documents in <100ms.
  Without this step, the chatbot has no knowledge of your documents.

RUN (from project root):
  python notebooks/lessons/03_ingest_documents.py
"""

import sys
from pathlib import Path

# ── Resolve project root regardless of where the script is run from ───────────
# __file__ = rag-chatbot-app/notebooks/lessons/03_ingest_documents.py
# .parent        → rag-chatbot-app/notebooks/lessons/
# .parent.parent → rag-chatbot-app/notebooks/
# .parent.parent.parent → rag-chatbot-app/   ← project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

from src.config import cfg  # dot-access config: cfg.ingestion.chunk_size


# ── Configuration (from config.yaml via cfg) ──────────────────────────────────
# All paths are resolved relative to the project root so the script works
# whether you run it from the repo root, notebooks/, or notebooks/lessons/.

DOCUMENTS_DIR = str(PROJECT_ROOT / "data" / "documents")
CHROMA_DIR    = str(PROJECT_ROOT / cfg.ingestion.persist_directory.lstrip("./"))
COLLECTION    = cfg.ingestion.collection_name     # "rag_documents"
CHUNK_SIZE    = cfg.ingestion.chunk_size          # 512
CHUNK_OVERLAP = cfg.ingestion.chunk_overlap       # 64


# ── Step functions ────────────────────────────────────────────────────────────

def load_documents(directory: str) -> list:
    """
    Load all text/markdown files from a directory.

    WHY DirectoryLoader?
      It recursively finds all matching files and loads them with the
      correct loader. We use TextLoader for .txt and .md files.
      For PDFs you'd add PyPDFLoader; for Word docs, UnstructuredWordDocumentLoader.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"[ingest] Directory not found: {directory}")
        return []

    loader = DirectoryLoader(
        str(dir_path),
        glob="**/*.txt",          # match all .txt files recursively
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        silent_errors=True,       # skip unreadable files instead of crashing
    )
    docs = loader.load()

    # Also load .md files
    md_loader = DirectoryLoader(
        str(dir_path),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        silent_errors=True,
    )
    docs += md_loader.load()

    print(f"[ingest] Loaded {len(docs)} document(s) from {directory}")
    return docs


def chunk_documents(documents: list) -> list:
    """
    Split documents into overlapping chunks.

    WHY chunk at all?
      LLMs have a context window limit (~32K tokens for Mistral).
      A 100-page PDF won't fit. We split it into pieces that do.

    WHY overlap?
      If a sentence spans a chunk boundary, overlap ensures neither
      chunk loses the full context of that sentence.

    WHY RecursiveCharacterTextSplitter?
      It tries to split on natural boundaries in order:
        1. Double newlines (paragraph breaks) — preferred
        2. Single newlines
        3. Spaces (word boundaries)
        4. Characters (last resort)
      This keeps paragraphs and sentences together when possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Add a unique ID to each chunk's metadata for retrieval tracking
    for i, chunk in enumerate(chunks):
        chunk.metadata["id"] = f"chunk-{i:06d}"
        chunk.metadata.setdefault("source", "unknown")

    print(f"[ingest] Split into {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def store_in_chromadb(chunks: list) -> Chroma:
    """
    Embed each chunk and store in ChromaDB.

    WHAT HAPPENS INSIDE Chroma.from_documents():
      For each chunk:
        1. Send chunk.page_content to nomic-embed-text → get 768-dim vector
        2. Store (vector, text, metadata) in ChromaDB on disk

    WHY ChromaDB?
      File-based, no server needed, persists to disk.
      For production with many concurrent users, use PostgreSQL + pgvector.
    """
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)

    embeddings = OllamaEmbeddings(
        model=cfg.models.embedder,
        base_url=cfg.models.ollama_base_url,
    )

    print(f"[ingest] Embedding {len(chunks)} chunks with {cfg.models.embedder}...")
    print("[ingest] This may take a minute on first run...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION,
    )

    print(f"[ingest] Stored in ChromaDB at {CHROMA_DIR}/ (collection: {COLLECTION})")
    return vector_store


def verify_ingestion(vector_store: Chroma) -> None:
    """Run a quick test search to confirm ingestion worked."""
    print("\n[ingest] Verifying with a test search...")
    test_query = "leave policy"
    results = vector_store.similarity_search(test_query, k=2)
    if results:
        print(f"[ingest] ✅ Test search for '{test_query}' returned {len(results)} result(s):")
        for r in results:
            print(f"         → \"{r.page_content[:80]}...\"")
    else:
        print("[ingest] ⚠️  Test search returned no results. Check your documents.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("LESSON 03: DOCUMENT INGESTION PIPELINE")
    print("=" * 60)
    print(f"""
Pipeline:
  {DOCUMENTS_DIR}/  →  load  →  chunk  →  embed  →  ChromaDB ({CHROMA_DIR}/)

Settings (from config.yaml):
  chunk_size    = {CHUNK_SIZE}
  chunk_overlap = {CHUNK_OVERLAP}
  embedder      = {cfg.models.embedder}
  collection    = {COLLECTION}
""")

    # Step 1: Load
    documents = load_documents(DOCUMENTS_DIR)
    if not documents:
        print(f"\n⚠️  No documents found in {DOCUMENTS_DIR}/")
        print("   The sample documents should already be there.")
        print("   Add your own .txt or .md files and run again.")
        return

    # Step 2: Chunk
    chunks = chunk_documents(documents)

    # Show a sample chunk so you can see what gets stored
    if chunks:
        print(f"\nSample chunk (first 200 chars):")
        print(f"  \"{chunks[0].page_content[:200]}...\"")
        print(f"  Metadata: {chunks[0].metadata}")

    # Step 3: Embed + Store
    vector_store = store_in_chromadb(chunks)

    # Step 4: Verify
    verify_ingestion(vector_store)

    print(f"""
{'=' * 60}
INGESTION COMPLETE
{'=' * 60}
  Documents loaded : {len(documents)}
  Chunks created   : {len(chunks)}
  Vector store     : {CHROMA_DIR}/
  Collection       : {COLLECTION}

Next step: python notebooks/lessons/04_rag_chatbot.py
""")


if __name__ == "__main__":
    main()
