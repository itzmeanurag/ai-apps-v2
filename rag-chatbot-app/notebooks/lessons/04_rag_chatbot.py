"""
LESSON 04: RAG Chatbot — Retrieval + Generation Pipeline
=========================================================
CONCEPT: Grounding LLM answers in your actual documents

WHAT THIS DOES:
  1. Takes your question
  2. Searches ChromaDB for the most relevant document chunks (RETRIEVAL)
  3. Injects those chunks into the prompt (AUGMENTATION)
  4. Sends to Mistral for an answer grounded in your documents (GENERATION)

WHY THIS MATTERS:
  Without RAG, the LLM answers from its training data (may be wrong/outdated).
  With RAG, every answer is grounded in your specific documents.
  This is the core loop of the entire project.

BEFORE RUNNING:
  python notebooks/lessons/03_ingest_documents.py  (must run first!)

RUN (from project root):
  python notebooks/lessons/04_rag_chatbot.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.config import cfg


# ── Configuration ─────────────────────────────────────────────────────────────

CHROMA_DIR  = cfg.ingestion.persist_directory
COLLECTION  = cfg.ingestion.collection_name
TOP_K       = cfg.retrieval.top_k   # number of chunks to retrieve per question


# ── Helpers ───────────────────────────────────────────────────────────────────

def format_docs(docs) -> str:
    """
    Format retrieved document chunks into a single context string.

    WHY number them?
      The LLM can reference "Document 1" in its answer, giving you
      source attribution — you know which document the answer came from.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        source = Path(doc.metadata.get("source", "unknown")).name
        parts.append(f"[Document {i} — {source}]\n{doc.page_content}")
    return "\n\n".join(parts)


def show_retrieved_chunks(retriever, query: str) -> None:
    """Debug helper: show what documents were retrieved for a query."""
    docs = retriever.invoke(query)
    print(f"\n  Retrieved {len(docs)} chunk(s) for: \"{query}\"")
    for i, doc in enumerate(docs, 1):
        source = Path(doc.metadata.get("source", "?")).name
        print(f"  [{i}] {source}: \"{doc.page_content[:100]}...\"")


# ── RAG prompt ────────────────────────────────────────────────────────────────

# WHY this specific prompt structure?
#   1. "ONLY use information from the context" → prevents hallucination
#   2. "If not in context, say so" → honest about knowledge gaps
#   3. "Mention which document" → enables source attribution
#   4. "Do NOT make up information" → explicit anti-hallucination instruction
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions based ONLY on the provided context documents.

RULES:
1. ONLY use information from the context documents below to answer.
2. If the context does not contain enough information, say "I don't have that information in the available documents."
3. Mention which document your answer comes from (e.g., "According to company_policy.txt...").
4. Be concise and accurate. Do NOT make up information.

CONTEXT DOCUMENTS:
{context}"""),
    ("human", "{question}"),
])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("LESSON 04: RAG CHATBOT")
    print("=" * 60)

    # Step 1: Load the vector store from disk.
    # WHY not re-ingest? Ingestion is slow (embeds every chunk).
    # We just load the already-embedded data from disk.
    print("\nLoading vector store from disk...")
    embeddings = OllamaEmbeddings(
        model=cfg.models.embedder,
        base_url=cfg.models.ollama_base_url,
    )
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION,
    )

    # Verify documents exist
    count = vector_store._collection.count()
    if count == 0:
        print("ERROR: No documents in vector store!")
        print("Run 'python notebooks/lessons/03_ingest_documents.py' first.")
        return
    print(f"Loaded {count} document chunks.\n")

    # Step 2: Create the retriever.
    # search_type="similarity" uses cosine similarity (from Lesson 02).
    # k=TOP_K returns the TOP_K most relevant chunks.
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    # Step 3: Create the LLM.
    # temperature=0.3 → more factual, less creative (good for RAG).
    llm = ChatOllama(
        model=cfg.models.generator,
        base_url=cfg.models.ollama_base_url,
        temperature=0.3,
    )

    # Step 4: Build the RAG chain.
    # HOW IT WORKS:
    #   {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #   → retriever finds relevant chunks → format_docs formats them as text
    #   → RunnablePassthrough() passes the question through unchanged
    #   → RAG_PROMPT fills {context} and {question} placeholders
    #   → llm generates the answer
    #   → StrOutputParser() extracts the text string
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    # Step 5: Interactive chat loop
    print("=" * 60)
    print(f"RAG CHATBOT — Searching top {TOP_K} chunks per question")
    print("Commands: 'sources <question>' to see retrieved chunks, 'quit' to exit")
    print("=" * 60)

    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Special command: show retrieved chunks
        if question.lower().startswith("sources "):
            query = question[8:].strip()
            show_retrieved_chunks(retriever, query)
            continue

        print("\nAI: ", end="", flush=True)
        answer = rag_chain.invoke(question)
        print(answer)


if __name__ == "__main__":
    main()
