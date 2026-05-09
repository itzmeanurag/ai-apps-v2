"""
LESSON 13: Streaming — Token-by-Token Output
=============================================
CONCEPT: Show answers as they're generated instead of waiting for completion

WHAT THIS DOES:
  Demonstrates streaming from src/chatbot.py:
    - Synchronous streaming with chain.stream()
    - Async streaming for FastAPI StreamingResponse
    - Time-to-first-token vs total time comparison
    - Streaming in the full RAG pipeline

WHY THIS MATTERS:
  Without streaming, users wait 5-15 seconds staring at a blank screen.
  With streaming, the first word appears in <1 second.
  This dramatically improves perceived performance and user experience.

RUN (from project root):
  python notebooks/lessons/13_streaming.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import cfg


# ── Demo 1: Non-streaming vs streaming comparison ─────────────────────────────

def demo_streaming_comparison() -> None:
    """Compare the user experience of streaming vs non-streaming."""
    print("\n--- DEMO 1: Non-Streaming vs Streaming ---")

    llm = ChatOllama(
        model=cfg.models.generator,
        base_url=cfg.models.ollama_base_url,
        temperature=0.7,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer clearly."),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    question = "Explain what RAG is in 3 sentences."

    # Without streaming
    print(f"\n  Question: \"{question}\"")
    print("\n  [WITHOUT STREAMING]")
    print("  Waiting for complete response", end="", flush=True)
    start = time.time()
    response = chain.invoke({"question": question})
    elapsed = time.time() - start
    print(f" ({elapsed:.1f}s)")
    print(f"  Answer: {response[:150]}...")

    # With streaming
    print("\n  [WITH STREAMING]")
    print("  Answer: ", end="", flush=True)
    start = time.time()
    first_token_time = None
    token_count = 0

    for chunk in chain.stream({"question": question}):
        if first_token_time is None:
            first_token_time = time.time() - start
        print(chunk, end="", flush=True)
        token_count += 1

    total_time = time.time() - start
    print(f"\n\n  Time to first token: {first_token_time:.2f}s  (user sees something immediately)")
    print(f"  Total time:          {total_time:.1f}s")
    print(f"  Tokens generated:    {token_count}")
    print("""
  KEY INSIGHT:
    Non-streaming: user waits the FULL time before seeing anything.
    Streaming: user sees the first word in <1 second.
    Same total time, but streaming FEELS much faster.
""")


# ── Demo 2: Streaming in the RAG pipeline ────────────────────────────────────

def demo_rag_streaming() -> None:
    """Show streaming in the full RAG pipeline."""
    print("\n--- DEMO 2: Streaming in the RAG Pipeline ---")

    # Load vector store
    embeddings = OllamaEmbeddings(
        model=cfg.models.embedder,
        base_url=cfg.models.ollama_base_url,
    )
    vector_store = Chroma(
        persist_directory=cfg.ingestion.persist_directory,
        embedding_function=embeddings,
        collection_name=cfg.ingestion.collection_name,
    )

    if vector_store._collection.count() == 0:
        print("  No documents ingested. Run 03_ingest_documents.py first.")
        print("  Skipping RAG streaming demo.")
        return

    retriever = vector_store.as_retriever(search_kwargs={"k": cfg.retrieval.top_k})
    llm = ChatOllama(
        model=cfg.models.generator,
        base_url=cfg.models.ollama_base_url,
        temperature=0.3,
    )

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based ONLY on the context below.\n\nContext:\n{context}"),
        ("human", "{question}"),
    ])

    question = "What is the leave policy?"
    print(f"\n  Question: \"{question}\"")

    # Retrieve first (non-streaming)
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    print(f"  Retrieved {len(docs)} chunks. Streaming answer...\n")

    # Stream the generation
    chain = rag_prompt | llm | StrOutputParser()
    print("  AI: ", end="", flush=True)
    for chunk in chain.stream({"context": context, "question": question}):
        print(chunk, end="", flush=True)
    print("\n")


# ── Demo 3: Streaming from RAGChatbot ─────────────────────────────────────────

def demo_chatbot_streaming() -> None:
    """Show the stream() method on the RAGChatbot class."""
    print("\n--- DEMO 3: RAGChatbot.stream() ---")
    print("  The full chatbot (src/chatbot.py) has a stream() method.")
    print("  It runs the complete pipeline: sanitize → retrieve → generate.\n")

    print("""
  Usage:
    from src.chatbot import RAGChatbot
    bot = RAGChatbot()

    # Synchronous streaming
    for chunk in bot.stream("What is the leave policy?"):
        print(chunk, end="", flush=True)

    # Async streaming (for FastAPI)
    async for chunk in bot.astream("What is the leave policy?"):
        yield chunk  # StreamingResponse

  In the FastAPI server (api/server.py):
    @app.post("/ask")
    async def ask(body: AskRequest):
        if body.stream:
            async def generate():
                async for chunk in chatbot.astream(body.question):
                    yield chunk
            return StreamingResponse(generate(), media_type="text/plain")
""")


# ── Demo 4: Interactive streaming chat ────────────────────────────────────────

def demo_interactive_streaming() -> None:
    """Interactive chat with streaming output."""
    print("\n--- DEMO 4: Interactive Streaming Chat ---")
    print("  Type questions and see answers stream token by token.")
    print("  Type 'quit' to exit.\n")

    llm = ChatOllama(
        model=cfg.models.generator,
        base_url=cfg.models.ollama_base_url,
        temperature=0.7,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer clearly and concisely."),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()

    while True:
        question = input("  You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break

        print("  AI: ", end="", flush=True)
        for chunk in chain.stream({"question": question}):
            print(chunk, end="", flush=True)
        print("\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("LESSON 13: STREAMING RESPONSES")
    print("=" * 60)
    print("""
Streaming pattern:
  for chunk in chain.stream({"question": q}):
      print(chunk, end="", flush=True)

vs non-streaming:
  response = chain.invoke({"question": q})  # waits for full response

Same total time. Streaming shows first token in <1 second.
""")

    demo_streaming_comparison()
    demo_rag_streaming()
    demo_chatbot_streaming()
    demo_interactive_streaming()


if __name__ == "__main__":
    main()
