"""
LESSON 02: Embeddings and Vector Similarity
============================================
CONCEPT: How text becomes searchable numbers (vectors)

WHAT THIS DOES:
  Converts sentences to 768-dimensional vectors using nomic-embed-text,
  then measures how similar they are using cosine similarity.
  This is the mathematical foundation of all document search in RAG.

WHY THIS MATTERS:
  When you ask "How many vacation days do I get?", the system finds
  "Employees receive 20 days of annual leave" — even though none of
  the words match. That's because both sentences have similar vectors.
  Without embeddings, RAG is impossible.

BEFORE RUNNING:
  ollama pull nomic-embed-text

RUN (from project root):
  python notebooks/lessons/02_embeddings.py
"""

import math
from langchain_ollama import OllamaEmbeddings


# ── Cosine similarity ─────────────────────────────────────────────────────────

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Measure how similar two vectors are.

    Returns a value between -1 and 1:
      1.0  = identical meaning (same direction in vector space)
      0.0  = completely unrelated (perpendicular)
     -1.0  = opposite meaning (opposite direction)

    HOW IT WORKS (visual analogy):
      Imagine two arrows pointing from the center of a sphere.
      If they point the same direction → similarity = 1.0
      If they point at right angles   → similarity = 0.0
      If they point opposite ways     → similarity = -1.0

    WHY cosine and not Euclidean distance?
      Cosine measures DIRECTION (meaning), not magnitude (length).
      "cat" and "cats" have similar direction but different lengths.
      Cosine correctly says they're similar; Euclidean might not.
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


def score_bar(score: float, width: int = 30) -> str:
    """Visual bar for a similarity score."""
    filled = int(max(score, 0) * width)
    return "█" * filled + "░" * (width - filled)


# ── Demos ─────────────────────────────────────────────────────────────────────

def demo_what_is_an_embedding() -> None:
    """Show what an embedding actually looks like."""
    print("\n--- DEMO 1: What Does an Embedding Look Like? ---")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    text = "The cat sat on the mat"
    vector = embeddings.embed_query(text)

    print(f"Text:       \"{text}\"")
    print(f"Dimensions: {len(vector)}  (768 numbers per sentence)")
    print(f"First 8:    {[round(v, 4) for v in vector[:8]]}")
    print(f"Last 8:     {[round(v, 4) for v in vector[-8:]]}")
    print("""
WHY 768 dimensions?
  Each dimension captures a different aspect of meaning.
  Some dimensions encode grammar, some encode topic, some encode sentiment.
  768 dimensions gives enough resolution to distinguish millions of concepts.
  (AWS TITAN uses 1024 dimensions; all-MiniLM uses 384 — more/fewer is a tradeoff.)
""")


def demo_similarity_comparison() -> None:
    """Compare similarity between sentence pairs."""
    print("\n--- DEMO 2: Similarity Between Sentence Pairs ---")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    sentences = [
        "The cat sat on the mat",                    # 0
        "A kitten was resting on the rug",           # 1 — similar to 0
        "The dog played in the park",                # 2 — different animal
        "Python is a programming language",          # 3 — unrelated topic
        "Java is used for software development",     # 4 — similar to 3
    ]

    print("Generating embeddings...")
    vectors = embeddings.embed_documents(sentences)
    print(f"Generated {len(vectors)} vectors, each with {len(vectors[0])} dimensions.\n")

    comparisons = [
        (0, 1, "cat vs kitten (same meaning, different words)"),
        (0, 2, "cat vs dog (same category, different animal)"),
        (0, 3, "cat vs Python (completely unrelated)"),
        (3, 4, "Python vs Java (both programming languages)"),
        (1, 4, "kitten vs Java (completely unrelated)"),
    ]

    print(f"{'Pair':<45} {'Score':>6}  {'Visual'}")
    print("-" * 75)
    for i, j, description in comparisons:
        score = cosine_similarity(vectors[i], vectors[j])
        bar = score_bar(score, 20)
        print(f"{description:<45} {score:>6.4f}  {bar}")

    print("""
KEY INSIGHT:
  The model understands that "cat" ≈ "kitten" and "Python" ≈ "Java"
  even though the words are completely different.
  This is SEMANTIC understanding, not keyword matching.
""")


def demo_rag_search_simulation() -> None:
    """Simulate how RAG finds the most relevant document chunk."""
    print("\n--- DEMO 3: Simulating a RAG Search ---")
    print("This is exactly what happens when you ask the chatbot a question.\n")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # These represent document chunks stored in ChromaDB
    document_chunks = [
        "Full-time employees receive 20 days of paid annual leave per year.",
        "Remote work is permitted up to 3 days per week with manager approval.",
        "Business expenses over $50 require a receipt and manager sign-off.",
        "The API rate limit is 100 requests per minute per API key.",
        "Performance reviews are conducted twice a year, in June and December.",
        "Python is a high-level programming language known for readability.",
    ]

    # This is the user's question
    query = "How many vacation days do employees get?"

    print(f"User query: \"{query}\"\n")
    print("Searching document chunks by semantic similarity...\n")

    # Embed everything
    doc_vectors = embeddings.embed_documents(document_chunks)
    query_vector = embeddings.embed_query(query)

    # Score each chunk against the query
    results = []
    for chunk, vec in zip(document_chunks, doc_vectors):
        score = cosine_similarity(query_vector, vec)
        results.append((score, chunk))

    # Sort by score (highest first) — this is what ChromaDB does internally
    results.sort(reverse=True)

    print(f"{'Rank':<5} {'Score':>6}  {'Chunk (first 60 chars)'}")
    print("-" * 75)
    for rank, (score, chunk) in enumerate(results, 1):
        marker = " ← TOP RESULT" if rank == 1 else ""
        print(f"#{rank:<4} {score:>6.4f}  \"{chunk[:60]}...\"  {marker}")

    print(f"""
RESULT: The system correctly identified that "vacation days" maps to
"annual leave" — even though those exact words don't appear in the query.
This is why vector search beats keyword search for natural language.
""")


def main() -> None:
    print("=" * 60)
    print("LESSON 02: EMBEDDINGS AND VECTOR SIMILARITY")
    print("=" * 60)
    print("""
KEY CONCEPT: Text → Numbers → Similarity Search
  embedding = model.embed_query("some text")  # returns list of 768 floats
  similarity = cosine_similarity(vec_a, vec_b) # returns 0.0 to 1.0

ChromaDB stores these vectors and finds the closest ones to your query.
That's the entire secret of semantic search.
""")

    demo_what_is_an_embedding()
    demo_similarity_comparison()
    demo_rag_search_simulation()


if __name__ == "__main__":
    main()
