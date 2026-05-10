"""
LESSON 12: Hybrid Search — BM25 + Vector + RRF + CrossEncoder
==============================================================
CONCEPT: Combining keyword and semantic search for +25-35% accuracy

WHAT THIS DOES:
  Demonstrates the HybridRetriever from retrieval/hybrid.py:
    Stage 1 — BM25: exact keyword matching (finds "Policy 4.2", "FMLA")
    Stage 2 — Vector: semantic similarity (finds synonyms)
    Stage 3 — RRF: Reciprocal Rank Fusion merges both result lists
    Stage 4 — CrossEncoder: re-ranks with full query+document attention

WHY THIS MATTERS:
  Vector search alone misses exact identifiers ("Policy 4.2", "Section 3.1").
  BM25 alone misses synonyms ("vacation" ≠ "annual leave").
  Hybrid combines both: +15-20% from BM25, +10-20% from re-ranking.

BEFORE RUNNING:
  python notebooks/lessons/03_ingest_documents.py

RUN (from project root):
  python notebooks/lessons/12_hybrid_search.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from the actual project modules
from retrieval.hybrid import BM25, Document, reciprocal_rank_fusion, CrossEncoderReranker
from src.config import cfg


# ── Sample documents ──────────────────────────────────────────────────────────

SAMPLE_DOCS = [
    Document(id="d1", content="All full-time employees are entitled to 20 days of paid annual leave per calendar year."),
    Document(id="d2", content="Policy 4.2: Remote work is allowed up to 3 days per week with manager approval."),
    Document(id="d3", content="Business expenses over $50 require a receipt and manager approval."),
    Document(id="d4", content="The API rate limit is 100 requests per minute per API key."),
    Document(id="d5", content="Performance reviews are conducted twice a year, in June and December."),
    Document(id="d6", content="FMLA provides up to 12 weeks of unpaid leave for qualifying medical reasons."),
    Document(id="d7", content="Section 3.1.2: All employees must complete security training annually."),
    Document(id="d8", content="Vacation days can be carried over up to 10 days into the next calendar year."),
]


# ── Demo 1: BM25 keyword search ───────────────────────────────────────────────

def demo_bm25() -> None:
    """Show BM25 exact keyword matching."""
    print("\n--- DEMO 1: BM25 Keyword Search ---")
    print("  BM25 finds documents by EXACT word matching.")
    print("  Excellent for identifiers, acronyms, and section numbers.\n")

    bm25 = BM25()
    bm25.index(SAMPLE_DOCS)

    queries = [
        ("Policy 4.2",           "Exact identifier — BM25 excels"),
        ("FMLA",                 "Acronym — BM25 finds exact match"),
        ("Section 3.1.2",        "Section number — BM25 excels"),
        ("How many vacation days", "Synonym query — BM25 may miss 'annual leave'"),
    ]

    for query, note in queries:
        results = bm25.search(query, top_k=3)
        print(f"  Query: \"{query}\"  ({note})")
        for i, doc in enumerate(results[:2], 1):
            print(f"    #{i} (score {doc.score:.3f}): \"{doc.content[:70]}...\"")
        print()


# ── Demo 2: Why vector search alone fails ─────────────────────────────────────

def demo_vector_search_gap() -> None:
    """Explain where vector search falls short."""
    print("\n--- DEMO 2: The Vector Search Gap ---")
    print("""
  Vector search finds MEANING but misses EXACT keywords:

  Query: "Policy 4.2"
    Vector search: returns random policy chunks (doesn't know "4.2" is special)
    BM25 search:   returns the chunk containing "Policy 4.2" ✅

  Query: "FMLA"
    Vector search: returns vague insurance/leave chunks
    BM25 search:   returns the chunk containing "FMLA" ✅

  Query: "How many vacation days do I get?"
    Vector search: returns "20 days of paid annual leave" ✅ (synonym understanding)
    BM25 search:   misses it (no word "vacation" in the document)

  SOLUTION: Use BOTH and merge the results with RRF.
""")


# ── Demo 3: Reciprocal Rank Fusion ────────────────────────────────────────────

def demo_rrf() -> None:
    """Show how RRF merges two ranked lists."""
    print("\n--- DEMO 3: Reciprocal Rank Fusion (RRF) ---")
    print("  RRF merges ranked lists using position, not raw scores.")
    print("  Score = Σ 1/(k + rank_i)  where k=60\n")

    # Simulate two ranked lists (as if from BM25 and vector search)
    bm25_results = [
        Document(id="d2", content="Policy 4.2: Remote work...", score=8.5),
        Document(id="d6", content="FMLA provides...", score=6.2),
        Document(id="d1", content="20 days of annual leave...", score=3.1),
    ]
    vector_results = [
        Document(id="d1", content="20 days of annual leave...", score=0.92),
        Document(id="d8", content="Vacation days can be carried over...", score=0.87),
        Document(id="d2", content="Policy 4.2: Remote work...", score=0.71),
    ]

    fused = reciprocal_rank_fusion([bm25_results, vector_results], k=cfg.retrieval.rrf_k)

    print(f"  BM25 top-3:   {[d.id for d in bm25_results]}")
    print(f"  Vector top-3: {[d.id for d in vector_results]}")
    print(f"  RRF merged:   {[d.id for d in fused]}")
    print(f"\n  RRF scores:")
    for doc in fused:
        print(f"    {doc.id}: {doc.score:.4f} — \"{doc.content[:50]}...\"")

    print("""
  WHY RRF uses rank position (not raw scores)?
    BM25 scores are in range 0-20. Vector scores are in range 0-1.
    You can't directly compare them. RRF uses rank position instead,
    which is scale-independent and works across any scoring system.
""")


# ── Demo 4: CrossEncoder re-ranking ──────────────────────────────────────────

def demo_crossencoder() -> None:
    """Explain CrossEncoder re-ranking."""
    print("\n--- DEMO 4: CrossEncoder Re-Ranking ---")
    print("""
  Initial retrieval uses COMPRESSED representations (embeddings).
  A CrossEncoder reads the FULL query + FULL document TOGETHER.
  This is much more accurate but slower (~100-200ms per batch).

  HOW IT WORKS:
    Bi-encoder (vector search):
      embed(query) → [0.2, 0.8, ...]   (compressed, fast)
      embed(doc)   → [0.3, 0.7, ...]   (compressed, fast)
      similarity = cosine(query_vec, doc_vec)

    Cross-encoder (re-ranking):
      score = model(query + "[SEP]" + document)  (full attention, accurate)
      The model reads both together and outputs a single relevance score.

  ACCURACY IMPACT:
    Vector only:          baseline
    + BM25 hybrid:        +15-20%
    + CrossEncoder:       +10-20% on top of hybrid
    All combined:         +25-35% total

  LATENCY COST:
    CrossEncoder adds ~100-200ms for top-5 documents.
    LLM generation takes 5-15 seconds.
    The re-ranking cost is negligible compared to generation.
""")

    # Try to load the CrossEncoder (requires sentence-transformers)
    reranker = CrossEncoderReranker(model_name=cfg.models.reranker)
    if reranker._available:
        query = "How many vacation days do employees get?"
        results = reranker.rerank(query, SAMPLE_DOCS[:5], top_k=3)
        print(f"  Re-ranked top-3 for: \"{query}\"")
        for i, doc in enumerate(results, 1):
            print(f"    #{i} (score {doc.score:.4f}): \"{doc.content[:70]}...\"")
    else:
        print("  (CrossEncoder not available — install sentence-transformers)")
        print("  pip install sentence-transformers")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("LESSON 12: HYBRID SEARCH & RE-RANKING")
    print("=" * 60)
    print("""
HybridRetriever (retrieval/hybrid.py):
  retriever = HybridRetriever(vector_store, cfg)
  retriever.build_bm25_index(documents)
  results = retriever.retrieve("How many vacation days?")

Pipeline:
  BM25 (keywords) + Vector (semantics) → RRF (merge) → CrossEncoder (re-rank)
""")

    demo_bm25()
    demo_vector_search_gap()
    demo_rrf()
    demo_crossencoder()


if __name__ == "__main__":
    main()
