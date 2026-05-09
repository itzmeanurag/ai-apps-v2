"""
LESSON 16: RAG Monitoring — Quality, Truthfulness, and Health Reports
======================================================================
CONCEPT: Measuring and tracking system quality over time

WHAT THIS DOES:
  Demonstrates the RAGMonitor from evaluation/rag_monitor.py:
    - Quality evaluation (faithfulness, relevance, completeness)
    - Truthfulness scoring
    - Semantic cache (avoid re-generating identical answers)
    - Graceful degradation (safe fallback when quality is too low)
    - Average quality tracking over time

WHY THIS MATTERS:
  You can't manually review every answer in production.
  Monitoring tells you: "Is my system healthy or degrading?"
  Semantic caching reduces latency and LLM costs by 20-40%.
  Graceful degradation prevents bad answers from reaching users.

BEFORE RUNNING:
  python notebooks/lessons/03_ingest_documents.py

RUN (from project root):
  python notebooks/lessons/16_rag_monitoring.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from the actual project module (evaluation/rag_monitor.py)
from evaluation.rag_monitor import RAGMonitor, QualityMetrics, SemanticCache
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import cfg


# ── Demo 1: Quality evaluation ────────────────────────────────────────────────

def demo_quality_evaluation() -> None:
    """Show LLM-based quality evaluation."""
    print("\n--- DEMO 1: Quality Evaluation ---")
    print("  Scores each answer on faithfulness, relevance, and completeness.\n")

    llm = ChatOllama(
        model=cfg.models.generator,
        base_url=cfg.models.ollama_base_url,
        temperature=0.1,
    )
    monitor = RAGMonitor(
        llm=llm,
        quality_threshold=cfg.evaluation.quality_threshold,
        truthfulness_threshold=cfg.evaluation.truthfulness_threshold,
    )

    test_cases = [
        {
            "question": "How many leave days do employees get?",
            "answer": "Full-time employees receive 20 days of paid annual leave per year.",
            "context": "Full-time employees are entitled to 20 days of paid annual leave per calendar year.",
            "label": "GOOD — grounded answer",
        },
        {
            "question": "How many leave days do employees get?",
            "answer": "Employees get 50 days of leave per year plus unlimited sick days.",
            "context": "Full-time employees are entitled to 20 days of paid annual leave per calendar year.",
            "label": "BAD — hallucinated answer",
        },
        {
            "question": "What is the company's quantum computing policy?",
            "answer": "I don't have information about quantum computing in the available documents.",
            "context": "Full-time employees are entitled to 20 days of paid annual leave per calendar year.",
            "label": "GOOD — honest about knowledge gap",
        },
    ]

    for case in test_cases:
        print(f"  [{case['label']}]")
        print(f"  Q: {case['question']}")
        print(f"  A: {case['answer'][:80]}...")

        metrics = monitor.evaluate(case["question"], case["answer"], case["context"])
        monitor.record_metrics(metrics, case["question"])

        print(f"  Faithfulness:  {metrics.faithfulness:.2f}")
        print(f"  Relevance:     {metrics.relevance:.2f}")
        print(f"  Completeness:  {metrics.completeness:.2f}")
        print(f"  Overall:       {metrics.overall:.2f}")
        print(f"  Degrade?       {'YES — use fallback' if monitor.should_degrade(metrics) else 'NO — answer is acceptable'}")
        print()

    # Show average quality
    avg = monitor.get_average_quality()
    if avg:
        print(f"  Average quality across {len(test_cases)} evaluations:")
        for k, v in avg.items():
            print(f"    {k}: {v:.3f}")


# ── Demo 2: Graceful degradation ──────────────────────────────────────────────

def demo_graceful_degradation() -> None:
    """Show the fallback response when quality is too low."""
    print("\n--- DEMO 2: Graceful Degradation ---")
    print("  When quality is below threshold, return a safe fallback.\n")

    monitor = RAGMonitor(
        quality_threshold=cfg.evaluation.quality_threshold,
        truthfulness_threshold=cfg.evaluation.truthfulness_threshold,
    )

    # Simulate a low-quality answer
    bad_metrics = QualityMetrics(
        faithfulness=0.2,
        relevance=0.3,
        completeness=0.1,
        overall=0.2,
        explanation="Answer contains claims not in the context",
    )

    good_metrics = QualityMetrics(
        faithfulness=0.9,
        relevance=0.85,
        completeness=0.8,
        overall=0.87,
        explanation="Answer is well-grounded in the context",
    )

    print(f"  Bad answer metrics (overall: {bad_metrics.overall:.2f}):")
    print(f"    should_degrade() = {monitor.should_degrade(bad_metrics)}")
    print(f"    Fallback: \"{monitor.graceful_degradation_response('test question')}\"")

    print(f"\n  Good answer metrics (overall: {good_metrics.overall:.2f}):")
    print(f"    should_degrade() = {monitor.should_degrade(good_metrics)}")
    print(f"    → Return the actual answer to the user")


# ── Demo 3: Semantic cache ────────────────────────────────────────────────────

def demo_semantic_cache() -> None:
    """Show how the semantic cache avoids re-generating similar answers."""
    print("\n--- DEMO 3: Semantic Cache ---")
    print("  Caches answers and returns them for semantically similar queries.")
    print("  Threshold: {:.0%} similarity → cache hit\n".format(
        cfg.evaluation.semantic_cache_threshold
    ))

    # Try to use embeddings for the cache
    try:
        embeddings = OllamaEmbeddings(
            model=cfg.models.embedder,
            base_url=cfg.models.ollama_base_url,
        )
        cache = SemanticCache(
            similarity_threshold=cfg.evaluation.semantic_cache_threshold,
            ttl_seconds=300,
            embedder=embeddings,
        )

        # Store an answer
        cache.store(
            query="How many leave days do employees get?",
            answer="Full-time employees receive 20 days of paid annual leave per year.",
            context="",
        )

        # Try similar queries
        similar_queries = [
            "How many vacation days do I get?",
            "What is the annual leave entitlement?",
            "How many days off per year?",
            "What is the API rate limit?",  # unrelated — should miss
        ]

        print(f"  Stored: \"How many leave days do employees get?\"")
        print(f"\n  Testing similar queries:")
        for query in similar_queries:
            hit = cache.lookup(query)
            status = "✅ CACHE HIT" if hit else "❌ CACHE MISS"
            print(f"  {status} | \"{query}\"")
            if hit:
                print(f"           → \"{hit.answer[:60]}...\"")

        stats = cache.stats()
        print(f"\n  Cache stats: {stats}")

    except Exception as e:
        print(f"  (Semantic cache requires Ollama running: {e})")
        print("""
  HOW IT WORKS (without running):
    1. User asks "How many vacation days?"
    2. Embed the query → vector
    3. Compare against cached query vectors
    4. If similarity >= 0.92 → return cached answer (no LLM call!)
    5. If similarity < 0.92 → generate new answer, cache it

  BENEFIT: 20-40% of questions are near-duplicates.
  Cache hit = 0ms response time instead of 5-15 seconds.
""")


# ── Demo 4: Health report ─────────────────────────────────────────────────────

def demo_health_report() -> None:
    """Show how to generate a quality health report."""
    print("\n--- DEMO 4: Quality Health Report ---")
    print("""
  The RAGMonitor tracks metrics over time and can generate health reports.

  In production (api/server.py):
    GET /metrics → returns MetricsTracker.get_summary()
    {
      "total_requests": 1247,
      "error_rate": 0.023,
      "cache_hit_rate": 0.31,
      "latency": {"avg": 4823, "p95": 12100, "p99": 18400},
      "avg_quality_score": 0.74
    }

  WHAT TO MONITOR:
    avg_quality_score < 0.6  → retrieval or generation degrading
    cache_hit_rate < 0.1     → queries are too diverse (normal) or cache broken
    error_rate > 0.05        → something is broken
    latency.p95 > 30000ms    → LLM is overloaded or model is too large

  ALERTING THRESHOLDS (suggested):
    quality_score < 0.5      → page on-call
    error_rate > 0.1         → page on-call
    latency.p99 > 60000ms    → page on-call
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("LESSON 16: RAG MONITORING")
    print("=" * 60)
    print(f"""
RAGMonitor (evaluation/rag_monitor.py):
  monitor = RAGMonitor(llm=llm, embedder=embeddings)
  metrics = monitor.evaluate(question, answer, context)
  if monitor.should_degrade(metrics):
      return monitor.graceful_degradation_response(question)
  cached = monitor.cache.lookup(question)

Quality threshold:     {cfg.evaluation.quality_threshold}
Truthfulness threshold: {cfg.evaluation.truthfulness_threshold}
Cache similarity:       {cfg.evaluation.semantic_cache_threshold}
""")

    demo_quality_evaluation()
    demo_graceful_degradation()
    demo_semantic_cache()
    demo_health_report()


if __name__ == "__main__":
    main()
