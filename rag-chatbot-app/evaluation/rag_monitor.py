"""
evaluation/rag_monitor.py
Quality monitor, truthfulness scorer, graceful degradation, semantic cache.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ── Quality metrics ───────────────────────────────────────────────────────────

@dataclass
class QualityMetrics:
    faithfulness: float = 0.0
    relevance: float = 0.0
    completeness: float = 0.0
    truthfulness: float = 0.0
    overall: float = 0.0
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "faithfulness": self.faithfulness,
            "relevance": self.relevance,
            "completeness": self.completeness,
            "truthfulness": self.truthfulness,
            "overall": self.overall,
            "explanation": self.explanation,
        }


# ── Semantic cache ────────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    query_hash: str
    query: str
    answer: str
    context: str
    embedding: list[float]
    timestamp: float = field(default_factory=time.time)
    hits: int = 0


class SemanticCache:
    """
    Embedding-based semantic cache.
    Returns cached answers for semantically similar queries.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        max_entries: int = 500,
        ttl_seconds: int = 3600,
        embedder: Optional[Any] = None,
    ) -> None:
        self._threshold = similarity_threshold
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self._embedder = embedder
        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()

    def _embed(self, text: str) -> Optional[list[float]]:
        if self._embedder is None:
            return None
        try:
            result = self._embedder.embed_query(text)
            return result
        except Exception:
            return None

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def lookup(self, query: str) -> Optional[CacheEntry]:
        """Return a cached entry if a semantically similar query exists."""
        if self._embedder is None:
            return None

        query_emb = self._embed(query)
        if query_emb is None:
            return None

        now = time.time()
        with self._lock:
            best_entry: Optional[CacheEntry] = None
            best_score = 0.0

            for entry in list(self._cache.values()):
                # Evict expired entries
                if now - entry.timestamp > self._ttl:
                    del self._cache[entry.query_hash]
                    continue
                score = self._cosine_similarity(query_emb, entry.embedding)
                if score > best_score:
                    best_score = score
                    best_entry = entry

            if best_score >= self._threshold and best_entry is not None:
                best_entry.hits += 1
                return best_entry
        return None

    def store(self, query: str, answer: str, context: str = "") -> None:
        """Cache a query-answer pair."""
        if self._embedder is None:
            return

        embedding = self._embed(query)
        if embedding is None:
            return

        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self._max_entries:
                oldest_key = min(self._cache, key=lambda k: self._cache[k].timestamp)
                del self._cache[oldest_key]

            self._cache[query_hash] = CacheEntry(
                query_hash=query_hash,
                query=query,
                answer=answer,
                context=context,
                embedding=embedding,
            )

    def stats(self) -> dict:
        with self._lock:
            total_hits = sum(e.hits for e in self._cache.values())
            return {
                "entries": len(self._cache),
                "total_hits": total_hits,
            }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# ── RAG quality monitor ───────────────────────────────────────────────────────

class RAGMonitor:
    """
    Monitors RAG pipeline quality with:
    - Automatic evaluation via LLM
    - Truthfulness scoring
    - Graceful degradation (fallback answers)
    - Semantic caching
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        embedder: Optional[Any] = None,
        quality_threshold: float = 0.5,
        truthfulness_threshold: float = 0.6,
        semantic_cache_threshold: float = 0.92,
        cache_ttl: int = 3600,
    ) -> None:
        self._llm = llm
        self._quality_threshold = quality_threshold
        self._truthfulness_threshold = truthfulness_threshold
        self.cache = SemanticCache(
            similarity_threshold=semantic_cache_threshold,
            ttl_seconds=cache_ttl,
            embedder=embedder,
        )
        self._metrics_history: list[dict] = []
        self._lock = threading.Lock()

    def evaluate(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> QualityMetrics:
        """Run LLM-based evaluation. Returns QualityMetrics."""
        if self._llm is None:
            return QualityMetrics(overall=1.0, explanation="LLM evaluation disabled")

        try:
            from generation.prompts import PromptAssembler
            pa = PromptAssembler()
            prompt = pa.build("eval", question=question, answer=answer, context=context[:2000])
            response = self._llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)

            # Parse JSON response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                metrics = QualityMetrics(
                    faithfulness=float(data.get("faithfulness", 0)),
                    relevance=float(data.get("relevance", 0)),
                    completeness=float(data.get("completeness", 0)),
                    explanation=data.get("explanation", ""),
                )
                metrics.overall = (
                    metrics.faithfulness * 0.4
                    + metrics.relevance * 0.4
                    + metrics.completeness * 0.2
                )
                return metrics
        except Exception as exc:
            print(f"[monitor] Evaluation failed: {exc}")

        return QualityMetrics(overall=0.5, explanation="Evaluation parse error")

    def score_truthfulness(self, answer: str, context: str) -> float:
        """Return a truthfulness score 0-1."""
        if self._llm is None:
            return 1.0

        try:
            from generation.prompts import PromptAssembler
            pa = PromptAssembler()
            prompt = pa.build("truthfulness", answer=answer, context=context[:2000])
            response = self._llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)

            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                return float(data.get("score", 0.5))
        except Exception as exc:
            print(f"[monitor] Truthfulness scoring failed: {exc}")

        return 0.5

    def should_degrade(self, metrics: QualityMetrics) -> bool:
        """Return True if quality is below threshold (trigger graceful degradation)."""
        return (
            metrics.overall < self._quality_threshold
            or metrics.truthfulness < self._truthfulness_threshold
        )

    def graceful_degradation_response(self, question: str) -> str:
        """Return a safe fallback response when quality is too low."""
        return (
            "I wasn't able to find a reliable answer to your question in the available documents. "
            "Please try rephrasing your question, or contact support for assistance."
        )

    def record_metrics(self, metrics: QualityMetrics, question: str) -> None:
        """Store metrics for trend analysis."""
        with self._lock:
            self._metrics_history.append({
                "question": question[:100],
                "metrics": metrics.to_dict(),
                "timestamp": time.time(),
            })
            # Keep last 1000 entries
            if len(self._metrics_history) > 1000:
                self._metrics_history = self._metrics_history[-1000:]

    def get_average_quality(self) -> dict:
        """Return average quality metrics over recorded history."""
        with self._lock:
            if not self._metrics_history:
                return {}
            keys = ["faithfulness", "relevance", "completeness", "overall"]
            totals = {k: 0.0 for k in keys}
            for entry in self._metrics_history:
                for k in keys:
                    totals[k] += entry["metrics"].get(k, 0)
            n = len(self._metrics_history)
            return {k: round(v / n, 3) for k, v in totals.items()}
