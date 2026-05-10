"""
evaluation/rag_monitor.py

Spec-required classes:
  RAGQualityMonitor  – rolling window, health report
  TruthfulnessScorer – ground truth testing
  GracefulDegradation – 6 failure types with helpful messages
  SemanticCache       – embed-based caching, cosine similarity

Also keeps RAGMonitor as a backward-compat alias.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional


# ── Quality metrics dataclass ─────────────────────────────────────────────────

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


# ── SemanticCache ─────────────────────────────────────────────────────────────

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
            return self._embedder.embed_query(text)
        except Exception:
            return None

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    def lookup(self, query: str) -> Optional[CacheEntry]:
        if self._embedder is None:
            return None
        emb = self._embed(query)
        if emb is None:
            return None
        now = time.time()
        with self._lock:
            best: Optional[CacheEntry] = None
            best_score = 0.0
            for entry in list(self._cache.values()):
                if now - entry.timestamp > self._ttl:
                    del self._cache[entry.query_hash]
                    continue
                score = self._cosine(emb, entry.embedding)
                if score > best_score:
                    best_score, best = score, entry
            if best_score >= self._threshold and best is not None:
                best.hits += 1
                return best
        return None

    def store(self, query: str, answer: str, context: str = "") -> None:
        if self._embedder is None:
            return
        emb = self._embed(query)
        if emb is None:
            return
        qhash = hashlib.sha256(query.encode()).hexdigest()[:16]
        with self._lock:
            if len(self._cache) >= self._max_entries:
                oldest = min(self._cache, key=lambda k: self._cache[k].timestamp)
                del self._cache[oldest]
            self._cache[qhash] = CacheEntry(
                query_hash=qhash, query=query, answer=answer,
                context=context, embedding=emb,
            )

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._cache),
                "total_hits": sum(e.hits for e in self._cache.values()),
            }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# ── RAGQualityMonitor ─────────────────────────────────────────────────────────

class RAGQualityMonitor:
    """
    Rolling-window quality monitor.

    Tracks retrieval_score, relevance, groundedness, response_time_ms
    over a configurable window and generates health reports.
    """

    def __init__(self, window_size: int = 100) -> None:
        self._window: deque[dict] = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def record(
        self,
        question: str,
        answer: str,
        retrieval_score: float = 0.0,
        relevance: float = 0.0,
        groundedness: float = 0.0,
        sources: Optional[list] = None,
        response_time_ms: float = 0.0,
    ) -> None:
        """Record a single query's quality metrics."""
        with self._lock:
            self._window.append({
                "question": question[:80],
                "retrieval_score": retrieval_score,
                "relevance": relevance,
                "groundedness": groundedness,
                "response_time_ms": response_time_ms,
                "sources_count": len(sources) if sources else 0,
                "timestamp": time.time(),
            })

    def get_health_report(self) -> dict:
        """
        Generate a health report from the rolling window.

        Returns:
            {
              "status": "HEALTHY" | "DEGRADED" | "UNHEALTHY",
              "truthfulness": {avg_relevance, avg_groundedness, pass_rate},
              "retrieval": {avg_score, avg_sources, avg_response_time_ms},
              "sample_size": int,
            }
        """
        with self._lock:
            data = list(self._window)

        if not data:
            return {"status": "NO_DATA", "truthfulness": {}, "retrieval": {}, "sample_size": 0}

        n = len(data)
        avg_rel = sum(d["relevance"] for d in data) / n
        avg_gnd = sum(d["groundedness"] for d in data) / n
        avg_ret = sum(d["retrieval_score"] for d in data) / n
        avg_rt  = sum(d["response_time_ms"] for d in data) / n
        pass_rate = sum(1 for d in data if d["relevance"] >= 0.6 and d["groundedness"] >= 0.6) / n

        if avg_rel >= 0.7 and avg_gnd >= 0.7:
            status = "HEALTHY"
        elif avg_rel >= 0.5 and avg_gnd >= 0.5:
            status = "DEGRADED"
        else:
            status = "UNHEALTHY"

        return {
            "status": status,
            "truthfulness": {
                "avg_relevance": round(avg_rel, 3),
                "avg_groundedness": round(avg_gnd, 3),
                "pass_rate": round(pass_rate, 3),
            },
            "retrieval": {
                "avg_score": round(avg_ret, 3),
                "avg_response_time_ms": round(avg_rt, 1),
            },
            "sample_size": n,
        }

    # Backward-compat methods used by RAGMonitor alias
    def evaluate(self, question: str, answer: str, context: str) -> QualityMetrics:
        return QualityMetrics(overall=0.75, explanation="RAGQualityMonitor.evaluate stub")

    def record_metrics(self, metrics: QualityMetrics, question: str) -> None:
        self.record(question=question, answer="", relevance=metrics.relevance,
                    groundedness=metrics.faithfulness)

    def should_degrade(self, metrics: QualityMetrics) -> bool:
        return metrics.overall < 0.4

    def graceful_degradation_response(self, question: str) -> str:
        return GracefulDegradation.handle("LOW_CONFIDENCE")["answer"]

    def get_average_quality(self) -> dict:
        report = self.get_health_report()
        t = report.get("truthfulness", {})
        return {
            "relevance": t.get("avg_relevance", 0.0),
            "groundedness": t.get("avg_groundedness", 0.0),
            "overall": (t.get("avg_relevance", 0.0) + t.get("avg_groundedness", 0.0)) / 2,
        }


# ── TruthfulnessScorer ────────────────────────────────────────────────────────

class TruthfulnessScorer:
    """
    Ground-truth-based truthfulness testing.

    Usage:
        scorer = TruthfulnessScorer()
        scorer.add_ground_truth("How many leave days?", "20 days per year", "leave")
        results = scorer.run_evaluation(chatbot)
    """

    def __init__(self) -> None:
        self.ground_truth: list[dict] = []

    def add_ground_truth(
        self,
        question: str,
        expected_answer: str,
        category: str = "general",
    ) -> None:
        """Add a ground-truth Q&A pair."""
        self.ground_truth.append({
            "question": question,
            "expected": expected_answer,
            "category": category,
        })

    def run_evaluation(self, chatbot: Any) -> dict:
        """
        Run all ground-truth questions through the chatbot and score accuracy.

        Returns:
            {
              "total": int,
              "passed": int,
              "accuracy": float,
              "failures": [{"question", "expected", "got"}],
              "by_category": {category: accuracy},
            }
        """
        failures = []
        by_cat: dict[str, list[bool]] = {}

        for item in self.ground_truth:
            q = item["question"]
            expected = item["expected"].lower()
            cat = item["category"]

            try:
                result = chatbot.ask(q)
                got = result.get("answer", "").lower()
                passed = expected in got or got in expected
            except Exception:
                got = ""
                passed = False

            by_cat.setdefault(cat, []).append(passed)
            if not passed:
                failures.append({"question": q, "expected": item["expected"], "got": got[:100]})

        total = len(self.ground_truth)
        passed_count = total - len(failures)
        cat_accuracy = {
            cat: round(sum(results) / len(results), 3)
            for cat, results in by_cat.items()
        }

        return {
            "total": total,
            "passed": passed_count,
            "accuracy": round(passed_count / total, 3) if total else 0.0,
            "failures": failures,
            "by_category": cat_accuracy,
        }


# ── GracefulDegradation ───────────────────────────────────────────────────────

class GracefulDegradation:
    """
    Returns helpful fallback messages for 6 failure types.

    Failure types:
      LOW_CONFIDENCE, HALLUCINATION_DETECTED, GUARDRAIL_BLOCKED,
      NO_DOCUMENTS, MODEL_ERROR, TIMEOUT
    """

    _MESSAGES: dict[str, str] = {
        "LOW_CONFIDENCE": (
            "I found some related information but I'm not confident it fully answers "
            "your question. Please try rephrasing, or contact support for assistance."
        ),
        "HALLUCINATION_DETECTED": (
            "My answer didn't seem well-supported by the available documents. "
            "I'd rather not give you unreliable information. "
            "Please try a more specific question or consult the source documents directly."
        ),
        "GUARDRAIL_BLOCKED": (
            "Your request was flagged by our content safety system and cannot be processed. "
            "Please rephrase your question to focus on work-related topics."
        ),
        "NO_DOCUMENTS": (
            "I don't have any documents loaded to answer your question. "
            "Please ask an administrator to ingest the relevant documents first."
        ),
        "MODEL_ERROR": (
            "I encountered a technical issue while generating your answer. "
            "Please try again in a moment. If the problem persists, contact IT support."
        ),
        "TIMEOUT": (
            "Your request took too long to process. "
            "Please try a shorter or simpler question."
        ),
    }

    @classmethod
    def handle(cls, failure_type: str, **context) -> dict:
        """
        Return a graceful degradation response for the given failure type.

        Returns:
            {"answer": str, "failure_type": str, "context": dict}
        """
        message = cls._MESSAGES.get(
            failure_type,
            "Something went wrong. Please try again or contact support.",
        )

        # Inject context into message if placeholders exist
        if context:
            try:
                message = message.format(**context)
            except KeyError:
                pass

        return {
            "answer": message,
            "failure_type": failure_type,
            "context": context,
        }

    @classmethod
    def list_failure_types(cls) -> list[str]:
        return list(cls._MESSAGES.keys())


# ── RAGMonitor – backward-compat alias ───────────────────────────────────────

class RAGMonitor(RAGQualityMonitor):
    """
    Backward-compatible alias for RAGQualityMonitor.
    Adds the SemanticCache and the evaluate() / should_degrade() API
    expected by the old src/chatbot.py.
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        embedder: Optional[Any] = None,
        quality_threshold: float = 0.5,
        truthfulness_threshold: float = 0.6,
        semantic_cache_threshold: float = 0.92,
        cache_ttl: int = 3600,
        window_size: int = 100,
    ) -> None:
        super().__init__(window_size=window_size)
        self._llm = llm
        self._quality_threshold = quality_threshold
        self._truthfulness_threshold = truthfulness_threshold
        self.cache = SemanticCache(
            similarity_threshold=semantic_cache_threshold,
            ttl_seconds=cache_ttl,
            embedder=embedder,
        )

    def evaluate(self, question: str, answer: str, context: str) -> QualityMetrics:
        if self._llm is None:
            return QualityMetrics(overall=1.0, explanation="LLM evaluation disabled")
        try:
            from generation.prompts import prompts as _p
            prompt = _p.get("eval_combined").format(
                question=question, answer=answer, context=context[:2000]
            )
            resp = self._llm.invoke(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            start, end = text.find("{"), text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                rel = float(data.get("relevance", 0.5))
                gnd = float(data.get("groundedness", 0.5))
                overall = (rel + gnd) / 2
                return QualityMetrics(
                    faithfulness=gnd, relevance=rel,
                    completeness=rel, overall=overall,
                    explanation=data.get("explanation", ""),
                )
        except Exception as exc:
            print(f"[monitor] Evaluation failed: {exc}")
        return QualityMetrics(overall=0.5, explanation="Evaluation parse error")

    def should_degrade(self, metrics: QualityMetrics) -> bool:
        return (
            metrics.overall < self._quality_threshold
            or metrics.truthfulness < self._truthfulness_threshold
        )

    def graceful_degradation_response(self, question: str) -> str:
        return GracefulDegradation.handle("LOW_CONFIDENCE")["answer"]

    def score_truthfulness(self, answer: str, context: str) -> float:
        if self._llm is None:
            return 1.0
        try:
            from generation.prompts import prompts as _p
            prompt = _p.get("truthfulness").format(answer=answer, context=context[:2000])
            resp = self._llm.invoke(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            start, end = text.find("{"), text.rfind("}") + 1
            if start >= 0 and end > start:
                return float(json.loads(text[start:end]).get("score", 0.5))
        except Exception:
            pass
        return 0.5
