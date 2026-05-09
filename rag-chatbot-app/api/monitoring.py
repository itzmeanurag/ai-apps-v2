"""
api/monitoring.py
Metrics tracker: requests, latency, quality scores, cache hits.
"""
from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RequestMetric:
    endpoint: str
    user_id: Optional[str]
    latency_ms: float
    status_code: int
    cache_hit: bool = False
    quality_score: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


class MetricsTracker:
    """
    In-memory metrics tracker with optional JSON persistence.

    Tracks:
    - Total requests per endpoint
    - Average / p95 / p99 latency
    - Error rate
    - Cache hit rate
    - Average quality score
    """

    def __init__(
        self,
        metrics_file: Optional[str] = None,
        window_size: int = 1000,
    ) -> None:
        self._file = Path(metrics_file) if metrics_file else None
        self._window = window_size
        self._lock = threading.Lock()

        # Rolling window of recent requests
        self._recent: deque[RequestMetric] = deque(maxlen=window_size)

        # Cumulative counters
        self._total_requests: int = 0
        self._total_errors: int = 0
        self._total_cache_hits: int = 0
        self._requests_by_endpoint: dict[str, int] = defaultdict(int)
        self._errors_by_endpoint: dict[str, int] = defaultdict(int)

        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def record(
        self,
        endpoint: str,
        latency_ms: float,
        status_code: int,
        user_id: Optional[str] = None,
        cache_hit: bool = False,
        quality_score: Optional[float] = None,
    ) -> None:
        """Record a single request metric."""
        metric = RequestMetric(
            endpoint=endpoint,
            user_id=user_id,
            latency_ms=latency_ms,
            status_code=status_code,
            cache_hit=cache_hit,
            quality_score=quality_score,
        )
        with self._lock:
            self._recent.append(metric)
            self._total_requests += 1
            self._requests_by_endpoint[endpoint] += 1
            if status_code >= 400:
                self._total_errors += 1
                self._errors_by_endpoint[endpoint] += 1
            if cache_hit:
                self._total_cache_hits += 1

        if self._file:
            self._persist()

    def get_summary(self) -> dict:
        """Return a summary of all tracked metrics."""
        with self._lock:
            recent = list(self._recent)

        latencies = [m.latency_ms for m in recent]
        quality_scores = [m.quality_score for m in recent if m.quality_score is not None]
        cache_hits = sum(1 for m in recent if m.cache_hit)

        return {
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
            "error_rate": round(self._total_errors / max(self._total_requests, 1), 4),
            "total_cache_hits": self._total_cache_hits,
            "cache_hit_rate": round(self._total_cache_hits / max(self._total_requests, 1), 4),
            "requests_by_endpoint": dict(self._requests_by_endpoint),
            "errors_by_endpoint": dict(self._errors_by_endpoint),
            "latency": self._latency_stats(latencies),
            "avg_quality_score": round(sum(quality_scores) / len(quality_scores), 3)
            if quality_scores
            else None,
            "window_size": len(recent),
        }

    def get_endpoint_stats(self, endpoint: str) -> dict:
        """Return stats for a specific endpoint."""
        with self._lock:
            recent = [m for m in self._recent if m.endpoint == endpoint]
        latencies = [m.latency_ms for m in recent]
        return {
            "endpoint": endpoint,
            "requests": self._requests_by_endpoint.get(endpoint, 0),
            "errors": self._errors_by_endpoint.get(endpoint, 0),
            "latency": self._latency_stats(latencies),
        }

    def reset(self) -> None:
        """Clear all metrics (use with caution)."""
        with self._lock:
            self._recent.clear()
            self._total_requests = 0
            self._total_errors = 0
            self._total_cache_hits = 0
            self._requests_by_endpoint.clear()
            self._errors_by_endpoint.clear()

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _latency_stats(latencies: list[float]) -> dict:
        if not latencies:
            return {"avg": 0, "p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0}
        sorted_l = sorted(latencies)
        n = len(sorted_l)
        return {
            "avg": round(sum(sorted_l) / n, 2),
            "p50": round(sorted_l[int(n * 0.50)], 2),
            "p95": round(sorted_l[int(n * 0.95)], 2),
            "p99": round(sorted_l[min(int(n * 0.99), n - 1)], 2),
            "min": round(sorted_l[0], 2),
            "max": round(sorted_l[-1], 2),
        }

    def _persist(self) -> None:
        """Save cumulative counters to JSON (non-blocking best-effort)."""
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "total_requests": self._total_requests,
                "total_errors": self._total_errors,
                "total_cache_hits": self._total_cache_hits,
                "requests_by_endpoint": dict(self._requests_by_endpoint),
                "errors_by_endpoint": dict(self._errors_by_endpoint),
            }
            with open(self._file, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except Exception:
            pass  # metrics persistence is best-effort

    def _load(self) -> None:
        """Restore cumulative counters from disk on startup."""
        if self._file is None or not self._file.exists():
            return
        try:
            with open(self._file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._total_requests = data.get("total_requests", 0)
            self._total_errors = data.get("total_errors", 0)
            self._total_cache_hits = data.get("total_cache_hits", 0)
            self._requests_by_endpoint = defaultdict(int, data.get("requests_by_endpoint", {}))
            self._errors_by_endpoint = defaultdict(int, data.get("errors_by_endpoint", {}))
        except Exception:
            pass
