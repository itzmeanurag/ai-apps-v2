"""
api/monitoring.py

Metrics dataclass + MetricsTracker.

Spec-required methods:
  record_request(endpoint, latency_ms, status_code, user_id, cache_hit, quality_score)
  record_auth(event, success)
  get_summary() -> dict
  print_dashboard()

Also keeps record() as a backward-compat alias.
"""
from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
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
    Thread-safe in-memory metrics tracker with optional JSON persistence.
    """

    def __init__(
        self,
        metrics_file: Optional[str] = None,
        window_size: int = 1000,
    ) -> None:
        self._file = Path(metrics_file) if metrics_file else None
        self._window = window_size
        self._lock = threading.Lock()

        self._recent: deque[RequestMetric] = deque(maxlen=window_size)
        self._total_requests: int = 0
        self._total_errors: int = 0
        self._total_cache_hits: int = 0
        self._requests_by_endpoint: dict[str, int] = defaultdict(int)
        self._errors_by_endpoint: dict[str, int] = defaultdict(int)

        # Auth tracking
        self._auth_events: dict[str, int] = defaultdict(int)  # event -> count
        self._auth_failures: int = 0

        self._load()

    # ── Spec-required methods ─────────────────────────────────────────────────

    def record_request(
        self,
        endpoint: str,
        latency_ms: float,
        status_code: int,
        user_id: Optional[str] = None,
        cache_hit: bool = False,
        quality_score: Optional[float] = None,
    ) -> None:
        """Record a single API request metric."""
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

    def record_auth(self, event: str, success: bool = True) -> None:
        """Record an authentication event."""
        with self._lock:
            self._auth_events[event] += 1
            if not success:
                self._auth_failures += 1

    def get_summary(self) -> dict:
        """Return a comprehensive metrics summary."""
        with self._lock:
            recent = list(self._recent)

        latencies = [m.latency_ms for m in recent]
        quality_scores = [m.quality_score for m in recent if m.quality_score is not None]

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
            if quality_scores else None,
            "auth_events": dict(self._auth_events),
            "auth_failures": self._auth_failures,
            "window_size": len(recent),
        }

    def print_dashboard(self) -> None:
        """Print a formatted metrics dashboard to stdout."""
        s = self.get_summary()
        print("\n" + "=" * 60)
        print("  RAG CHATBOT — METRICS DASHBOARD")
        print("=" * 60)
        print(f"  Total requests  : {s['total_requests']}")
        print(f"  Error rate      : {s['error_rate']:.1%}")
        print(f"  Cache hit rate  : {s['cache_hit_rate']:.1%}")
        lat = s["latency"]
        print(f"  Latency (avg)   : {lat['avg']:.0f} ms")
        print(f"  Latency (p95)   : {lat['p95']:.0f} ms")
        print(f"  Latency (p99)   : {lat['p99']:.0f} ms")
        if s["avg_quality_score"] is not None:
            print(f"  Avg quality     : {s['avg_quality_score']:.3f}")
        print(f"  Auth failures   : {s['auth_failures']}")
        print("\n  Requests by endpoint:")
        for ep, count in sorted(s["requests_by_endpoint"].items()):
            print(f"    {ep:<25} {count}")
        print("=" * 60 + "\n")

    # ── Backward-compat alias ─────────────────────────────────────────────────

    def record(
        self,
        endpoint: str,
        latency_ms: float,
        status_code: int,
        user_id: Optional[str] = None,
        cache_hit: bool = False,
        quality_score: Optional[float] = None,
    ) -> None:
        """Alias for record_request."""
        self.record_request(endpoint, latency_ms, status_code, user_id, cache_hit, quality_score)

    def get_endpoint_stats(self, endpoint: str) -> dict:
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
        with self._lock:
            self._recent.clear()
            self._total_requests = 0
            self._total_errors = 0
            self._total_cache_hits = 0
            self._requests_by_endpoint.clear()
            self._errors_by_endpoint.clear()
            self._auth_events.clear()
            self._auth_failures = 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _latency_stats(latencies: list[float]) -> dict:
        if not latencies:
            return {"avg": 0, "p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0}
        s = sorted(latencies)
        n = len(s)
        return {
            "avg": round(sum(s) / n, 2),
            "p50": round(s[int(n * 0.50)], 2),
            "p95": round(s[int(n * 0.95)], 2),
            "p99": round(s[min(int(n * 0.99), n - 1)], 2),
            "min": round(s[0], 2),
            "max": round(s[-1], 2),
        }

    def _persist(self) -> None:
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "total_requests": self._total_requests,
                "total_errors": self._total_errors,
                "total_cache_hits": self._total_cache_hits,
                "requests_by_endpoint": dict(self._requests_by_endpoint),
                "errors_by_endpoint": dict(self._errors_by_endpoint),
                "auth_failures": self._auth_failures,
            }
            with open(self._file, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except Exception:
            pass

    def _load(self) -> None:
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
            self._auth_failures = data.get("auth_failures", 0)
        except Exception:
            pass
