"""
api/audit.py

AuditLogger – thread-safe JSONL audit logger.

Spec-required methods:
  log_question(user_id, question, answer, session_id, quality, latency_ms, cached, blocked)
  log_auth_event(event, username, ip, success, role)
  log_admin_action(user_id, action, details)
  query_logs(event_type, user_id, limit) -> list[dict]
  get_stats() -> dict

Also keeps the generic log() method for backward compat.
"""
from __future__ import annotations

import json
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class AuditLogger:
    """
    Append-only, thread-safe JSONL audit logger.

    Usage:
        logger = AuditLogger("./data/audit.jsonl")
        logger.log_question("usr-001", "What is PTO?", "20 days.", "sess-1")
        logger.log_auth_event("login", "admin", "127.0.0.1", True, "admin")
    """

    def __init__(self, log_file: str = "./data/audit.jsonl") -> None:
        self._path = Path(log_file)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ── Spec-required methods ─────────────────────────────────────────────────

    def log_question(
        self,
        user_id: str,
        question: str,
        answer: str,
        session_id: str = "",
        quality: Optional[dict] = None,
        latency_ms: float = 0.0,
        cached: bool = False,
        blocked: bool = False,
    ) -> None:
        """Log a Q&A interaction."""
        self._write({
            "event": "question",
            "user_id": user_id,
            "session_id": session_id,
            "question": question[:200],
            "answer_preview": answer[:100],
            "quality_score": quality.get("overall") if quality else None,
            "latency_ms": round(latency_ms, 1),
            "cached": cached,
            "blocked": blocked,
        })

    def log_auth_event(
        self,
        event: str,
        username: str,
        ip: str = "",
        success: bool = True,
        role: str = "",
    ) -> None:
        """Log an authentication event (login, logout, failed attempt)."""
        self._write({
            "event": f"auth_{event}",
            "username": username,
            "ip": ip,
            "success": success,
            "role": role,
            "status": "ok" if success else "denied",
        })

    def log_admin_action(
        self,
        user_id: str,
        action: str,
        details: Optional[dict] = None,
    ) -> None:
        """Log an administrative action (ingest, user management, etc.)."""
        self._write({
            "event": "admin_action",
            "user_id": user_id,
            "action": action,
            **(details or {}),
        })

    def query_logs(
        self,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Query audit logs with optional filters.

        Args:
            event_type: Filter by event type (e.g. "question", "auth_login").
            user_id:    Filter by user ID.
            limit:      Maximum number of entries to return.
        """
        all_entries = self._read_all()
        filtered = []
        for entry in reversed(all_entries):
            if event_type and entry.get("event") != event_type:
                continue
            if user_id and entry.get("user_id") != user_id:
                continue
            filtered.append(entry)
            if len(filtered) >= limit:
                break
        return filtered

    def get_stats(self) -> dict:
        """
        Return aggregate statistics from the audit log.

        Keys: total_events, events_by_type, unique_users,
              total_questions, blocked_questions, cache_hits,
              auth_failures
        """
        all_entries = self._read_all()
        by_type: dict[str, int] = defaultdict(int)
        users: set[str] = set()
        total_q = blocked_q = cache_hits = auth_failures = 0

        for entry in all_entries:
            evt = entry.get("event", "unknown")
            by_type[evt] += 1
            uid = entry.get("user_id")
            if uid:
                users.add(uid)
            if evt == "question":
                total_q += 1
                if entry.get("blocked"):
                    blocked_q += 1
                if entry.get("cached"):
                    cache_hits += 1
            if evt.startswith("auth_") and not entry.get("success", True):
                auth_failures += 1

        return {
            "total_events": len(all_entries),
            "events_by_type": dict(by_type),
            "unique_users": len(users),
            "total_questions": total_q,
            "blocked_questions": blocked_q,
            "cache_hits": cache_hits,
            "auth_failures": auth_failures,
        }

    # ── Generic log (backward compat) ─────────────────────────────────────────

    def log(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        role: Optional[str] = None,
        ip_address: Optional[str] = None,
        status: str = "ok",
        **extra: Any,
    ) -> None:
        """Generic log method kept for backward compatibility."""
        record: dict = {
            "event": event_type,
            "status": status,
        }
        if user_id:
            record["user_id"] = user_id
        if role:
            record["role"] = role
        if ip_address:
            record["ip"] = ip_address
        record.update(extra)
        self._write(record)

    # ── Backward-compat read methods ──────────────────────────────────────────

    def read_recent(self, n: int = 100) -> list[dict]:
        return self._read_all()[-n:]

    def read_by_user(self, user_id: str, limit: int = 200) -> list[dict]:
        return self.query_logs(user_id=user_id, limit=limit)

    def read_by_event(self, event_type: str, limit: int = 200) -> list[dict]:
        return self.query_logs(event_type=event_type, limit=limit)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _write(self, record: dict) -> None:
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def _read_all(self) -> list[dict]:
        if not self._path.exists():
            return []
        with self._lock:
            with open(self._path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
        entries = []
        for line in lines:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return entries
