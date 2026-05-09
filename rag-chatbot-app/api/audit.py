"""
api/audit.py
Thread-safe JSONL audit logger.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class AuditLogger:
    """
    Append-only, thread-safe audit logger.
    Each event is written as a single JSON line to a .jsonl file.

    Usage:
        logger = AuditLogger("./data/audit.jsonl")
        logger.log("ask", user_id="alice", question="What is PTO?", status="ok")
    """

    def __init__(self, log_file: str = "./data/audit.jsonl") -> None:
        self._path = Path(log_file)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        role: Optional[str] = None,
        ip_address: Optional[str] = None,
        status: str = "ok",
        **extra: Any,
    ) -> None:
        """
        Write an audit event.

        Args:
            event_type: e.g. "ask", "login", "logout", "ingest", "error"
            user_id:    Authenticated user identifier.
            role:       User role at time of event.
            ip_address: Client IP (anonymized if needed).
            status:     "ok" | "denied" | "error"
            **extra:    Additional key-value pairs to include.
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "user_id": user_id,
            "role": role,
            "ip": ip_address,
            "status": status,
            **extra,
        }
        # Remove None values to keep logs clean
        record = {k: v for k, v in record.items() if v is not None}

        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def read_recent(self, n: int = 100) -> list[dict]:
        """Return the last n audit entries."""
        if not self._path.exists():
            return []
        with self._lock:
            with open(self._path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
        entries = []
        for line in lines[-n:]:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return entries

    def read_by_user(self, user_id: str, limit: int = 200) -> list[dict]:
        """Return audit entries for a specific user."""
        all_entries = self.read_recent(limit * 5)
        return [e for e in all_entries if e.get("user_id") == user_id][-limit:]

    def read_by_event(self, event_type: str, limit: int = 200) -> list[dict]:
        """Return audit entries for a specific event type."""
        all_entries = self.read_recent(limit * 5)
        return [e for e in all_entries if e.get("event") == event_type][-limit:]
