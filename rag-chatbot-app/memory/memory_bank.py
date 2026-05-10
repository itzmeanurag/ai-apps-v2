"""
memory/memory_bank.py

MemoryBank – 3-layer persistent conversation memory.

Layers:
  1. Buffer  – last N exchanges (full detail)
  2. Summary – LLM-compressed older exchanges
  3. Facts   – extracted key facts

Sessions stored as JSON files in memory_bank/ directory.

Spec-required methods:
  add_exchange(session_id, user_msg, assistant_msg)
  get_context(session_id) -> dict
  _compress_buffer(session_id)
  _extract_facts(session_id, conversation_text)
  list_sessions() -> list[str]
  delete_session(session_id)
  cleanup_old_sessions(max_age_days)
  get_session_stats(session_id) -> dict
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Exchange:
    user: str
    assistant: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Exchange":
        return cls(**d)


@dataclass
class SessionState:
    session_id: str
    buffer: list[Exchange] = field(default_factory=list)
    summary: str = ""
    key_facts: list[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "buffer": [e.to_dict() for e in self.buffer],
            "summary": self.summary,
            "key_facts": self.key_facts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionState":
        s = cls(session_id=d["session_id"])
        s.buffer = [Exchange.from_dict(e) for e in d.get("buffer", [])]
        s.summary = d.get("summary", "")
        s.key_facts = d.get("key_facts", [])
        s.created_at = d.get("created_at", s.created_at)
        s.updated_at = d.get("updated_at", s.updated_at)
        return s


# ── MemoryBank ────────────────────────────────────────────────────────────────

class MemoryBank:
    """
    3-layer persistent conversation memory.

    Default session directory: memory_bank/  (relative to cwd)
    """

    def __init__(
        self,
        session_dir: str = "memory_bank",
        buffer_size: int = 6,
        summary_threshold: int = 4,
        max_key_facts: int = 20,
        session_age_days: int = 30,
        llm: Optional[Any] = None,
    ) -> None:
        self._dir = Path(session_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._buffer_size = buffer_size
        self._summary_threshold = summary_threshold
        self._max_key_facts = max_key_facts
        self._session_age_days = session_age_days
        self._llm = llm
        self._sessions: dict[str, SessionState] = {}
        self._locks: dict[str, threading.Lock] = {}

    # ── Session I/O ───────────────────────────────────────────────────────────

    def _lock(self, session_id: str) -> threading.Lock:
        if session_id not in self._locks:
            self._locks[session_id] = threading.Lock()
        return self._locks[session_id]

    def _path(self, session_id: str) -> Path:
        return self._dir / f"{session_id}.json"

    def _load(self, session_id: str) -> SessionState:
        if session_id in self._sessions:
            return self._sessions[session_id]
        p = self._path(session_id)
        if p.exists():
            with open(p, "r", encoding="utf-8") as fh:
                state = SessionState.from_dict(json.load(fh))
        else:
            state = SessionState(session_id=session_id)
        self._sessions[session_id] = state
        return state

    def _save(self, session_id: str) -> None:
        state = self._sessions.get(session_id)
        if state is None:
            return
        state.updated_at = datetime.now(timezone.utc).isoformat()
        with open(self._path(session_id), "w", encoding="utf-8") as fh:
            json.dump(state.to_dict(), fh, indent=2, ensure_ascii=False)

    # ── Spec-required public methods ──────────────────────────────────────────

    def add_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """
        Add a user/assistant exchange to the buffer.
        Triggers compression when buffer exceeds threshold.
        """
        with self._lock(session_id):
            state = self._load(session_id)
            state.buffer.append(Exchange(user=user_message, assistant=assistant_message))
            if len(state.buffer) > self._buffer_size:
                self._compress_buffer(session_id)
            self._save(session_id)

    def get_context(self, session_id: str) -> dict:
        """
        Return all memory layers as a dict for prompt injection.

        Keys: history (str), summary (str), facts (str),
              total_exchanges (int), has_summary (bool)
        """
        state = self._load(session_id)
        history_lines = []
        for ex in state.buffer:
            history_lines.append(f"User: {ex.user}")
            history_lines.append(f"Assistant: {ex.assistant}")

        return {
            "history": "\n".join(history_lines),
            "summary": state.summary,
            "facts": "\n".join(f"- {f}" for f in state.key_facts) if state.key_facts else "",
            "total_exchanges": len(state.buffer),
            "has_summary": bool(state.summary),
        }

    def _compress_buffer(self, session_id: str) -> None:
        """
        Compress the oldest exchanges into the summary layer.
        Called automatically when buffer overflows.
        """
        state = self._sessions.get(session_id)
        if state is None:
            return

        to_compress = state.buffer[: self._summary_threshold]
        state.buffer = state.buffer[self._summary_threshold:]

        conversation_text = "\n".join(
            f"User: {ex.user}\nAssistant: {ex.assistant}" for ex in to_compress
        )

        if self._llm is not None:
            try:
                from generation.prompts import prompts as _p
                if state.summary:
                    prompt = _p.get("memory_summarize_update").format(
                        existing_summary=state.summary,
                        new_conversation=conversation_text,
                    )
                else:
                    prompt = _p.get("memory_summarize_new").format(
                        conversation=conversation_text
                    )
                resp = self._llm.invoke(prompt)
                state.summary = resp.content if hasattr(resp, "content") else str(resp)
            except Exception as exc:
                print(f"[memory] Summarization failed: {exc}")
                # Fallback: append raw text
                state.summary = (state.summary + "\n" + conversation_text).strip()[-1000:]

            # Extract facts
            self._extract_facts(session_id, conversation_text)
        else:
            # No LLM: keep a simple rolling text summary
            state.summary = (state.summary + "\n" + conversation_text).strip()[-1000:]

    def _extract_facts(self, session_id: str, conversation_text: str) -> None:
        """
        Extract key facts from conversation_text and add to the facts layer.
        Requires LLM to be set.
        """
        state = self._sessions.get(session_id)
        if state is None or self._llm is None:
            return
        try:
            import json as _json
            from generation.prompts import prompts as _p
            prompt = _p.get("memory_extract_facts").format(conversation=conversation_text)
            resp = self._llm.invoke(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            start, end = text.find("["), text.rfind("]") + 1
            if start >= 0 and end > start:
                new_facts: list[str] = _json.loads(text[start:end])
                for fact in new_facts:
                    if fact not in state.key_facts:
                        state.key_facts.append(fact)
                if len(state.key_facts) > self._max_key_facts:
                    state.key_facts = state.key_facts[-self._max_key_facts:]
        except Exception as exc:
            print(f"[memory] Fact extraction failed: {exc}")

    def list_sessions(self) -> list[str]:
        """Return all session IDs found on disk."""
        return [p.stem for p in self._dir.glob("*.json")]

    def delete_session(self, session_id: str) -> None:
        """Remove session from memory and disk."""
        self._sessions.pop(session_id, None)
        self._locks.pop(session_id, None)
        p = self._path(session_id)
        if p.exists():
            p.unlink()

    def cleanup_old_sessions(self, max_age_days: Optional[int] = None) -> int:
        """
        Delete sessions older than max_age_days.
        Returns the number of sessions deleted.
        """
        max_age = max_age_days or self._session_age_days
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age)
        deleted = 0
        for session_id in self.list_sessions():
            state = self._load(session_id)
            try:
                updated = datetime.fromisoformat(state.updated_at)
                if updated < cutoff:
                    self.delete_session(session_id)
                    deleted += 1
            except (ValueError, TypeError):
                pass
        return deleted

    def get_session_stats(self, session_id: str) -> dict:
        """
        Return statistics for a session.

        Keys: session_id, total_exchanges, buffer_size, has_summary,
              key_facts_count, key_facts, created_at, updated_at
        """
        state = self._load(session_id)
        return {
            "session_id": session_id,
            "total_exchanges": len(state.buffer),
            "buffer_size": len(state.buffer),
            "has_summary": bool(state.summary),
            "key_facts_count": len(state.key_facts),
            "key_facts": state.key_facts,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
        }

    # ── Backward-compat aliases (used by src/chatbot.py old API) ─────────────

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Backward-compat: buffer individual messages, flush on assistant turn."""
        if not hasattr(self, "_pending"):
            self._pending: dict[str, str] = {}
        if role == "user":
            self._pending[session_id] = content
        elif role == "assistant":
            user_msg = self._pending.pop(session_id, "")
            self.add_exchange(session_id, user_msg, content)

    def get_buffer(self, session_id: str) -> list:
        return self._load(session_id).buffer

    def get_summary(self, session_id: str) -> str:
        return self._load(session_id).summary

    def get_facts(self, session_id: str) -> list[str]:
        return self._load(session_id).key_facts

    def save_session(self, session_id: str) -> None:
        self._save(session_id)

    def load_session(self, session_id: str) -> SessionState:
        return self._load(session_id)
