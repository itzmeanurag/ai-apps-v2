"""
memory/memory_bank.py
3-layer persistent memory: buffer + summary + facts.
JSON file per session, LLM summarization, fact extraction.
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Message:
    role: str          # "user" | "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(**data)


@dataclass
class MemoryState:
    session_id: str
    buffer: list[Message] = field(default_factory=list)
    summary: str = ""
    facts: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "buffer": [m.to_dict() for m in self.buffer],
            "summary": self.summary,
            "facts": self.facts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryState":
        state = cls(session_id=data["session_id"])
        state.buffer = [Message.from_dict(m) for m in data.get("buffer", [])]
        state.summary = data.get("summary", "")
        state.facts = data.get("facts", [])
        state.created_at = data.get("created_at", state.created_at)
        state.updated_at = data.get("updated_at", state.updated_at)
        return state


# ── MemoryBank ────────────────────────────────────────────────────────────────

class MemoryBank:
    """
    3-layer persistent memory manager.

    Layers:
      1. Buffer  – recent N messages (fast, in-memory)
      2. Summary – LLM-generated rolling summary of older messages
      3. Facts   – extracted key facts (names, decisions, preferences)

    Each session is stored as a JSON file in session_dir.
    Thread-safe via per-session locks.
    """

    def __init__(
        self,
        session_dir: str,
        buffer_size: int = 10,
        summary_threshold: int = 8,
        facts_max: int = 50,
        llm: Optional[Any] = None,  # LangChain LLM or None
        enable_summarization: bool = True,
    ) -> None:
        self._session_dir = Path(session_dir)
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._buffer_size = buffer_size
        self._summary_threshold = summary_threshold
        self._facts_max = facts_max
        self._llm = llm
        self._enable_summarization = enable_summarization
        self._sessions: dict[str, MemoryState] = {}
        self._locks: dict[str, threading.Lock] = {}

    # ── Session management ────────────────────────────────────────────────────

    def _get_lock(self, session_id: str) -> threading.Lock:
        if session_id not in self._locks:
            self._locks[session_id] = threading.Lock()
        return self._locks[session_id]

    def _session_path(self, session_id: str) -> Path:
        return self._session_dir / f"{session_id}.json"

    def load_session(self, session_id: str) -> MemoryState:
        """Load or create a session."""
        if session_id in self._sessions:
            return self._sessions[session_id]

        path = self._session_path(session_id)
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            state = MemoryState.from_dict(data)
        else:
            state = MemoryState(session_id=session_id)

        self._sessions[session_id] = state
        return state

    def save_session(self, session_id: str) -> None:
        """Persist session to disk."""
        state = self._sessions.get(session_id)
        if state is None:
            return
        state.updated_at = datetime.now(timezone.utc).isoformat()
        path = self._session_path(session_id)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(state.to_dict(), fh, indent=2, ensure_ascii=False)

    def delete_session(self, session_id: str) -> None:
        """Remove session from memory and disk."""
        self._sessions.pop(session_id, None)
        self._locks.pop(session_id, None)
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()

    # ── Core operations ───────────────────────────────────────────────────────

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add a message to the buffer and trigger summarization if needed."""
        with self._get_lock(session_id):
            state = self.load_session(session_id)
            state.buffer.append(Message(role=role, content=content))

            # Trim buffer and summarize if over threshold
            if len(state.buffer) > self._buffer_size:
                self._maybe_summarize(state)

            self.save_session(session_id)

    def get_context(self, session_id: str) -> dict[str, Any]:
        """Return all memory layers as a dict for prompt injection."""
        state = self.load_session(session_id)
        return {
            "summary": state.summary,
            "facts": "\n".join(f"- {f}" for f in state.facts) if state.facts else "None",
            "history": self._format_buffer(state.buffer),
        }

    def get_buffer(self, session_id: str) -> list[Message]:
        return self.load_session(session_id).buffer

    def get_summary(self, session_id: str) -> str:
        return self.load_session(session_id).summary

    def get_facts(self, session_id: str) -> list[str]:
        return self.load_session(session_id).facts

    def add_fact(self, session_id: str, fact: str) -> None:
        """Manually add a fact to the facts layer."""
        with self._get_lock(session_id):
            state = self.load_session(session_id)
            if fact not in state.facts:
                state.facts.append(fact)
                if len(state.facts) > self._facts_max:
                    state.facts = state.facts[-self._facts_max:]
            self.save_session(session_id)

    def list_sessions(self) -> list[str]:
        """Return all session IDs found on disk."""
        return [p.stem for p in self._session_dir.glob("*.json")]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _format_buffer(self, buffer: list[Message]) -> str:
        lines = []
        for msg in buffer:
            prefix = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{prefix}: {msg.content}")
        return "\n".join(lines)

    def _maybe_summarize(self, state: MemoryState) -> None:
        """
        When buffer exceeds threshold, summarize the oldest messages
        and extract facts, then trim the buffer.
        """
        if not self._enable_summarization or self._llm is None:
            # Without LLM, just trim the buffer
            state.buffer = state.buffer[-self._buffer_size:]
            return

        # Take the oldest messages to summarize
        to_summarize = state.buffer[: self._summary_threshold]
        state.buffer = state.buffer[self._summary_threshold:]

        conversation_text = self._format_buffer(to_summarize)

        # Summarize
        try:
            from generation.prompts import PromptAssembler
            pa = PromptAssembler()
            summary_prompt = pa.build("summarize", conversation=conversation_text)
            new_summary = self._llm.invoke(summary_prompt)
            if hasattr(new_summary, "content"):
                new_summary = new_summary.content
            # Merge with existing summary
            if state.summary:
                merge_prompt = (
                    f"Merge these two summaries into one concise summary:\n\n"
                    f"Summary 1: {state.summary}\n\nSummary 2: {new_summary}\n\nMerged:"
                )
                merged = self._llm.invoke(merge_prompt)
                state.summary = merged.content if hasattr(merged, "content") else str(merged)
            else:
                state.summary = str(new_summary)
        except Exception as exc:
            print(f"[memory] Summarization failed: {exc}")

        # Extract facts
        try:
            from generation.prompts import PromptAssembler
            pa = PromptAssembler()
            facts_prompt = pa.build("extract_facts", conversation=conversation_text)
            facts_raw = self._llm.invoke(facts_prompt)
            facts_text = facts_raw.content if hasattr(facts_raw, "content") else str(facts_raw)
            import json as _json
            # Try to parse JSON list
            start = facts_text.find("[")
            end = facts_text.rfind("]") + 1
            if start >= 0 and end > start:
                new_facts: list[str] = _json.loads(facts_text[start:end])
                for fact in new_facts:
                    if fact not in state.facts:
                        state.facts.append(fact)
                if len(state.facts) > self._facts_max:
                    state.facts = state.facts[-self._facts_max:]
        except Exception as exc:
            print(f"[memory] Fact extraction failed: {exc}")
