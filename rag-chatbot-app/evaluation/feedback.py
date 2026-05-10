"""
evaluation/feedback.py

FeedbackCollector – collect human feedback and export as training data.

Spec-required methods:
  record(question, answer, rating, comment, relevance_score)
  get_analytics() -> dict
  export_training_data(output_path, min_rating) -> int
  save_training_data(output_path, min_rating) -> int  (alias)

Also keeps FeedbackStore as a backward-compat alias.
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional


FeedbackType = Literal["positive", "negative", "neutral"]


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class FeedbackRecord:
    feedback_id: str
    question: str
    answer: str
    rating: int                    # 1–5
    comment: str = ""
    relevance_score: float = 0.0   # LLM-judged relevance (0–1)
    session_id: str = ""
    user_id: str = ""
    feedback: FeedbackType = "neutral"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_training_example(self) -> dict:
        return {
            "instruction": self.question,
            "response": self.answer,
            "rating": self.rating,
            "quality": "good" if self.rating >= 4 else "bad",
        }


# ── FeedbackCollector ─────────────────────────────────────────────────────────

class FeedbackCollector:
    """
    Thread-safe feedback collector backed by a JSONL file.

    Usage:
        collector = FeedbackCollector()
        collector.record("How many leave days?", "20 days.", rating=5)
        collector.save_training_data("./data/training.jsonl", min_rating=4)
    """

    def __init__(
        self,
        feedback_file: str = "./data/feedback.jsonl",
    ) -> None:
        self._path = Path(feedback_file)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._counter = self._count_existing()

    def _count_existing(self) -> int:
        if not self._path.exists():
            return 0
        with open(self._path, "r", encoding="utf-8") as fh:
            return sum(1 for _ in fh)

    def _next_id(self) -> str:
        self._counter += 1
        return f"fb-{self._counter:06d}"

    # ── Spec-required methods ─────────────────────────────────────────────────

    def record(
        self,
        question: str,
        answer: str,
        rating: int,
        comment: str = "",
        relevance_score: float = 0.0,
        session_id: str = "",
        user_id: str = "",
        context: str = "",          # accepted but not stored separately
        feedback: FeedbackType = "neutral",
    ) -> FeedbackRecord:
        """Record a feedback entry and persist it."""
        if rating >= 4:
            feedback = "positive"
        elif rating <= 2:
            feedback = "negative"
        else:
            feedback = "neutral"

        with self._lock:
            entry = FeedbackRecord(
                feedback_id=self._next_id(),
                question=question,
                answer=answer,
                rating=rating,
                comment=comment,
                relevance_score=relevance_score,
                session_id=session_id,
                user_id=user_id,
                feedback=feedback,
            )
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        return entry

    def get_analytics(self) -> dict:
        """
        Return aggregate analytics.

        Keys: total_feedback, satisfaction_rate, average_rating,
              feedback_with_comments, llm_vs_human_false_positives,
              rating_distribution
        """
        entries = self._load_all()
        total = len(entries)
        if total == 0:
            return {
                "total_feedback": 0,
                "satisfaction_rate": 0.0,
                "average_rating": 0.0,
                "feedback_with_comments": 0,
                "llm_vs_human_false_positives": 0,
                "rating_distribution": {},
            }

        positive = sum(1 for e in entries if e.feedback == "positive")
        with_comments = sum(1 for e in entries if e.comment)
        avg_rating = sum(e.rating for e in entries) / total

        # LLM false positives: LLM said good (relevance >= 0.7) but human said bad (rating <= 2)
        false_positives = sum(
            1 for e in entries
            if e.relevance_score >= 0.7 and e.rating <= 2
        )

        dist: dict[int, int] = {}
        for e in entries:
            dist[e.rating] = dist.get(e.rating, 0) + 1

        return {
            "total_feedback": total,
            "satisfaction_rate": round(positive / total, 3),
            "average_rating": round(avg_rating, 2),
            "feedback_with_comments": with_comments,
            "llm_vs_human_false_positives": false_positives,
            "rating_distribution": dist,
        }

    def export_training_data(
        self,
        output_path: str,
        min_rating: int = 4,
    ) -> int:
        """
        Export high-quality feedback as JSONL training data.
        Returns the number of examples exported.
        """
        entries = self._load_all()
        filtered = [e for e in entries if e.rating >= min_rating]

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            for e in filtered:
                fh.write(json.dumps(e.to_training_example(), ensure_ascii=False) + "\n")
        return len(filtered)

    def save_training_data(
        self,
        output_path: str,
        min_rating: int = 4,
    ) -> int:
        """Alias for export_training_data (spec-required name)."""
        return self.export_training_data(output_path, min_rating=min_rating)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_all(self) -> list[FeedbackRecord]:
        if not self._path.exists():
            return []
        entries = []
        with open(self._path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        entries.append(FeedbackRecord(**json.loads(line)))
                    except (TypeError, KeyError):
                        pass
        return entries

    def get_stats(self) -> dict:
        """Alias for get_analytics (backward compat)."""
        return self.get_analytics()

    # ── Thumbs up/down helpers ────────────────────────────────────────────────

    def thumbs_up(
        self,
        session_id: str,
        question: str,
        answer: str,
        context: str = "",
        user_id: str = "",
    ) -> FeedbackRecord:
        return self.record(
            question=question, answer=answer, rating=5,
            session_id=session_id, user_id=user_id, feedback="positive",
        )

    def thumbs_down(
        self,
        session_id: str,
        question: str,
        answer: str,
        context: str = "",
        comment: str = "",
        user_id: str = "",
    ) -> FeedbackRecord:
        return self.record(
            question=question, answer=answer, rating=1,
            comment=comment, session_id=session_id, user_id=user_id,
            feedback="negative",
        )

    def load_all(self) -> list[FeedbackRecord]:
        return self._load_all()


# ── FeedbackStore – backward-compat alias ─────────────────────────────────────

class FeedbackStore(FeedbackCollector):
    """
    Backward-compatible alias for FeedbackCollector.
    Adds the record() signature used by the old API server.
    """

    def record(  # type: ignore[override]
        self,
        session_id: str = "",
        question: str = "",
        answer: str = "",
        context: str = "",
        feedback: FeedbackType = "neutral",
        rating: Optional[int] = None,
        comment: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> FeedbackRecord:
        r = rating if rating is not None else (5 if feedback == "positive" else 1)
        return super().record(
            question=question,
            answer=answer,
            rating=r,
            comment=comment or "",
            session_id=session_id,
            user_id=user_id or "",
            feedback=feedback,
        )

    @property
    def feedback_id(self) -> str:
        return f"fb-{self._counter:06d}"
