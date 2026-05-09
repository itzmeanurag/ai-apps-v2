"""
evaluation/feedback.py
Human feedback collection (thumbs up/down) and export as training data.
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional


FeedbackType = Literal["positive", "negative", "neutral"]


@dataclass
class FeedbackEntry:
    feedback_id: str
    session_id: str
    question: str
    answer: str
    context: str
    feedback: FeedbackType
    rating: Optional[int] = None          # 1-5 stars (optional)
    comment: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_training_example(self) -> dict:
        """Convert to a fine-tuning training example (instruction-response format)."""
        return {
            "instruction": self.question,
            "context": self.context,
            "response": self.answer,
            "quality": "good" if self.feedback == "positive" else "bad",
            "rating": self.rating,
        }


class FeedbackStore:
    """
    Thread-safe feedback store backed by a JSONL file.

    Usage:
        store = FeedbackStore("./data/feedback.jsonl")
        store.record("sess-1", "What is PTO?", "You get 15 days.", "...", "positive")
        store.export_training_data("./data/training.jsonl", only_positive=True)
    """

    def __init__(self, feedback_file: str = "./data/feedback.jsonl") -> None:
        self._path = Path(feedback_file)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._counter = self._count_existing()

    def _count_existing(self) -> int:
        if not self._path.exists():
            return 0
        with open(self._path, "r", encoding="utf-8") as fh:
            return sum(1 for _ in fh)

    def _generate_id(self) -> str:
        self._counter += 1
        return f"fb-{self._counter:06d}"

    def record(
        self,
        session_id: str,
        question: str,
        answer: str,
        context: str,
        feedback: FeedbackType,
        rating: Optional[int] = None,
        comment: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> FeedbackEntry:
        """Record a feedback entry and persist it."""
        with self._lock:
            entry = FeedbackEntry(
                feedback_id=self._generate_id(),
                session_id=session_id,
                question=question,
                answer=answer,
                context=context,
                feedback=feedback,
                rating=rating,
                comment=comment,
                user_id=user_id,
            )
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        return entry

    def thumbs_up(
        self,
        session_id: str,
        question: str,
        answer: str,
        context: str = "",
        user_id: Optional[str] = None,
    ) -> FeedbackEntry:
        return self.record(session_id, question, answer, context, "positive", user_id=user_id)

    def thumbs_down(
        self,
        session_id: str,
        question: str,
        answer: str,
        context: str = "",
        comment: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> FeedbackEntry:
        return self.record(
            session_id, question, answer, context, "negative",
            comment=comment, user_id=user_id
        )

    def load_all(self) -> list[FeedbackEntry]:
        """Load all feedback entries from disk."""
        entries: list[FeedbackEntry] = []
        if not self._path.exists():
            return entries
        with open(self._path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(FeedbackEntry(**json.loads(line)))
        return entries

    def get_stats(self) -> dict:
        """Return aggregate feedback statistics."""
        entries = self.load_all()
        total = len(entries)
        positive = sum(1 for e in entries if e.feedback == "positive")
        negative = sum(1 for e in entries if e.feedback == "negative")
        return {
            "total": total,
            "positive": positive,
            "negative": negative,
            "positive_rate": round(positive / total, 3) if total else 0.0,
        }

    def export_training_data(
        self,
        output_path: str,
        only_positive: bool = False,
        min_rating: Optional[int] = None,
    ) -> int:
        """
        Export feedback as training data (JSONL).

        Args:
            output_path: Destination file path.
            only_positive: If True, export only positive feedback.
            min_rating: If set, only export entries with rating >= min_rating.

        Returns:
            Number of examples exported.
        """
        entries = self.load_all()
        filtered = []
        for entry in entries:
            if only_positive and entry.feedback != "positive":
                continue
            if min_rating is not None and (entry.rating is None or entry.rating < min_rating):
                continue
            filtered.append(entry)

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            for entry in filtered:
                fh.write(json.dumps(entry.to_training_example(), ensure_ascii=False) + "\n")

        return len(filtered)
