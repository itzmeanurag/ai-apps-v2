"""
LESSON 15: Human Feedback — Closing the Improvement Loop
=========================================================
CONCEPT: Collecting ratings to generate fine-tuning training data

WHAT THIS DOES:
  Demonstrates the FeedbackStore from evaluation/feedback.py:
    - Record thumbs up/down ratings with optional comments
    - Analyze feedback statistics
    - Export positively-rated Q&A pairs as fine-tuning training data

WHY THIS MATTERS:
  Automatic evaluation (LLM-as-judge) catches obvious quality issues.
  Human feedback catches real-world failures the LLM misses.
  Positive ratings become training data → fine-tuning → better answers.
  This closes the feedback → training → deployment loop.

RUN (from project root):
  python notebooks/lessons/15_human_feedback.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from the actual project module (evaluation/feedback.py)
from evaluation.feedback import FeedbackStore, FeedbackEntry
from src.config import cfg


# ── Demo 1: Recording feedback ────────────────────────────────────────────────

def demo_record_feedback() -> None:
    """Show how to record thumbs up/down feedback."""
    print("\n--- DEMO 1: Recording Feedback ---")

    store = FeedbackStore("./data/feedback_demo.jsonl")

    # Simulate a session with mixed feedback
    simulated = [
        ("How many leave days?",
         "Full-time employees get 20 days of annual leave per year.",
         "positive", 5, "Perfect answer"),
        ("Remote work policy?",
         "3 days per week with manager approval.",
         "positive", 4, ""),
        ("What is quantum physics?",
         "I don't have relevant documents about quantum physics.",
         "negative", 2, "Should say it doesn't know"),
        ("Expense receipt limit?",
         "Expenses over $50 require a receipt.",
         "positive", 5, "Accurate and concise"),
        ("Who is the CEO?",
         "The CEO is John Smith.",
         "negative", 1, "Wrong — made up a name (hallucination)"),
    ]

    print("\n  Recording feedback entries:")
    for question, answer, feedback, rating, comment in simulated:
        entry = store.record(
            session_id="lesson15-demo",
            question=question,
            answer=answer,
            context="",
            feedback=feedback,
            rating=rating,
            comment=comment if comment else None,
        )
        icon = "👍" if feedback == "positive" else "👎"
        print(f"  {icon} [{rating}/5] \"{question[:40]}\"")
        if comment:
            print(f"       Comment: {comment}")

    return store


# ── Demo 2: Feedback analytics ────────────────────────────────────────────────

def demo_analytics(store: FeedbackStore) -> None:
    """Show feedback statistics."""
    print("\n--- DEMO 2: Feedback Analytics ---")

    stats = store.get_stats()
    print(f"\n  Total feedback:    {stats['total']}")
    print(f"  Positive:          {stats['positive']}")
    print(f"  Negative:          {stats['negative']}")
    print(f"  Positive rate:     {stats['positive_rate']:.0%}")

    # Load all entries to show details
    entries = store.load_all()
    print(f"\n  All entries:")
    for entry in entries:
        icon = "👍" if entry.feedback == "positive" else "👎"
        print(f"  {icon} [{entry.rating}/5] \"{entry.question[:40]}\"")
        if entry.comment:
            print(f"       → {entry.comment}")


# ── Demo 3: Export as training data ──────────────────────────────────────────

def demo_export_training_data(store: FeedbackStore) -> None:
    """Show how positive feedback becomes fine-tuning training data."""
    print("\n--- DEMO 3: Export as Training Data ---")
    print("  Positive ratings (👍) become fine-tuning training examples.\n")

    output_path = "./data/feedback_training_demo.jsonl"
    count = store.export_training_data(
        output_path=output_path,
        only_positive=True,
        min_rating=4,
    )

    print(f"  Exported {count} high-quality examples to {output_path}")

    # Show what the training data looks like
    if count > 0:
        import json
        print(f"\n  Sample training example:")
        with open(output_path, "r") as f:
            first = json.loads(f.readline())
        print(f"  {json.dumps(first, indent=4)}")

    print(f"""
  THE FEEDBACK LOOP:
    1. User asks question → bot answers
    2. User rates: 👍 or 👎
    3. 👍 ratings exported as training data
    4. Fine-tune model on training data
    5. Deploy improved model
    6. Collect more feedback → repeat

  This is how ChatGPT, Claude, and every major AI improves over time.
  It's called RLHF (Reinforcement Learning from Human Feedback).
""")

    # Clean up demo files
    Path("./data/feedback_demo.jsonl").unlink(missing_ok=True)
    Path("./data/feedback_training_demo.jsonl").unlink(missing_ok=True)


# ── Demo 4: Feedback in the full system ──────────────────────────────────────

def demo_feedback_in_system() -> None:
    """Show how feedback integrates with the RAGChatbot."""
    print("\n--- DEMO 4: Feedback in the Full System ---")
    print("""
  The RAGChatbot exposes a feedback_store attribute:

    from src.chatbot import RAGChatbot
    bot = RAGChatbot()

    # Ask a question
    result = bot.ask("How many leave days?", session_id="sess-1")
    print(result["answer"])

    # User gives thumbs up
    bot.feedback_store.thumbs_up(
        session_id="sess-1",
        question="How many leave days?",
        answer=result["answer"],
        context="",
    )

    # User gives thumbs down with comment
    bot.feedback_store.thumbs_down(
        session_id="sess-1",
        question="Who is the CEO?",
        answer="The CEO is John Smith.",
        comment="Wrong — made up a name",
    )

  Via the API (api/server.py):
    POST /feedback
    {
      "session_id": "sess-1",
      "question": "How many leave days?",
      "answer": "20 days per year",
      "feedback": "positive"
    }

  Via the Gradio UI (app.py):
    Click 👍 or 👎 after each answer.
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("LESSON 15: HUMAN FEEDBACK")
    print("=" * 60)
    print(f"""
FeedbackStore (evaluation/feedback.py):
  store = FeedbackStore("./data/feedback.jsonl")
  store.thumbs_up(session_id, question, answer)
  store.thumbs_down(session_id, question, answer, comment="...")
  n = store.export_training_data("./data/training.jsonl", only_positive=True)

Feedback file: {cfg.logging.audit_log_file.replace("audit.jsonl", "feedback.jsonl")}
""")

    store = demo_record_feedback()
    demo_analytics(store)
    demo_export_training_data(store)
    demo_feedback_in_system()


if __name__ == "__main__":
    main()
