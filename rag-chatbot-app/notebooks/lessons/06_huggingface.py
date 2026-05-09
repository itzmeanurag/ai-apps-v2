"""
LESSON 06: Hugging Face — The Open-Source AI Hub
=================================================
CONCEPT: Alternative models for embeddings and classification

WHAT THIS DOES:
  1. Uses sentence-transformers for embeddings (alternative to Ollama)
  2. Uses a pre-trained classifier for sentiment/content safety
  3. Shows zero-shot classification for topic routing

WHY THIS MATTERS:
  Hugging Face hosts 500,000+ free models. When you need something
  specialized (content safety, named entity recognition, translation),
  Hugging Face has a pre-trained model for it.
  Ollama is simpler for chat; Hugging Face gives more specialized options.

NOTE:
  First run downloads models (~500MB total, cached after that).

RUN (from project root):
  python notebooks/lessons/06_huggingface.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ── Cosine similarity (same as Lesson 02) ────────────────────────────────────

def cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


# ── Demo 1: Hugging Face Embeddings ──────────────────────────────────────────

def demo_hf_embeddings() -> None:
    """
    sentence-transformers runs DIRECTLY in Python — no Ollama needed.
    Useful when you want more embedding model options or offline use.
    """
    print("\n--- DEMO 1: Hugging Face Embeddings (sentence-transformers) ---")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  Install: pip install sentence-transformers")
        return

    # all-MiniLM-L6-v2: 384 dimensions, ~80MB, fast and accurate
    # Compare: nomic-embed-text = 768 dims, AWS TITAN = 1024 dims
    print("  Loading all-MiniLM-L6-v2 (first run downloads ~80MB)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    sentences = [
        "The employee handbook covers leave policies",
        "Workers can take vacation days as per company rules",
        "The API uses Bearer token authentication",
    ]

    embeddings = model.encode(sentences)
    print(f"  Dimensions: {embeddings.shape[1]} (vs 768 for nomic, 1024 for TITAN)")

    sim_01 = cosine_similarity(embeddings[0], embeddings[1])
    sim_02 = cosine_similarity(embeddings[0], embeddings[2])

    print(f"\n  Similarity (leave policy vs vacation rules): {sim_01:.4f}  ← Similar topic")
    print(f"  Similarity (leave policy vs API auth):       {sim_02:.4f}  ← Different topic")
    print("""
  WHEN TO USE HF EMBEDDINGS vs OLLAMA:
    Ollama nomic-embed-text: Simple setup, good quality, 768 dims
    HF sentence-transformers: More model choices, runs without Ollama
    AWS TITAN: Best quality, 1024 dims, requires AWS account
""")


# ── Demo 2: Text Classification ───────────────────────────────────────────────

def demo_text_classification() -> None:
    """
    Pre-trained classifiers for content safety.
    This is a simplified local version of AWS Bedrock Guardrails.
    """
    print("\n--- DEMO 2: Text Classification (Content Safety) ---")

    try:
        from transformers import pipeline
    except ImportError:
        print("  Install: pip install transformers torch")
        return

    print("  Loading sentiment classifier (first run downloads ~250MB)...")
    # distilbert is a smaller, faster version of BERT
    # Fine-tuned on SST-2 (Stanford Sentiment Treebank)
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )

    test_texts = [
        "The company policy is very clear and helpful",
        "I love working with this API documentation",
        "This is terrible and completely useless",
        "The refund process takes 30 business days",
    ]

    print("\n  Classifying text sentiment:")
    for text in test_texts:
        result = classifier(text)[0]
        label = result["label"]
        score = result["score"]
        icon = "✅" if label == "POSITIVE" else "⚠️"
        print(f"  {icon} [{label:8s} {score:.3f}] \"{text}\"")

    print("""
  NOTE: In production, use specialized safety models:
    - Meta Llama Guard 3: Detects harmful content categories
    - NVIDIA NeMo Guardrails: Structured safety rules
    - Our guardrails/content_safety.py: Regex + optional LLM classification
""")


# ── Demo 3: Zero-Shot Classification ─────────────────────────────────────────

def demo_zero_shot_classification() -> None:
    """
    Classify text into ANY categories without training on those categories.
    Useful for routing queries to the right knowledge base or persona.
    """
    print("\n--- DEMO 3: Zero-Shot Classification (Topic Routing) ---")

    try:
        from transformers import pipeline
    except ImportError:
        print("  Install: pip install transformers torch")
        return

    print("  Loading zero-shot classifier (first run downloads ~1.5GB)...")
    # facebook/bart-large-mnli: trained on natural language inference
    # Can classify into ANY categories without task-specific training
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # These categories can be ANYTHING — no training needed
    # In our project, this could route to different personas:
    #   "HR Policy" → hr persona, "Technical" → tech persona
    categories = ["HR Policy", "Technical Documentation", "Financial", "Legal"]

    test_texts = [
        "Employees get 20 days of annual leave per year",
        "The API rate limit is 100 requests per minute",
        "Revenue increased by 15% in Q3",
    ]

    print(f"\n  Categories: {categories}\n")
    for text in test_texts:
        result = classifier(text, categories)
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        print(f"  \"{text}\"")
        print(f"  → {top_label} (confidence: {top_score:.3f})\n")

    print("""
  HOW THIS CONNECTS TO OUR PROJECT:
    In generation/prompts.py, we have different personas:
      "hr"      → HR-focused answers
      "tech"    → Technical support answers
      "default" → General answers
    Zero-shot classification could automatically pick the right persona.
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("LESSON 06: HUGGING FACE ECOSYSTEM")
    print("=" * 60)
    print("""
Hugging Face is to AI models what GitHub is to code.
500,000+ free models. Three roles in our project:
  1. Alternative embedding models (sentence-transformers)
  2. Content safety classification (transformers pipeline)
  3. Zero-shot topic routing (bart-large-mnli)
""")

    demo_hf_embeddings()
    demo_text_classification()
    demo_zero_shot_classification()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
  Ollama  → Chat models + embeddings (simple, local)
  HF      → Specialized tasks (classification, NER, translation)
  Both    → Work together in our project

  WHEN TO USE WHAT:
    Ollama: Chat, embeddings (simple setup, no Python model loading)
    HF sentence-transformers: Embeddings (more model choices)
    HF transformers pipeline: Classification, NER, summarization
    HF zero-shot: Topic routing without training data
""")


if __name__ == "__main__":
    main()
