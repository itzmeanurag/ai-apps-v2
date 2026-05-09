"""
LESSON 14: Fine-Tuning Concepts — QLoRA Explained
==================================================
CONCEPT: Teaching the model HOW to answer (style, tone, terminology)

WHAT THIS DOES:
  Explains QLoRA fine-tuning concepts and prepares sample training data.
  Actual training runs via scripts/finetune.py or notebooks/colab_finetune.ipynb.
  This lesson is the CONCEPT + DATA PREP lesson — not the training itself.

WHY THIS MATTERS:
  RAG provides WHAT to answer with (documents).
  Fine-tuning teaches HOW to answer (format, tone, domain terminology).
  Together they produce the best results.

RUN (from project root):
  python notebooks/lessons/14_fine_tuning.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import cfg


# ── Demo 1: Fine-tuning vs RAG ────────────────────────────────────────────────

def demo_finetune_vs_rag() -> None:
    """Explain when to use fine-tuning vs RAG."""
    print("\n--- DEMO 1: Fine-Tuning vs RAG ---")
    print("""
  FINE-TUNING teaches the model HOW to answer:
    - Writing style and tone ("Be formal and concise")
    - Domain terminology ("FMLA", "PTO", "401k")
    - Output format ("Always cite the policy section")
    - Company-specific knowledge baked into weights

  RAG provides WHAT to answer with:
    - Current document content (updated without retraining)
    - Source attribution (which document the answer came from)
    - Grounded answers (reduces hallucination)

  ┌─────────────────────────────────────────────────────────┐
  │ Situation                    │ RAG  │ Fine-Tune │ Both  │
  ├─────────────────────────────────────────────────────────┤
  │ Answer from specific docs    │  ✅  │           │       │
  │ Change writing style/tone    │      │    ✅     │       │
  │ Teach domain terminology     │      │    ✅     │       │
  │ Keep answers up-to-date      │  ✅  │           │       │
  │ Reduce hallucination         │      │           │  ✅   │
  │ Format outputs consistently  │      │    ✅     │       │
  └─────────────────────────────────────────────────────────┘

  BEST APPROACH: Fine-tune for HOW, RAG for WHAT.
""")


# ── Demo 2: How QLoRA works ───────────────────────────────────────────────────

def demo_qlora_explained() -> None:
    """Explain QLoRA step by step."""
    print("\n--- DEMO 2: How QLoRA Works ---")
    print("""
  STEP 1: Quantization (Q)
    Original Mistral 7B: 7B params × 16-bit = 14GB VRAM
    After 4-bit quantization: 7B params × 4-bit = 3.5GB VRAM
    The model gets slightly less precise but fits on consumer hardware.

  STEP 2: LoRA (Low-Rank Adaptation)
    Instead of updating all 7B parameters (impossible on your GPU),
    LoRA adds tiny "adapter" layers — only ~7M trainable parameters (0.1%).
    The original model stays FROZEN. Only adapters are trained.

    ┌─────────────────────────────────────────┐
    │  Original Mistral 7B (FROZEN)           │
    │  7 billion parameters                   │
    │  ┌─────────────────────────────────┐    │
    │  │  LoRA Adapters (TRAINABLE)      │    │
    │  │  ~7 million parameters (0.1%)   │    │
    │  │  These are the ONLY thing       │    │
    │  │  updated during training        │    │
    │  └─────────────────────────────────┘    │
    └─────────────────────────────────────────┘

  STEP 3: Save only the adapters
    Adapter file: ~50-200MB (not the full 4GB model)
    To use: load base model + apply adapters

  HARDWARE REQUIREMENTS:
    QLoRA on 7B model: 6-12GB VRAM (RTX 3060 or better)
    QLoRA on 13B model: 12-16GB VRAM (RTX 3090/4090)
    QLoRA on 70B model: 40GB+ VRAM (A100) — use Colab Pro+

  TRAINING TIME (RTX 3090, 1000 examples, 3 epochs):
    ~30-60 minutes
""")


# ── Demo 3: Training data format ──────────────────────────────────────────────

def demo_training_data_format() -> None:
    """Show the training data format and create sample data."""
    print("\n--- DEMO 3: Training Data Format ---")

    # The instruction-response format used by scripts/finetune.py
    sample_examples = [
        {
            "instruction": "How many days of annual leave do employees get?",
            "context": "Full-time employees receive 20 days of paid annual leave per calendar year.",
            "response": "Full-time employees receive 20 days of paid annual leave per year. "
                       "Part-time employees (20-39 hrs/week) receive leave at 50% of the full-time rate.",
            "quality": "good",
        },
        {
            "instruction": "Can I work from home?",
            "context": "Remote work is permitted up to 3 days per week with manager approval.",
            "response": "Yes, employees may work remotely up to 3 days per week. "
                       "This requires manager approval and must be documented in the HR system.",
            "quality": "good",
        },
        {
            "instruction": "What is the expense receipt threshold?",
            "context": "Business expenses over $50 require a receipt and manager sign-off.",
            "response": "Any business expense over $50 requires both a receipt and manager approval. "
                       "Expenses under $50 can be submitted without a receipt.",
            "quality": "good",
        },
    ]

    # Save as JSONL (one JSON object per line)
    output_path = Path("./data/training_sample.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for example in sample_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"\n  Created {output_path} with {len(sample_examples)} examples.")
    print(f"\n  Format (JSONL — one JSON per line):")
    print(f"  {json.dumps(sample_examples[0], indent=2)}")

    print("""
  QUALITY GUIDELINES:
    - Each example should be accurate and verifiable
    - Responses should match your desired style/format
    - Include context when the answer depends on a document
    - Minimum 100 examples for simple tasks, 1000+ for complex ones
    - Validate with: python scripts/training_validator.py
""")


# ── Demo 4: The fine-tuning workflow ──────────────────────────────────────────

def demo_finetune_workflow() -> None:
    """Show the complete fine-tuning workflow."""
    print("\n--- DEMO 4: Complete Fine-Tuning Workflow ---")
    print(f"""
  STEP 1: Collect training data
    Option A: Export from human feedback
      python -c "
      from evaluation.feedback import FeedbackStore
      store = FeedbackStore('./data/feedback.jsonl')
      n = store.export_training_data('./data/training.jsonl', only_positive=True)
      print(f'Exported {{n}} examples')
      "

    Option B: Use the sample data created in Demo 3
      ./data/training_sample.jsonl

  STEP 2: Validate training data
    python scripts/training_validator.py --data_path ./data/training_sample.jsonl

  STEP 3: Fine-tune (local GPU)
    pip install -r requirements-finetune.txt
    python scripts/finetune.py \\
      --model_name mistralai/Mistral-7B-v0.1 \\
      --data_path ./data/training_sample.jsonl \\
      --output_dir ./models/finetuned \\
      --epochs 3

  STEP 4: Fine-tune (Google Colab — free GPU)
    Upload notebooks/colab_finetune.ipynb to colab.research.google.com
    Runtime → Change runtime type → T4 GPU
    Run all cells → Download model.gguf

  STEP 5: Import into Ollama
    echo "FROM ./model.gguf" > Modelfile
    ollama create my-finetuned-model -f Modelfile

  STEP 6: Use in the chatbot
    Edit config.yaml:
      models:
        generator: "my-finetuned-model"  # was "mistral"
    Restart the chatbot — done!

  The fine-tuned model is a DROP-IN REPLACEMENT.
  LangChain, RAG, Memory Bank, Guardrails — all stay the same.
  Only the brain (the model) changes.
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("LESSON 14: FINE-TUNING CONCEPTS")
    print("=" * 60)
    print(f"""
Fine-tuning teaches HOW to answer. RAG provides WHAT to answer with.
Use both together for the best results.

Training script: scripts/finetune.py
Colab notebook:  notebooks/colab_finetune.ipynb
Validator:       scripts/training_validator.py
""")

    demo_finetune_vs_rag()
    demo_qlora_explained()
    demo_training_data_format()
    demo_finetune_workflow()


if __name__ == "__main__":
    main()
