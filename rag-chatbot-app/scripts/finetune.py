"""
scripts/finetune.py
QLoRA fine-tuning skeleton for the RAG chatbot generator model.
Uses PEFT + bitsandbytes for 4-bit quantized LoRA training.

Usage:
    python scripts/finetune.py \
        --model_name mistralai/Mistral-7B-v0.1 \
        --data_path ./data/training.jsonl \
        --output_dir ./models/finetuned \
        --epochs 3
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for RAG chatbot")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--data_path", type=str, default="./data/training.jsonl")
    parser.add_argument("--output_dir", type=str, default="./models/finetuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--bf16", action="store_true", default=False)
    return parser.parse_args()


def load_training_data(data_path: str) -> list[dict]:
    """Load JSONL training data."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")
    examples = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    print(f"[finetune] Loaded {len(examples)} training examples.")
    return examples


def format_example(example: dict) -> str:
    """
    Format a training example as an instruction-following prompt.
    Expected keys: instruction, context, response
    """
    instruction = example.get("instruction", "")
    context = example.get("context", "")
    response = example.get("response", "")

    if context:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Context:\n{context}\n\n"
            f"### Response:\n{response}"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"


def get_bnb_config():
    """4-bit quantization config for QLoRA."""
    import torch
    from transformers import BitsAndBytesConfig  # type: ignore

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def get_lora_config(args: argparse.Namespace):
    """LoRA adapter configuration."""
    from peft import LoraConfig, TaskType  # type: ignore

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


def train(args: argparse.Namespace) -> None:
    """Main training loop."""
    import torch
    from datasets import Dataset  # type: ignore
    from peft import get_peft_model, prepare_model_for_kbit_training  # type: ignore
    from transformers import (  # type: ignore
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    print(f"[finetune] Loading model: {args.model_name}")
    print(f"[finetune] CUDA available: {torch.cuda.is_available()}")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Load model with 4-bit quantization ────────────────────────────────────
    bnb_config = get_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    lora_config = get_lora_config(args)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Prepare dataset ───────────────────────────────────────────────────────
    raw_data = load_training_data(args.data_path)
    texts = [format_example(ex) for ex in raw_data]

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_len,
            padding="max_length",
        )

    dataset = Dataset.from_dict({"text": texts})
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Train/eval split (90/10)
    split = tokenized.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"[finetune] Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # ── Training arguments ────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16 and not args.bf16,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy="steps",
        eval_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="none",  # set to "wandb" or "tensorboard" if desired
        dataloader_num_workers=0,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
    )

    print("[finetune] Starting training...")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    model.save_pretrained(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"[finetune] Model saved to {output_dir / 'final'}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
