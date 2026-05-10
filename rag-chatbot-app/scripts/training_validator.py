"""
scripts/training_validator.py
Validates training data quality before fine-tuning.
Checks: format, length, duplicates, quality distribution, PII leakage.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate training data for fine-tuning")
    parser.add_argument("--data_path", type=str, default="./data/training.jsonl")
    parser.add_argument("--min_examples", type=int, default=10)
    parser.add_argument("--max_instruction_len", type=int, default=500)
    parser.add_argument("--max_response_len", type=int, default=2000)
    parser.add_argument("--min_response_len", type=int, default=10)
    parser.add_argument("--check_pii", action="store_true", default=True)
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser.parse_args()


def load_data(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"[validator] ERROR: File not found: {path}")
        sys.exit(1)
    examples = []
    with open(p, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[validator] WARNING: Line {i} is not valid JSON: {exc}")
    return examples


def check_required_fields(examples: list[dict]) -> list[str]:
    """Check that all examples have required fields."""
    errors = []
    required = {"instruction", "response"}
    for i, ex in enumerate(examples):
        missing = required - set(ex.keys())
        if missing:
            errors.append(f"Example {i}: missing fields {missing}")
    return errors


def check_lengths(
    examples: list[dict],
    max_instruction: int,
    max_response: int,
    min_response: int,
) -> dict:
    """Check text length constraints."""
    issues = {"too_long_instruction": [], "too_long_response": [], "too_short_response": []}
    for i, ex in enumerate(examples):
        instr_len = len(ex.get("instruction", ""))
        resp_len = len(ex.get("response", ""))
        if instr_len > max_instruction:
            issues["too_long_instruction"].append(i)
        if resp_len > max_response:
            issues["too_long_response"].append(i)
        if resp_len < min_response:
            issues["too_short_response"].append(i)
    return issues


def check_duplicates(examples: list[dict]) -> list[tuple[int, int]]:
    """Find duplicate instruction-response pairs."""
    seen: dict[str, int] = {}
    duplicates: list[tuple[int, int]] = []
    for i, ex in enumerate(examples):
        key = ex.get("instruction", "") + "|||" + ex.get("response", "")
        if key in seen:
            duplicates.append((seen[key], i))
        else:
            seen[key] = i
    return duplicates


def check_quality_distribution(examples: list[dict]) -> dict:
    """Analyze quality label distribution."""
    qualities = [ex.get("quality", "unknown") for ex in examples]
    return dict(Counter(qualities))


def check_pii(examples: list[dict]) -> list[int]:
    """Check for PII in training data."""
    import re
    pii_patterns = [
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),  # email
        re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),  # SSN
        re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14})\b"),  # credit card
    ]
    flagged = []
    for i, ex in enumerate(examples):
        text = ex.get("instruction", "") + " " + ex.get("response", "")
        if any(p.search(text) for p in pii_patterns):
            flagged.append(i)
    return flagged


def compute_stats(examples: list[dict]) -> dict:
    """Compute basic statistics about the dataset."""
    instr_lens = [len(ex.get("instruction", "")) for ex in examples]
    resp_lens = [len(ex.get("response", "")) for ex in examples]
    ctx_lens = [len(ex.get("context", "")) for ex in examples]

    def stats(values: list[int]) -> dict:
        if not values:
            return {}
        return {
            "min": min(values),
            "max": max(values),
            "avg": round(sum(values) / len(values), 1),
        }

    return {
        "total_examples": len(examples),
        "instruction_length": stats(instr_lens),
        "response_length": stats(resp_lens),
        "context_length": stats(ctx_lens),
        "has_context": sum(1 for ex in examples if ex.get("context")),
    }


def validate(args: argparse.Namespace) -> bool:
    """Run all validation checks. Returns True if data is valid."""
    print(f"\n[validator] Validating: {args.data_path}")
    print("=" * 60)

    examples = load_data(args.data_path)
    print(f"[validator] Loaded {len(examples)} examples.")

    passed = True

    # Minimum count
    if len(examples) < args.min_examples:
        print(f"[validator] FAIL: Only {len(examples)} examples (minimum: {args.min_examples})")
        passed = False
    else:
        print(f"[validator] PASS: Example count ({len(examples)} >= {args.min_examples})")

    # Required fields
    field_errors = check_required_fields(examples)
    if field_errors:
        print(f"[validator] FAIL: {len(field_errors)} examples missing required fields")
        if args.verbose:
            for err in field_errors[:10]:
                print(f"  {err}")
        passed = False
    else:
        print("[validator] PASS: All examples have required fields")

    # Length checks
    length_issues = check_lengths(
        examples,
        args.max_instruction_len,
        args.max_response_len,
        args.min_response_len,
    )
    for issue_type, indices in length_issues.items():
        if indices:
            print(f"[validator] WARN: {len(indices)} examples with '{issue_type}'")
            if args.verbose:
                print(f"  Indices: {indices[:10]}")
        else:
            print(f"[validator] PASS: {issue_type}")

    # Duplicates
    duplicates = check_duplicates(examples)
    if duplicates:
        print(f"[validator] WARN: {len(duplicates)} duplicate pairs found")
        if args.verbose:
            for a, b in duplicates[:5]:
                print(f"  Examples {a} and {b} are identical")
    else:
        print("[validator] PASS: No duplicates found")

    # Quality distribution
    quality_dist = check_quality_distribution(examples)
    print(f"[validator] Quality distribution: {quality_dist}")
    if quality_dist.get("bad", 0) > quality_dist.get("good", 0):
        print("[validator] WARN: More 'bad' examples than 'good' – consider filtering")

    # PII check
    if args.check_pii:
        pii_flagged = check_pii(examples)
        if pii_flagged:
            print(f"[validator] WARN: {len(pii_flagged)} examples may contain PII")
            if args.verbose:
                print(f"  Indices: {pii_flagged[:10]}")
        else:
            print("[validator] PASS: No PII detected")

    # Statistics
    stats = compute_stats(examples)
    print("\n[validator] Dataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("=" * 60)
    if passed:
        print("[validator] ✅ Validation PASSED – data is ready for fine-tuning")
    else:
        print("[validator] ❌ Validation FAILED – fix errors before fine-tuning")

    return passed


if __name__ == "__main__":
    args = parse_args()
    ok = validate(args)
    sys.exit(0 if ok else 1)
