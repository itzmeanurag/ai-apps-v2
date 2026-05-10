"""
LESSON 10: PII Detection — Protecting Personal Information
===========================================================
CONCEPT: Detect, anonymize, or block Personally Identifiable Information

WHAT THIS DOES:
  Demonstrates PII handling from guardrails/content_safety.py:
    - Detect: email, phone, SSN, credit card
    - Anonymize: replace with [EMAIL], [PHONE], [SSN], [CREDIT_CARD]
    - Block: reject queries containing sensitive PII types

WHY THIS MATTERS:
  Users often include PII in their questions without thinking.
  If PII reaches the LLM, it might appear in logs, audit trails,
  or even in the model's responses to other users.
  PII detection ensures the model never sees real personal data.

RUN (from project root):
  python notebooks/lessons/10_pii_detection.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from the actual project module (guardrails/content_safety.py)
from guardrails.content_safety import ContentSafetyFilter


# ── Demo 1: PII detection and anonymization ───────────────────────────────────

def demo_pii_detection() -> None:
    """Show PII detection across all supported types."""
    print("\n--- DEMO 1: PII Detection and Anonymization ---")
    print("  PII is replaced with [TYPE] placeholders before reaching the LLM.\n")

    csf = ContentSafetyFilter()

    test_cases = [
        "My email is john.doe@company.com and I need help with leave",
        "Call me at 555-123-4567 about my benefits",
        "My SSN is 123-45-6789, please verify my account",
        "Charge my card 4111-1111-1111-1111 for the subscription",
        "No personal info here — just asking about the leave policy",
        "Contact support@hr.com or call 800-555-0199 for assistance",
    ]

    for text in test_cases:
        result = csf.check_input(text)
        print(f"  Input:  \"{text}\"")
        if result.pii_detected:
            print(f"  PII:    {result.pii_detected}")
            print(f"  Clean:  \"{result.anonymized_text}\"")
        else:
            print(f"  PII:    none detected")
        print(f"  Safe:   {result.is_safe}")
        print()


# ── Demo 2: Anonymize standalone ─────────────────────────────────────────────

def demo_anonymize_method() -> None:
    """Show the anonymize() method directly."""
    print("\n--- DEMO 2: Direct Anonymization ---")

    csf = ContentSafetyFilter()

    texts = [
        "Please update john@example.com's record",
        "The employee's phone is (415) 555-0100",
        "SSN on file: 987-65-4321",
        "Visa card ending in 4111111111111111",
    ]

    print(f"  {'Original':<50} {'Anonymized'}")
    print("  " + "-" * 80)
    for text in texts:
        anonymized, detected = csf.anonymize(text)
        print(f"  {text:<50} {anonymized}")


# ── Demo 3: PII in model output ───────────────────────────────────────────────

def demo_output_pii_scan() -> None:
    """Show that PII in model outputs is also caught."""
    print("\n--- DEMO 3: PII in Model Output ---")
    print("  The model might echo PII from documents. Output scanning catches this.\n")

    csf = ContentSafetyFilter()

    # Simulate model outputs that might contain PII
    model_outputs = [
        "The HR contact is Sarah Johnson at sarah.j@acmecorp.com.",
        "Call the benefits hotline at 1-800-555-0100.",
        "Your account number is 4532015112830366.",
        "The leave policy allows 20 days per year.",  # safe
    ]

    for output in model_outputs:
        result = csf.check_output(output)
        status = "✅ SAFE" if result.is_safe else "⚠️  PII FOUND"
        print(f"  {status} | \"{output[:60]}\"")
        if result.pii_detected:
            print(f"         PII types: {result.pii_detected}")
            print(f"         Cleaned:   \"{result.anonymized_text[:60]}\"")
        print()


# ── Demo 4: PII patterns explained ───────────────────────────────────────────

def demo_pii_patterns() -> None:
    """Explain what each PII pattern matches."""
    print("\n--- DEMO 4: PII Pattern Reference ---")
    print("""
  Pattern       Example                          Action
  ─────────────────────────────────────────────────────────
  email         john.doe@company.com             Anonymize → [EMAIL]
  phone         555-123-4567 / (415) 555-0100    Anonymize → [PHONE]
  ssn           123-45-6789                      Anonymize → [SSN]
  credit_card   4111-1111-1111-1111              Anonymize → [CREDIT_CARD]

  WHY anonymize instead of block?
    Blocking rejects the entire question.
    Anonymizing lets the question through with PII replaced.
    "My email is [EMAIL], how do I reset my password?" is still answerable.

  WHEN to block instead?
    For highly sensitive PII (credit cards, SSNs), you might want to
    block entirely and ask the user to contact support directly.
    Configure this in config.yaml: guardrails.enable_pii_anonymization
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("LESSON 10: PII DETECTION")
    print("=" * 60)
    print("""
ContentSafetyFilter handles PII (guardrails/content_safety.py):
  result = csf.check_input("My email is john@co.com")
  # result.pii_detected = ["email"]
  # result.anonymized_text = "My email is [EMAIL]"

PII is replaced BEFORE the question reaches the LLM.
The LLM never sees real emails, phones, or SSNs.
""")

    demo_pii_detection()
    demo_anonymize_method()
    demo_output_pii_scan()
    demo_pii_patterns()


if __name__ == "__main__":
    main()
