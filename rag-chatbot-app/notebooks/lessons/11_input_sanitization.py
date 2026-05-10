"""
LESSON 11: Input Sanitization and Model Governance
===================================================
CONCEPT: Defending against low-level input attacks and model tampering

WHAT THIS DOES:
  Demonstrates ModelGovernance from guardrails/model_governance.py:
    - Input sanitization: null bytes, NFKC normalization, control chars
    - Pickle detection: blocks .pkl files and files with pickle magic bytes
    - SHA-256 checksum verification: detects model file tampering
    - Supply chain validation: flags suspicious package names

WHY THIS MATTERS:
  Attackers don't just use words — they use invisible characters,
  homoglyphs (Cyrillic 'а' looks like Latin 'a'), and null bytes
  to bypass regex-based guardrails.
  Model governance ensures the AI model files themselves haven't been tampered with.

RUN (from project root):
  python notebooks/lessons/11_input_sanitization.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from the actual project module (guardrails/model_governance.py)
from guardrails.model_governance import (
    sanitize_input,
    is_pickle_file,
    ModelChecksumRegistry,
    validate_package_name,
    ModelGovernance,
)
from src.config import cfg


# ── Demo 1: Input sanitization ────────────────────────────────────────────────

def demo_input_sanitization() -> None:
    """Show how dangerous characters are removed from inputs."""
    print("\n--- DEMO 1: Input Sanitization ---")
    print("  Removes null bytes, normalizes Unicode, strips control chars.\n")

    test_inputs = [
        ("Normal question about leave policy",          "Clean input"),
        ("Question with \x00 null byte",                "Null byte injection"),
        ("Cаt policy",                                  "Homoglyph: Cyrillic 'а' in 'Cat'"),
        ("Query with\x01\x02control\x03chars",          "Control characters"),
        ("Ｆｕｌｌｗｉｄｔｈ text",                    "Fullwidth Unicode (NFKC normalizes)"),
        ("Normal\ttab and\nnewline are kept",            "Tabs/newlines preserved"),
    ]

    for text, description in test_inputs:
        cleaned = sanitize_input(text)
        changed = " ← MODIFIED" if cleaned != text else ""
        print(f"  {description}")
        print(f"    Input:  {repr(text)}")
        print(f"    Output: {repr(cleaned)}{changed}")
        print()

    print("""
  WHY NFKC normalization?
    Cyrillic 'а' (U+0430) looks identical to Latin 'a' (U+0061).
    An attacker writes "hаck" with a Cyrillic 'а' to bypass "hack" regex.
    NFKC normalization converts all lookalike characters to their canonical form.
    After normalization, "hаck" becomes "hack" and the regex catches it.
""")


# ── Demo 2: Pickle detection ──────────────────────────────────────────────────

def demo_pickle_detection() -> None:
    """Show how pickle files are detected and blocked."""
    print("\n--- DEMO 2: Pickle Detection ---")
    print("  Pickle files can execute arbitrary code when loaded.")
    print("  We block them by extension AND by magic bytes.\n")

    # Test by extension
    test_files = [
        ("model.pkl",         True,  "Pickle by extension"),
        ("model.pickle",      True,  "Pickle by extension"),
        ("model.safetensors", False, "SafeTensors (safe format)"),
        ("model.gguf",        False, "GGUF (used by Ollama)"),
        ("weights.bin",       False, "Binary weights (safe)"),
        ("data.json",         False, "JSON (safe)"),
    ]

    for filename, expected_blocked, description in test_files:
        is_blocked = is_pickle_file(filename)
        status = "🚫 BLOCKED" if is_blocked else "✅ ALLOWED"
        match = "✓" if is_blocked == expected_blocked else "✗ UNEXPECTED"
        print(f"  {status} | {description} ({filename}) {match}")

    # Test with actual pickle magic bytes
    print("\n  Testing magic byte detection (catches renamed pickle files)...")
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        # Write pickle protocol 2 magic bytes
        f.write(b"\x80\x02")
        f.write(b"some content")
        temp_path = f.name

    is_blocked = is_pickle_file(temp_path)
    print(f"  File named .bin but with pickle magic bytes: {'🚫 BLOCKED' if is_blocked else '✅ ALLOWED'}")
    Path(temp_path).unlink()

    print("""
  WHY block pickle?
    import pickle; pickle.load(open("model.pkl", "rb"))
    This executes ANY Python code embedded in the file.
    A malicious model.pkl could delete files, steal data, or install malware.
    SafeTensors format is the safe alternative — it cannot execute code.
""")


# ── Demo 3: SHA-256 checksum verification ─────────────────────────────────────

def demo_checksum_verification() -> None:
    """Show how model file integrity is verified."""
    print("\n--- DEMO 3: SHA-256 Checksum Verification ---")
    print("  Detects if a model file has been tampered with.\n")

    # Create a temporary file to simulate a model
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False, mode="wb") as f:
        f.write(b"This is a simulated model file with weights data " * 100)
        model_path = f.name

    registry = ModelChecksumRegistry("./data/model_checksums_demo.json")

    # Register the model (first-time trust)
    checksum = registry.register("demo-model-v1", model_path)
    print(f"  Registered 'demo-model-v1'")
    print(f"  SHA-256: {checksum[:32]}...")

    # Verify (should pass)
    ok = registry.verify("demo-model-v1", model_path)
    print(f"\n  Verification (original file): {'✅ PASS' if ok else '❌ FAIL'}")

    # Simulate tampering
    with open(model_path, "ab") as f:
        f.write(b"TAMPERED")

    ok_after = registry.verify("demo-model-v1", model_path)
    print(f"  Verification (tampered file): {'✅ PASS' if ok_after else '❌ FAIL — TAMPERING DETECTED'}")

    # Clean up
    Path(model_path).unlink()
    Path("./data/model_checksums_demo.json").unlink(missing_ok=True)

    print("""
  WHY checksum verification?
    An attacker could replace your fine-tuned model with a malicious one.
    SHA-256 checksums detect any change to the file, no matter how small.
    Register the checksum when you first download/train the model.
    Verify before every load in production.
""")


# ── Demo 4: Supply chain validation ──────────────────────────────────────────

def demo_supply_chain() -> None:
    """Show package name validation against typosquatting."""
    print("\n--- DEMO 4: Supply Chain Validation ---")
    print("  Flags suspicious package names that might be typosquatting attacks.\n")

    test_packages = [
        ("langchain",      True,  "Legitimate package"),
        ("chromadb",       True,  "Legitimate package"),
        ("colourama",      False, "Typosquat of 'colorama'"),
        ("requestes",      False, "Typosquat of 'requests'"),
        ("urllib4",        False, "Typosquat of 'urllib3'"),
        ("numpy",          True,  "Legitimate package"),
        ("pycrypto2",      False, "Typosquat of 'pycryptodome'"),
    ]

    for package, expected_safe, description in test_packages:
        is_safe, reason = validate_package_name(package)
        status = "✅ OK" if is_safe else "🚫 FLAGGED"
        print(f"  {status} | {package:<20} — {description}")
        if not is_safe:
            print(f"         Reason: {reason}")

    print("""
  WHY supply chain validation?
    Attackers publish packages with names similar to popular ones.
    "pip install colourama" installs malware, not colorama.
    Always verify package names before adding to requirements.txt.
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("LESSON 11: INPUT SANITIZATION & MODEL GOVERNANCE")
    print("=" * 60)
    print("""
ModelGovernance (guardrails/model_governance.py):
  gov = ModelGovernance(cfg.guardrails.checksum_file)
  clean = gov.sanitize(user_input)        # remove null bytes, normalize
  gov.assert_model_safe("model", path)    # verify checksum, block pickle

4 protections:
  1. Input sanitization  — null bytes, homoglyphs, control chars
  2. Pickle detection    — by extension AND magic bytes
  3. Checksum registry   — SHA-256 tamper detection
  4. Supply chain        — typosquatting detection
""")

    demo_input_sanitization()
    demo_pickle_detection()
    demo_checksum_verification()
    demo_supply_chain()


if __name__ == "__main__":
    main()
