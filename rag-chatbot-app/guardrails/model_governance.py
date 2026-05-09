"""
guardrails/model_governance.py
SHA-256 checksums, pickle detection, supply chain validation,
and input sanitization (null bytes, NFKC, control chars).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Optional


# ── Input sanitization ────────────────────────────────────────────────────────

# Control characters except tab (0x09), newline (0x0A), carriage return (0x0D)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def sanitize_input(text: str) -> str:
    """
    Clean user input:
    1. Remove null bytes
    2. Apply NFKC Unicode normalization
    3. Strip dangerous control characters
    """
    # 1. Null bytes
    text = text.replace("\x00", "")
    # 2. NFKC normalization (collapses lookalike chars, fullwidth, etc.)
    text = unicodedata.normalize("NFKC", text)
    # 3. Control characters
    text = _CONTROL_CHAR_RE.sub("", text)
    return text.strip()


# ── Pickle detection ──────────────────────────────────────────────────────────

# Magic bytes for common serialization formats that can execute code
_DANGEROUS_MAGIC: list[bytes] = [
    b"\x80\x02",  # pickle protocol 2
    b"\x80\x03",  # pickle protocol 3
    b"\x80\x04",  # pickle protocol 4
    b"\x80\x05",  # pickle protocol 5
    b"cos\n",     # pickle GLOBAL opcode
    b"cposixpath",
]


def is_pickle_file(path: Path | str) -> bool:
    """Return True if the file looks like a pickle (potentially unsafe)."""
    path = Path(path)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        return True
    try:
        with open(path, "rb") as fh:
            header = fh.read(16)
        return any(header.startswith(magic) for magic in _DANGEROUS_MAGIC)
    except (OSError, IOError):
        return False


# ── SHA-256 checksum management ───────────────────────────────────────────────

class ModelChecksumRegistry:
    """
    Maintains a JSON registry of model file SHA-256 checksums.
    Use to detect tampering or unexpected model swaps.
    """

    def __init__(self, registry_path: Path | str) -> None:
        self._path = Path(registry_path)
        self._registry: dict[str, str] = self._load()

    def _load(self) -> dict[str, str]:
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(self._registry, fh, indent=2)

    @staticmethod
    def compute_checksum(path: Path | str) -> str:
        """Compute SHA-256 hex digest of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def register(self, model_id: str, path: Path | str) -> str:
        """Compute and store checksum for a model file. Returns the checksum."""
        checksum = self.compute_checksum(path)
        self._registry[model_id] = checksum
        self._save()
        return checksum

    def verify(self, model_id: str, path: Path | str) -> bool:
        """Return True if the file matches the registered checksum."""
        if model_id not in self._registry:
            return False  # not registered → treat as unverified
        expected = self._registry[model_id]
        actual = self.compute_checksum(path)
        return actual == expected

    def get_checksum(self, model_id: str) -> Optional[str]:
        return self._registry.get(model_id)

    def list_models(self) -> dict[str, str]:
        return dict(self._registry)


# ── Supply chain validation ───────────────────────────────────────────────────

_ALLOWED_PACKAGE_SOURCES = {"pypi", "conda-forge", "local"}

# Known malicious / typosquatted package names (non-exhaustive example list)
_KNOWN_MALICIOUS: set[str] = {
    "colourama",   # typosquat of colorama
    "python-dateutil2",
    "requestes",
    "urllib4",
    "pycrypto2",
}


def validate_package_name(name: str) -> tuple[bool, str]:
    """
    Basic supply-chain check on a package name.
    Returns (is_safe, reason).
    """
    name_lower = name.lower().strip()
    if name_lower in _KNOWN_MALICIOUS:
        return False, f"Package '{name}' is on the known-malicious list."
    # Flag packages with suspicious patterns
    if re.search(r"[^a-z0-9_\-\.]", name_lower):
        return False, f"Package name '{name}' contains unexpected characters."
    return True, "ok"


# ── Governance facade ─────────────────────────────────────────────────────────

class ModelGovernance:
    """
    High-level facade combining all governance checks.

    Usage:
        gov = ModelGovernance(cfg.guardrails.checksum_file)
        clean_text = gov.sanitize("user input")
        gov.assert_model_safe("my_model", "/path/to/model.bin")
    """

    def __init__(self, checksum_file: str) -> None:
        self._registry = ModelChecksumRegistry(checksum_file)

    def sanitize(self, text: str) -> str:
        """Sanitize a text input."""
        return sanitize_input(text)

    def assert_model_safe(self, model_id: str, path: Path | str) -> None:
        """
        Raise RuntimeError if the model file is a pickle or fails checksum.
        If not yet registered, register it (first-run trust-on-first-use).
        """
        path = Path(path)
        if not path.exists():
            return  # remote / API model – nothing to check locally

        if is_pickle_file(path):
            raise RuntimeError(
                f"Model file '{path}' appears to be a pickle. "
                "Pickle files can execute arbitrary code. Refusing to load."
            )

        if model_id not in self._registry.list_models():
            # First time seeing this model – register it
            checksum = self._registry.register(model_id, path)
            print(f"[governance] Registered model '{model_id}' with checksum {checksum[:16]}…")
        elif not self._registry.verify(model_id, path):
            raise RuntimeError(
                f"Checksum mismatch for model '{model_id}'. "
                "The file may have been tampered with."
            )

    def validate_package(self, name: str) -> None:
        """Raise ValueError if the package name looks suspicious."""
        ok, reason = validate_package_name(name)
        if not ok:
            raise ValueError(f"Supply chain check failed: {reason}")
