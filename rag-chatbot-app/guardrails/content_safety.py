"""
guardrails/content_safety.py

GuardrailConfig dataclass + Guardrails class.

6 categories: SEXUAL, VIOLENCE, HATE, INSULTS, MISCONDUCT, PROMPT_ATTACK
PII types:    EMAIL, PHONE, SSN, CREDIT_CARD, ADDRESS, PASSPORT
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Category enum ─────────────────────────────────────────────────────────────

class SafetyCategory(str, Enum):
    SEXUAL       = "SEXUAL"
    VIOLENCE     = "VIOLENCE"
    HATE         = "HATE"
    INSULTS      = "INSULTS"
    MISCONDUCT   = "MISCONDUCT"
    PROMPT_ATTACK = "PROMPT_ATTACK"


# ── PII type enum ─────────────────────────────────────────────────────────────

class PIIType(str, Enum):
    EMAIL       = "EMAIL"
    PHONE       = "PHONE"
    SSN         = "SSN"
    CREDIT_CARD = "CREDIT_CARD"
    ADDRESS     = "ADDRESS"
    PASSPORT    = "PASSPORT"


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class GuardrailConfig:
    """Configuration for the Guardrails filter."""
    # Category levels: "HIGH" | "MEDIUM" | "LOW" | "OFF"
    sexual:        str = "HIGH"
    violence:      str = "HIGH"
    hate:          str = "HIGH"
    insults:       str = "HIGH"
    misconduct:    str = "HIGH"
    prompt_attack: str = "HIGH"

    # PII handling: "ANONYMIZE" | "BLOCK" | "OFF"
    pii_email:       str = "ANONYMIZE"
    pii_phone:       str = "ANONYMIZE"
    pii_ssn:         str = "ANONYMIZE"
    pii_credit_card: str = "BLOCK"
    pii_address:     str = "ANONYMIZE"
    pii_passport:    str = "BLOCK"

    # Optional LLM-based classification (slower but more accurate)
    use_llm_check: bool = False

    # Max input length (chars)
    max_input_length: int = 2000


# ── Regex patterns ────────────────────────────────────────────────────────────

_CATEGORY_PATTERNS: dict[SafetyCategory, list[str]] = {
    SafetyCategory.SEXUAL: [
        r"\b(porn|pornography|explicit\s+sex|nude|nudity|erotic|hentai|xxx)\b",
        r"\b(sexual\s+content|adult\s+content|nsfw)\b",
    ],
    SafetyCategory.VIOLENCE: [
        r"\b(kill|murder|assassinate|bomb|explode|shoot|stab|torture)\b",
        r"\b(how\s+to\s+(make|build)\s+(a\s+)?(bomb|weapon|gun))\b",
        r"\b(mass\s+shooting|terrorist\s+attack|genocide)\b",
    ],
    SafetyCategory.HATE: [
        r"\b(hate\s+speech|racial\s+slur|white\s+supremac|neo.?nazi)\b",
        r"\b(ethnic\s+cleansing|antisemit|islamophob)\b",
    ],
    SafetyCategory.INSULTS: [
        r"\b(idiot|moron|stupid|dumb|retard|imbecile)\b",
        r"\b(go\s+to\s+hell|shut\s+up|you\s+suck)\b",
    ],
    SafetyCategory.MISCONDUCT: [
        r"\b(fraud|scam|phishing|money\s+laundering|insider\s+trading)\b",
        r"\b(bribe|embezzle|extort|blackmail)\b",
        r"\b(how\s+to\s+hack|sql\s+injection|exploit)\b",
    ],
    SafetyCategory.PROMPT_ATTACK: [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
        r"you\s+are\s+now\s+(a\s+)?(different|new|another)\s+(ai|model|assistant)",
        r"(system\s+prompt|jailbreak|dan\s+mode|developer\s+mode)",
        r"forget\s+(everything|all)\s+(you|i)\s+(know|told)",
        r"pretend\s+(you\s+are|to\s+be)\s+(an?\s+)?(evil|unrestricted|unfiltered)",
        r"</?(system|user|assistant)>",
    ],
}

_PII_PATTERNS: dict[PIIType, re.Pattern] = {
    PIIType.EMAIL: re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    ),
    PIIType.PHONE: re.compile(
        r"\b(\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b"
    ),
    PIIType.SSN: re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
    PIIType.CREDIT_CARD: re.compile(
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|"
        r"3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b"
    ),
    PIIType.ADDRESS: re.compile(
        r"\b\d{1,5}\s+[A-Za-z0-9\s]{3,30}\s+(Street|St|Avenue|Ave|"
        r"Boulevard|Blvd|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|Way)\b",
        re.IGNORECASE,
    ),
    PIIType.PASSPORT: re.compile(r"\b[A-Z]{1,2}[0-9]{6,9}\b"),
}

_PII_REPLACEMENTS: dict[PIIType, str] = {
    PIIType.EMAIL:       "[EMAIL]",
    PIIType.PHONE:       "[PHONE]",
    PIIType.SSN:         "[SSN]",
    PIIType.CREDIT_CARD: "[CREDIT_CARD]",
    PIIType.ADDRESS:     "[ADDRESS]",
    PIIType.PASSPORT:    "[PASSPORT]",
}


# ── Guardrails class ──────────────────────────────────────────────────────────

class Guardrails:
    """
    Multi-layer content safety filter.

    check_input(text)  -> dict with keys: safe, violations, pii_found,
                          cleaned_text, message, blocked
    check_output(text) -> same structure
    """

    def __init__(self, config: Optional[GuardrailConfig] = None) -> None:
        self._cfg = config or GuardrailConfig()
        self._compiled: dict[SafetyCategory, list[re.Pattern]] = {
            cat: [re.compile(p, re.IGNORECASE) for p in patterns]
            for cat, patterns in _CATEGORY_PATTERNS.items()
        }
        self._stats = {
            "total_checks": 0,
            "blocked_count": 0,
            "violations_by_category": {},
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def check_input(self, text: str) -> dict:
        """
        Check user input for safety violations and PII.

        Returns:
            {
              "safe": bool,
              "violations": [{"category": str, "level": str}],
              "pii_found": [str],
              "cleaned_text": str,   # PII anonymized
              "message": str,        # human-readable reason if blocked
              "blocked": bool,
            }
        """
        self._stats["total_checks"] += 1

        # Length check
        if len(text) > self._cfg.max_input_length:
            text = text[: self._cfg.max_input_length]

        violations = self._check_categories(text)
        pii_found, cleaned = self._handle_pii(text)
        blocked = len(violations) > 0

        if blocked:
            self._stats["blocked_count"] += 1
            for v in violations:
                cat = v["category"]
                self._stats["violations_by_category"][cat] = (
                    self._stats["violations_by_category"].get(cat, 0) + 1
                )

        message = ""
        if blocked:
            cats = ", ".join(v["category"] for v in violations)
            message = f"Content blocked: {cats}"

        return {
            "safe": not blocked,
            "violations": violations,
            "pii_found": pii_found,
            "cleaned_text": cleaned,
            "message": message,
            "blocked": blocked,
        }

    def check_output(self, text: str) -> dict:
        """
        Scan model output for safety violations and PII leakage.
        Same return structure as check_input.
        """
        violations = self._check_categories(text)
        pii_found, cleaned = self._handle_pii(text)
        blocked = len(violations) > 0 or len(pii_found) > 0

        return {
            "safe": not blocked,
            "violations": violations,
            "pii_found": pii_found,
            "pii_found_in_output": pii_found,
            "cleaned_text": cleaned,
            "message": "Output contains unsafe content or PII" if blocked else "",
            "blocked": blocked,
        }

    def get_stats(self) -> dict:
        return dict(self._stats)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check_categories(self, text: str) -> list[dict]:
        violations = []
        for category, patterns in self._compiled.items():
            level = getattr(self._cfg, category.value.lower(), "HIGH")
            if level == "OFF":
                continue
            if any(p.search(text) for p in patterns):
                violations.append({"category": category.value, "level": level})
        return violations

    def _handle_pii(self, text: str) -> tuple[list[str], str]:
        """
        Detect and handle PII.
        Returns (list_of_pii_types_found, cleaned_text).
        """
        found: list[str] = []
        result = text

        for pii_type, pattern in _PII_PATTERNS.items():
            action = getattr(self._cfg, f"pii_{pii_type.value.lower()}", "ANONYMIZE")
            if action == "OFF":
                continue
            if pattern.search(result):
                found.append(pii_type.value)
                if action in ("ANONYMIZE", "BLOCK"):
                    result = pattern.sub(_PII_REPLACEMENTS[pii_type], result)

        return found, result


# ── Backward-compat alias ─────────────────────────────────────────────────────

# Lessons and older code may import ContentSafetyFilter
class ContentSafetyFilter(Guardrails):
    """Alias for Guardrails kept for backward compatibility."""

    def check_input(self, text: str):  # type: ignore[override]
        """Return a SafetyResult-like object for backward compat."""
        result = super().check_input(text)
        return _SafetyResultCompat(result)

    def check_output(self, text: str):  # type: ignore[override]
        result = super().check_output(text)
        return _SafetyResultCompat(result)


@dataclass
class _SafetyResultCompat:
    """Thin wrapper so old code using .is_safe / .flagged_categories still works."""
    _data: dict

    @property
    def is_safe(self) -> bool:
        return self._data["safe"]

    @property
    def flagged_categories(self) -> list:
        return [v["category"] for v in self._data.get("violations", [])]

    @property
    def pii_detected(self) -> list:
        return self._data.get("pii_found", [])

    @property
    def anonymized_text(self) -> Optional[str]:
        return self._data.get("cleaned_text")

    @property
    def reason(self) -> str:
        return self._data.get("message", "")

    def to_dict(self) -> dict:
        return self._data


# Convenience function used by lessons
def pii_check_and_anonymize(text: str, config: Optional[GuardrailConfig] = None) -> dict:
    """Standalone PII check + anonymize. Returns result dict."""
    g = Guardrails(config or GuardrailConfig())
    result = g.check_input(text)
    return {
        "text": result["cleaned_text"],
        "pii_found": result["pii_found"],
        "blocked": result["blocked"],
        "block_reason": result["message"] if result["blocked"] else "",
    }
