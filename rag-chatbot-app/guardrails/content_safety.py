"""
guardrails/content_safety.py
6-category regex filter, PII detection + anonymization, optional LLM classification,
and output scanning.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# ── Category definitions ──────────────────────────────────────────────────────

class SafetyCategory(str, Enum):
    SEXUAL = "sexual"
    VIOLENCE = "violence"
    HATE = "hate"
    INSULTS = "insults"
    MISCONDUCT = "misconduct"
    PROMPT_ATTACK = "prompt_attack"


_PATTERNS: dict[SafetyCategory, list[str]] = {
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
    ],
    SafetyCategory.PROMPT_ATTACK: [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
        r"you\s+are\s+now\s+(a\s+)?(different|new|another)\s+(ai|model|assistant)",
        r"(system\s+prompt|jailbreak|dan\s+mode|developer\s+mode)",
        r"forget\s+(everything|all)\s+(you|i)\s+(know|told)",
        r"pretend\s+(you\s+are|to\s+be)\s+(an?\s+)?(evil|unrestricted|unfiltered)",
        r"</?(system|user|assistant)>",  # injection via fake tags
    ],
}

# ── PII patterns ──────────────────────────────────────────────────────────────

_PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(
        r"\b(\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b"
    ),
    "ssn": re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
    "credit_card": re.compile(
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|"
        r"6(?:011|5[0-9]{2})[0-9]{12})\b"
    ),
}

_PII_REPLACEMENTS: dict[str, str] = {
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "ssn": "[SSN]",
    "credit_card": "[CREDIT_CARD]",
}

# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SafetyResult:
    is_safe: bool
    flagged_categories: list[SafetyCategory] = field(default_factory=list)
    pii_detected: list[str] = field(default_factory=list)
    anonymized_text: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "is_safe": self.is_safe,
            "flagged_categories": [c.value for c in self.flagged_categories],
            "pii_detected": self.pii_detected,
            "anonymized_text": self.anonymized_text,
            "reason": self.reason,
        }


# ── Main filter class ─────────────────────────────────────────────────────────

class ContentSafetyFilter:
    """
    Multi-layer content safety filter.

    Usage:
        csf = ContentSafetyFilter()
        result = csf.check_input("some user text")
        if not result.is_safe:
            raise ValueError(result.reason)
        safe_text = result.anonymized_text or "some user text"
    """

    def __init__(self, enable_llm_classification: bool = False) -> None:
        self._compiled: dict[SafetyCategory, list[re.Pattern]] = {
            cat: [re.compile(p, re.IGNORECASE) for p in patterns]
            for cat, patterns in _PATTERNS.items()
        }
        self._enable_llm = enable_llm_classification

    # ── Public API ────────────────────────────────────────────────────────────

    def check_input(self, text: str) -> SafetyResult:
        """Check user input for safety violations and PII."""
        flagged = self._regex_check(text)
        pii_types, anonymized = self._pii_check(text)

        if flagged:
            return SafetyResult(
                is_safe=False,
                flagged_categories=flagged,
                pii_detected=pii_types,
                anonymized_text=anonymized,
                reason=f"Content flagged: {', '.join(c.value for c in flagged)}",
            )

        if self._enable_llm:
            llm_flagged = self._llm_classify(anonymized or text)
            if llm_flagged:
                return SafetyResult(
                    is_safe=False,
                    flagged_categories=llm_flagged,
                    pii_detected=pii_types,
                    anonymized_text=anonymized,
                    reason=f"LLM classification flagged: {', '.join(c.value for c in llm_flagged)}",
                )

        return SafetyResult(
            is_safe=True,
            pii_detected=pii_types,
            anonymized_text=anonymized,
        )

    def check_output(self, text: str) -> SafetyResult:
        """Scan model output for safety violations and PII leakage."""
        flagged = self._regex_check(text)
        pii_types, anonymized = self._pii_check(text)

        if flagged or pii_types:
            return SafetyResult(
                is_safe=False,
                flagged_categories=flagged,
                pii_detected=pii_types,
                anonymized_text=anonymized,
                reason="Output contains unsafe content or PII",
            )
        return SafetyResult(is_safe=True)

    def anonymize(self, text: str) -> tuple[str, list[str]]:
        """Return (anonymized_text, list_of_detected_pii_types)."""
        _, anonymized = self._pii_check(text)
        detected = [k for k, p in _PII_PATTERNS.items() if p.search(text)]
        return anonymized or text, detected

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _regex_check(self, text: str) -> list[SafetyCategory]:
        flagged: list[SafetyCategory] = []
        for category, patterns in self._compiled.items():
            if any(p.search(text) for p in patterns):
                flagged.append(category)
        return flagged

    def _pii_check(self, text: str) -> tuple[list[str], Optional[str]]:
        detected: list[str] = []
        result = text
        for pii_type, pattern in _PII_PATTERNS.items():
            if pattern.search(result):
                detected.append(pii_type)
                result = pattern.sub(_PII_REPLACEMENTS[pii_type], result)
        return detected, result if detected else None

    def _llm_classify(self, text: str) -> list[SafetyCategory]:
        """
        Optional LLM-based classification.
        Requires OPENAI_API_KEY or similar in environment.
        Returns list of flagged categories (empty = safe).
        """
        try:
            import openai  # type: ignore

            prompt = (
                "Classify the following text for safety issues. "
                "Reply with a JSON list of categories from: "
                f"{[c.value for c in SafetyCategory]}. "
                "Reply with [] if safe.\n\nText: " + text[:500]
            )
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=64,
                temperature=0,
            )
            import json
            raw = response.choices[0].message.content.strip()
            cats = json.loads(raw)
            return [SafetyCategory(c) for c in cats if c in SafetyCategory._value2member_map_]
        except Exception:
            return []  # fail open – regex already ran
