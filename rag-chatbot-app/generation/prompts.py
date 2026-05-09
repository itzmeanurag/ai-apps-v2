"""
generation/prompts.py
PromptAssembler with 12 named templates, personas, and guardrail injection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ── Template registry ─────────────────────────────────────────────────────────

TEMPLATES: dict[str, str] = {
    # Core RAG
    "rag": (
        "{guardrail_prefix}"
        "You are a helpful assistant. Use ONLY the context below to answer.\n"
        "If the answer is not in the context, say 'I don't have enough information.'\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    # Evaluation
    "eval": (
        "You are an expert evaluator. Given the question, context, and answer below, "
        "rate the answer on:\n"
        "1. Faithfulness (0-1): Is the answer grounded in the context?\n"
        "2. Relevance (0-1): Does the answer address the question?\n"
        "3. Completeness (0-1): Is the answer complete?\n\n"
        "Question: {question}\nContext: {context}\nAnswer: {answer}\n\n"
        "Respond in JSON: {{\"faithfulness\": 0.0, \"relevance\": 0.0, \"completeness\": 0.0, "
        "\"explanation\": \"...\"}}"
    ),
    # Memory-aware RAG
    "memory": (
        "{guardrail_prefix}"
        "You are a helpful assistant with memory of past interactions.\n\n"
        "Conversation summary: {summary}\n"
        "Known facts: {facts}\n\n"
        "Context:\n{context}\n\n"
        "Recent conversation:\n{history}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    # Iterative refinement
    "refinement": (
        "You are refining a previous answer based on additional context.\n\n"
        "Original question: {question}\n"
        "Previous answer: {previous_answer}\n"
        "Additional context:\n{context}\n\n"
        "Provide an improved, more accurate answer:"
    ),
    # Summarization
    "summarize": (
        "Summarize the following conversation in 2-3 sentences, "
        "preserving key facts and decisions:\n\n{conversation}\n\nSummary:"
    ),
    # Fact extraction
    "extract_facts": (
        "Extract key facts from this conversation as a JSON list of strings. "
        "Focus on names, dates, decisions, and preferences.\n\n"
        "{conversation}\n\nFacts (JSON list):"
    ),
    # HR persona
    "hr_assistant": (
        "{guardrail_prefix}"
        "You are an HR assistant for {company_name}. "
        "You help employees with HR policies, benefits, and procedures. "
        "Be professional, empathetic, and concise.\n\n"
        "Context:\n{context}\n\n"
        "Employee question: {question}\n\n"
        "HR Assistant:"
    ),
    # Technical support persona
    "tech_support": (
        "{guardrail_prefix}"
        "You are a technical support specialist. "
        "Provide clear, step-by-step solutions based on the documentation below.\n\n"
        "Documentation:\n{context}\n\n"
        "Issue: {question}\n\n"
        "Solution:"
    ),
    # Truthfulness check
    "truthfulness": (
        "Given the following context and answer, determine if the answer is truthful "
        "and grounded in the context. Score from 0.0 (completely false) to 1.0 (fully truthful).\n\n"
        "Context: {context}\nAnswer: {answer}\n\n"
        "Respond in JSON: {{\"score\": 0.0, \"reasoning\": \"...\"}}"
    ),
    # Query rewriting
    "query_rewrite": (
        "Rewrite the following question to be more specific and searchable, "
        "preserving the original intent. Return only the rewritten question.\n\n"
        "Original: {question}\nRewritten:"
    ),
    # Hypothetical document embedding (HyDE)
    "hyde": (
        "Write a short passage (2-3 sentences) that would answer the following question. "
        "This will be used for document retrieval.\n\n"
        "Question: {question}\nPassage:"
    ),
    # Guardrail violation response
    "safety_refusal": (
        "I'm unable to assist with that request as it may violate content safety guidelines. "
        "Please rephrase your question or ask about a different topic.\n\n"
        "Flagged categories: {categories}"
    ),
}

# ── Persona definitions ───────────────────────────────────────────────────────

PERSONAS: dict[str, dict[str, str]] = {
    "default": {
        "name": "Assistant",
        "description": "A helpful, accurate, and concise assistant.",
        "guardrail_prefix": "",
    },
    "hr": {
        "name": "HR Assistant",
        "description": "An empathetic HR specialist focused on company policies.",
        "guardrail_prefix": (
            "IMPORTANT: Only discuss HR-related topics. "
            "Do not provide legal advice. Refer complex cases to HR management.\n\n"
        ),
    },
    "tech": {
        "name": "Tech Support",
        "description": "A technical expert providing clear, actionable solutions.",
        "guardrail_prefix": (
            "IMPORTANT: Base all answers on the provided documentation. "
            "If unsure, recommend escalating to the engineering team.\n\n"
        ),
    },
    "admin": {
        "name": "Admin Assistant",
        "description": "A comprehensive assistant with access to all knowledge bases.",
        "guardrail_prefix": "",
    },
}

# ── PromptAssembler ───────────────────────────────────────────────────────────

@dataclass
class PromptAssembler:
    """
    Assembles prompts from named templates with variable substitution,
    persona injection, and guardrail prefixes.

    Usage:
        pa = PromptAssembler(persona="hr")
        prompt = pa.build("rag", context="...", question="What is the PTO policy?")
    """

    persona: str = "default"
    extra_guardrails: str = ""

    def build(self, template_name: str, **kwargs) -> str:
        """
        Build a prompt from a named template.

        Args:
            template_name: Key in TEMPLATES dict.
            **kwargs: Variables to substitute into the template.

        Returns:
            Formatted prompt string.
        """
        if template_name not in TEMPLATES:
            raise ValueError(
                f"Unknown template '{template_name}'. "
                f"Available: {list(TEMPLATES.keys())}"
            )

        template = TEMPLATES[template_name]
        persona_data = PERSONAS.get(self.persona, PERSONAS["default"])

        # Inject guardrail prefix if template uses it
        if "{guardrail_prefix}" in template:
            kwargs.setdefault(
                "guardrail_prefix",
                persona_data["guardrail_prefix"] + self.extra_guardrails,
            )

        # Fill remaining defaults
        kwargs.setdefault("company_name", "the company")

        try:
            return template.format(**kwargs)
        except KeyError as exc:
            raise ValueError(
                f"Template '{template_name}' requires variable {exc}. "
                f"Provided: {list(kwargs.keys())}"
            ) from exc

    def list_templates(self) -> list[str]:
        return list(TEMPLATES.keys())

    def list_personas(self) -> list[str]:
        return list(PERSONAS.keys())

    def get_template(self, name: str) -> str:
        if name not in TEMPLATES:
            raise ValueError(f"Template '{name}' not found.")
        return TEMPLATES[name]

    def add_template(self, name: str, template: str) -> None:
        """Register a custom template at runtime."""
        TEMPLATES[name] = template

    def set_persona(self, persona: str) -> None:
        if persona not in PERSONAS:
            raise ValueError(f"Persona '{persona}' not found. Available: {list(PERSONAS.keys())}")
        self.persona = persona
