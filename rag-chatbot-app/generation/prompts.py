"""
generation/prompts.py

PromptAssembler with exactly 12 named templates (spec-required names):
  basic_chat, rag_simple, rag_with_history, rag_for_evaluation,
  eval_combined, eval_relevance, eval_groundedness, refine_query,
  memory_summarize_new, memory_summarize_update, memory_extract_facts,
  memory_demo_chat

Global instance: prompts = PromptAssembler()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── 12 named templates ────────────────────────────────────────────────────────

_TEMPLATES: dict[str, str] = {
    # 1. Basic chat (no RAG)
    "basic_chat": (
        "You are a helpful assistant. Answer clearly and concisely.\n\n"
        "Question: {question}\n\nAnswer:"
    ),

    # 2. Simple RAG (no history)
    "rag_simple": (
        "You are a helpful assistant. Use ONLY the context below to answer.\n"
        "If the answer is not in the context, say 'I don't have enough information.'\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\nAnswer:"
    ),

    # 3. RAG with conversation history
    "rag_with_history": (
        "You are a helpful document assistant.\n\n"
        "SAFETY RULES: Do NOT invent information not in the context.\n\n"
        "CONVERSATION SUMMARY:\n{summary}\n\n"
        "KEY FACTS:\n{facts}\n\n"
        "RECENT CONVERSATION:\n{history}\n\n"
        "CONTEXT DOCUMENTS:\n{context}\n\n"
        "Question: {question}\n\nAnswer:"
    ),

    # 4. RAG for evaluation testing
    "rag_for_evaluation": (
        "Answer the question based ONLY on the provided context. "
        "Cite the source document when possible.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\nAnswer:"
    ),

    # 5. Combined evaluation (relevance + groundedness)
    "eval_combined": (
        "You are an evaluation judge. Score the answer on two metrics.\n\n"
        "Question: {question}\n"
        "Context: {context}\n"
        "Answer: {answer}\n\n"
        "Score each from 0.0 to 1.0:\n"
        "- relevance: Does the answer address the question?\n"
        "- groundedness: Is every claim supported by the context?\n\n"
        "Respond ONLY with valid JSON:\n"
        "{{\"relevance\": 0.0, \"groundedness\": 0.0, \"explanation\": \"...\"}}"
    ),

    # 6. Relevance-only evaluation
    "eval_relevance": (
        "Score whether the answer addresses the question (0.0–1.0).\n\n"
        "Question: {question}\nAnswer: {answer}\n\n"
        "Respond ONLY with valid JSON: {{\"score\": 0.0, \"reasoning\": \"...\"}}"
    ),

    # 7. Groundedness-only evaluation
    "eval_groundedness": (
        "Score whether the answer is supported by the context (0.0–1.0).\n"
        "1.0 = every claim is in the context. 0.0 = entirely hallucinated.\n\n"
        "Context: {context}\nAnswer: {answer}\n\n"
        "Respond ONLY with valid JSON: {{\"score\": 0.0, \"reasoning\": \"...\"}}"
    ),

    # 8. Query refinement
    "refine_query": (
        "Rewrite the following query to be more specific and searchable. "
        "Return ONLY the improved query.\n\n"
        "Original: {query}\nImproved:"
    ),

    # 9. Summarize new conversation
    "memory_summarize_new": (
        "Summarize the following conversation in 2-3 sentences, "
        "preserving key facts and decisions:\n\n"
        "{conversation}\n\nSummary:"
    ),

    # 10. Update existing summary
    "memory_summarize_update": (
        "Merge these two summaries into one concise summary (2-3 sentences):\n\n"
        "Existing summary: {existing_summary}\n\n"
        "New conversation: {new_conversation}\n\n"
        "Merged summary:"
    ),

    # 11. Extract key facts
    "memory_extract_facts": (
        "Extract key facts from this conversation as a JSON list of strings. "
        "Focus on names, dates, decisions, and preferences.\n\n"
        "{conversation}\n\nFacts (JSON list):"
    ),

    # 12. Memory demo chat
    "memory_demo_chat": (
        "You are a helpful assistant with memory of past interactions.\n\n"
        "KEY FACTS FROM PREVIOUS CONVERSATIONS:\n{facts}\n\n"
        "SUMMARY OF EARLIER CONVERSATION:\n{summary}\n\n"
        "RECENT EXCHANGES:\n{history}\n\n"
        "Question: {question}\n\nAnswer:"
    ),
}

# Extra templates used internally (not in the 12 spec names but needed)
_EXTRA_TEMPLATES: dict[str, str] = {
    "eval": (
        "You are an expert evaluator. Rate the answer on:\n"
        "1. Faithfulness (0-1): Is the answer grounded in the context?\n"
        "2. Relevance (0-1): Does the answer address the question?\n"
        "3. Completeness (0-1): Is the answer complete?\n\n"
        "Question: {question}\nContext: {context}\nAnswer: {answer}\n\n"
        "Respond in JSON: {{\"faithfulness\": 0.0, \"relevance\": 0.0, "
        "\"completeness\": 0.0, \"explanation\": \"...\"}}"
    ),
    "memory": (
        "You are a helpful assistant with memory of past interactions.\n\n"
        "Conversation summary: {summary}\n"
        "Known facts: {facts}\n\n"
        "Context:\n{context}\n\n"
        "Recent conversation:\n{history}\n\n"
        "Question: {question}\n\nAnswer:"
    ),
    "summarize": (
        "Summarize the following conversation in 2-3 sentences:\n\n"
        "{conversation}\n\nSummary:"
    ),
    "extract_facts": (
        "Extract key facts from this conversation as a JSON list of strings.\n\n"
        "{conversation}\n\nFacts (JSON list):"
    ),
    "truthfulness": (
        "Score whether the answer is truthful and grounded in the context "
        "(0.0 = false, 1.0 = fully truthful).\n\n"
        "Context: {context}\nAnswer: {answer}\n\n"
        "Respond in JSON: {{\"score\": 0.0, \"reasoning\": \"...\"}}"
    ),
    "query_rewrite": (
        "Rewrite the following question to be more specific and searchable. "
        "Return only the rewritten question.\n\n"
        "Original: {question}\nRewritten:"
    ),
    "safety_refusal": (
        "I'm unable to assist with that request as it may violate content safety guidelines. "
        "Please rephrase your question or ask about a different topic.\n\n"
        "Flagged categories: {categories}"
    ),
    "rag": (
        "You are a helpful assistant. Use ONLY the context below to answer.\n"
        "If the answer is not in the context, say 'I don't have enough information.'\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\nAnswer:"
    ),
}

# Merge all templates
_ALL_TEMPLATES = {**_TEMPLATES, **_EXTRA_TEMPLATES}


# ── Personas ──────────────────────────────────────────────────────────────────

_PERSONAS: dict[str, dict] = {
    "default": {
        "name": "Assistant",
        "description": "A helpful, accurate, and concise assistant.",
        "guardrail_prefix": "",
    },
    "document_assistant": {
        "name": "Document Assistant",
        "description": "Answers questions strictly from provided documents.",
        "guardrail_prefix": "SAFETY RULES: Do NOT invent information not in the context.\n\n",
    },
    "hr": {
        "name": "HR Assistant",
        "description": "An empathetic HR specialist focused on company policies.",
        "guardrail_prefix": (
            "IMPORTANT: Only discuss HR-related topics. "
            "Do not provide legal advice.\n\n"
        ),
    },
    "tech": {
        "name": "Tech Support",
        "description": "A technical expert providing clear, actionable solutions.",
        "guardrail_prefix": (
            "IMPORTANT: Base all answers on the provided documentation.\n\n"
        ),
    },
    "admin": {
        "name": "Admin Assistant",
        "description": "A comprehensive assistant with access to all knowledge bases.",
        "guardrail_prefix": "",
    },
}

# ── Format rules ──────────────────────────────────────────────────────────────

_FORMAT_RULES: dict[str, str] = {
    "concise":    "Be concise. Answer in 1-3 sentences.",
    "detailed":   "Provide a detailed, comprehensive answer.",
    "json_only":  "Respond ONLY with valid JSON. No other text.",
    "cite_source": "Always mention which document your answer comes from.",
}

# ── Guardrail rules ───────────────────────────────────────────────────────────

_GUARDRAIL_RULES: dict[str, str] = {
    "no_hallucination": "Do NOT invent information not in the context.",
    "cite_sources":     "Always cite the source document.",
    "admit_ignorance":  "If you don't know, say so clearly.",
    "no_pii":           "Do NOT include personal information in your answer.",
}


# ── PromptAssembler ───────────────────────────────────────────────────────────

@dataclass
class PromptAssembler:
    """
    Assembles prompts from named templates with variable substitution,
    persona injection, and guardrail prefixes.

    Global instance: prompts = PromptAssembler()

    Usage:
        from generation.prompts import prompts
        text = prompts.get("rag_simple").format(context="...", question="...")
        # or
        text = prompts.build("rag_simple", context="...", question="...")
    """

    persona: str = "default"
    extra_guardrails: str = ""

    # Expose for lesson 08
    personas: dict = field(default_factory=lambda: dict(_PERSONAS))
    format_rules: dict = field(default_factory=lambda: dict(_FORMAT_RULES))
    guardrail_rules: dict = field(default_factory=lambda: dict(_GUARDRAIL_RULES))

    # ── Core API ──────────────────────────────────────────────────────────────

    def get(self, template_name: str) -> str:
        """Return the raw template string by name."""
        if template_name not in _ALL_TEMPLATES:
            raise ValueError(
                f"Unknown template '{template_name}'. "
                f"Available: {list(_ALL_TEMPLATES.keys())}"
            )
        return _ALL_TEMPLATES[template_name]

    def build(self, template_name: str, **kwargs) -> str:
        """
        Build a formatted prompt from a named template.

        Args:
            template_name: Key in the template registry.
            **kwargs: Variables to substitute.

        Returns:
            Formatted prompt string.
        """
        template = self.get(template_name)
        persona_data = _PERSONAS.get(self.persona, _PERSONAS["default"])

        if "{guardrail_prefix}" in template:
            kwargs.setdefault(
                "guardrail_prefix",
                persona_data["guardrail_prefix"] + self.extra_guardrails,
            )

        kwargs.setdefault("company_name", "the company")

        try:
            return template.format(**kwargs)
        except KeyError as exc:
            raise ValueError(
                f"Template '{template_name}' requires variable {exc}. "
                f"Provided: {list(kwargs.keys())}"
            ) from exc

    # ── Introspection ─────────────────────────────────────────────────────────

    def list_prompts(self) -> list[str]:
        """Return all registered template names (spec-required method name)."""
        return list(_ALL_TEMPLATES.keys())

    def list_templates(self) -> list[str]:
        return list(_ALL_TEMPLATES.keys())

    def list_personas(self) -> list[str]:
        return list(_PERSONAS.keys())

    def get_template(self, name: str) -> str:
        return self.get(name)

    def preview(self, template_name: str) -> str:
        """Return the raw template for inspection."""
        return self.get(template_name)

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add_template(self, name: str, template: str) -> None:
        """Register a custom template at runtime."""
        _ALL_TEMPLATES[name] = template

    def set_persona(self, persona: str) -> None:
        if persona not in _PERSONAS:
            raise ValueError(
                f"Persona '{persona}' not found. Available: {list(_PERSONAS.keys())}"
            )
        self.persona = persona

    def customize_persona(self, persona: str, description: str) -> None:
        """Update a persona's description (used by lesson 08)."""
        if persona not in self.personas:
            self.personas[persona] = {"name": persona, "description": description, "guardrail_prefix": ""}
        else:
            self.personas[persona]["description"] = description
        _PERSONAS[persona] = self.personas[persona]


# ── Module-level singleton ────────────────────────────────────────────────────

prompts: PromptAssembler = PromptAssembler()
