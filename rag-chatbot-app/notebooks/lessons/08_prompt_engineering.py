"""
LESSON 08: Prompt Engineering — Centralized Prompt Management
=============================================================
CONCEPT: One place to manage all prompts; change once, update everywhere

WHAT THIS DOES:
  Demonstrates the PromptAssembler from generation/prompts.py:
    - 12 named templates (rag, eval, memory, refinement, etc.)
    - 4 personas (default, hr, tech, admin)
    - Guardrail prefix injection
    - Runtime template customization

WHY THIS MATTERS:
  Without a central prompt manager, prompts are scattered across files.
  Change the safety rules? Edit 8 files. Add a new persona? Edit 5 files.
  With PromptAssembler, change one thing → all prompts update automatically.

RUN (from project root):
  python notebooks/lessons/08_prompt_engineering.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from the actual project module (generation/prompts.py)
from generation.prompts import PromptAssembler, TEMPLATES, PERSONAS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import cfg


# ── Demo 1: List all templates and personas ───────────────────────────────────

def demo_available_templates() -> None:
    """Show all registered templates and personas."""
    print("\n--- DEMO 1: Available Templates and Personas ---")

    assembler = PromptAssembler()

    print(f"\n  {len(TEMPLATES)} registered templates:")
    for name in TEMPLATES:
        # Show first 60 chars of each template
        preview = TEMPLATES[name][:60].replace("\n", " ")
        print(f"    {name:<20} → \"{preview}...\"")

    print(f"\n  {len(PERSONAS)} personas:")
    for name, data in PERSONAS.items():
        prefix_preview = data["guardrail_prefix"][:50].replace("\n", " ") or "(none)"
        print(f"    {name:<10} → guardrail: \"{prefix_preview}\"")


# ── Demo 2: Build prompts from templates ──────────────────────────────────────

def demo_build_prompts() -> None:
    """Show how prompts are assembled from templates."""
    print("\n--- DEMO 2: Building Prompts from Templates ---")

    assembler = PromptAssembler(persona="default")

    # Build a RAG prompt
    rag_prompt = assembler.build(
        "rag",
        context="Employees receive 20 days of annual leave per year.",
        question="How many vacation days do I get?",
    )
    print("\n  RAG prompt (default persona):")
    print("  " + "\n  ".join(rag_prompt.split("\n")))

    # Build the same prompt with HR persona
    assembler.set_persona("hr")
    hr_prompt = assembler.build(
        "rag",
        context="Employees receive 20 days of annual leave per year.",
        question="How many vacation days do I get?",
    )
    print("\n  RAG prompt (hr persona — notice the guardrail prefix):")
    print("  " + "\n  ".join(hr_prompt.split("\n")))

    print("""
  KEY INSIGHT:
    The only difference is the persona. The guardrail prefix is injected
    automatically. Change the persona → different safety instructions.
    Change the persona definition → all prompts using it update.
""")


# ── Demo 3: Persona comparison ────────────────────────────────────────────────

def demo_persona_comparison() -> None:
    """Show how different personas produce different answers."""
    print("\n--- DEMO 3: Persona Comparison ---")
    print("  Same question, same context, different persona → different answer style.\n")

    llm = ChatOllama(
        model=cfg.models.generator,
        base_url=cfg.models.ollama_base_url,
        temperature=0.5,
    )

    context = "Employees receive 20 days of annual leave per year. Leave must be approved by manager."
    question = "How do I request time off?"

    for persona_name in ["default", "hr", "tech"]:
        assembler = PromptAssembler(persona=persona_name)
        prompt_text = assembler.build("rag", context=context, question=question)

        # Use LangChain to send the assembled prompt
        lc_prompt = ChatPromptTemplate.from_messages([("human", "{prompt}")])
        chain = lc_prompt | llm | StrOutputParser()
        answer = chain.invoke({"prompt": prompt_text})

        print(f"  [{persona_name.upper()} PERSONA]")
        print(f"  {answer[:150]}...")
        print()


# ── Demo 4: Custom template ───────────────────────────────────────────────────

def demo_custom_template() -> None:
    """Show how to add a custom template at runtime."""
    print("\n--- DEMO 4: Adding a Custom Template ---")

    assembler = PromptAssembler()

    # Add a custom template for a specific use case
    assembler.add_template(
        "onboarding",
        "{guardrail_prefix}"
        "You are an onboarding assistant for new employees at {company_name}.\n"
        "Help them understand company policies in a friendly, welcoming way.\n\n"
        "Policy information:\n{context}\n\n"
        "New employee question: {question}\n\n"
        "Friendly answer:"
    )

    prompt = assembler.build(
        "onboarding",
        context="Employees get 20 days leave. Remote work allowed 3 days/week.",
        question="What benefits do I get as a new employee?",
        company_name="Acme Corp",
    )

    print(f"\n  Custom 'onboarding' template built successfully.")
    print(f"  Prompt preview (first 200 chars):")
    print(f"  \"{prompt[:200]}...\"")
    print(f"\n  Total templates now: {len(assembler.list_templates())}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("LESSON 08: PROMPT ENGINEERING")
    print("=" * 60)
    print("""
PromptAssembler (generation/prompts.py):
  Final Prompt = Persona guardrail + Template + Your variables

  assembler = PromptAssembler(persona="hr")
  prompt = assembler.build("rag", context="...", question="...")

Change persona → different safety prefix injected automatically.
Change template → different task instructions.
""")

    demo_available_templates()
    demo_build_prompts()
    demo_persona_comparison()
    demo_custom_template()


if __name__ == "__main__":
    main()
