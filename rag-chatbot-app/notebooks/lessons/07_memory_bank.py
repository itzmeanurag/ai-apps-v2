"""
LESSON 07: Memory Bank — Persistent Conversation Memory
========================================================
CONCEPT: 3-layer memory that survives restarts

WHAT THIS DOES:
  Demonstrates the MemoryBank from memory/memory_bank.py:
    Layer 1 — Buffer:  last N messages in full detail
    Layer 2 — Summary: LLM-compressed older messages
    Layer 3 — Facts:   extracted key facts (names, decisions, preferences)
  Sessions are saved as JSON files and survive process restarts.

WHY THIS MATTERS:
  Without memory, every question is independent.
  "How many exactly?" fails because the bot forgot you asked about leave.
  With memory, the bot maintains context across the entire conversation.

RUN (from project root):
  python notebooks/lessons/07_memory_bank.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import from the actual project module (memory/memory_bank.py)
from memory.memory_bank import MemoryBank
from src.config import cfg


# ── Demo 1: Basic memory operations ──────────────────────────────────────────

def demo_memory_layers() -> None:
    """Show the three memory layers and how they work."""
    print("\n--- DEMO 1: Three Memory Layers ---")

    # Create a MemoryBank (saves sessions to data/sessions/)
    bank = MemoryBank(
        session_dir=cfg.memory.session_dir,
        buffer_size=cfg.memory.buffer_size,
        summary_threshold=cfg.memory.summary_threshold,
        facts_max=cfg.memory.facts_max,
        enable_summarization=False,  # disable LLM calls for this demo
    )

    session_id = "lesson07-demo"

    # Add some messages to the buffer
    print("\n  Adding messages to buffer...")
    bank.add_message(session_id, "user", "How many leave days do I get?")
    bank.add_message(session_id, "assistant", "Full-time employees get 20 days of annual leave per year.")
    bank.add_message(session_id, "user", "Can I carry them over?")
    bank.add_message(session_id, "assistant", "Yes, up to 10 days can be carried over to the next year.")
    bank.add_message(session_id, "user", "What about sick leave?")
    bank.add_message(session_id, "assistant", "Sick leave is 10 days per year, separate from annual leave.")

    # Show what's in each layer
    buffer = bank.get_buffer(session_id)
    summary = bank.get_summary(session_id)
    facts = bank.get_facts(session_id)

    print(f"\n  LAYER 1 — Buffer ({len(buffer)} messages):")
    for msg in buffer:
        role = "User" if msg.role == "user" else "Bot "
        print(f"    {role}: {msg.content[:70]}")

    print(f"\n  LAYER 2 — Summary: {'(empty — no LLM summarization in this demo)' if not summary else summary[:100]}")

    print(f"\n  LAYER 3 — Facts ({len(facts)} items):")
    if facts:
        for f in facts:
            print(f"    - {f}")
    else:
        print("    (empty — facts are extracted by LLM during summarization)")

    # Show the context dict that gets injected into prompts
    ctx = bank.get_context(session_id)
    print(f"\n  Context injected into prompt:")
    print(f"    summary: \"{ctx['summary'][:60] or 'empty'}\"")
    print(f"    facts:   \"{ctx['facts'][:60]}\"")
    print(f"    history: \"{ctx['history'][:80]}...\"")

    # Clean up demo session
    bank.delete_session(session_id)


# ── Demo 2: Persistence across restarts ──────────────────────────────────────

def demo_persistence() -> None:
    """Show that sessions survive process restarts."""
    print("\n--- DEMO 2: Persistence Across Restarts ---")

    bank = MemoryBank(
        session_dir=cfg.memory.session_dir,
        buffer_size=10,
        enable_summarization=False,
    )
    session_id = "lesson07-persist-test"

    # Check if session already exists from a previous run
    existing_sessions = bank.list_sessions()
    if session_id in existing_sessions:
        print(f"\n  Found existing session '{session_id}' from a previous run!")
        state = bank.load_session(session_id)
        print(f"  Buffer has {len(state.buffer)} messages from before.")
        print(f"  Created at: {state.created_at}")
        bank.delete_session(session_id)
        print("  (Deleted for clean demo)")
    else:
        print(f"\n  No existing session found. Creating '{session_id}'...")
        bank.add_message(session_id, "user", "Remember: my name is Alex")
        bank.add_message(session_id, "assistant", "Got it, Alex! How can I help?")
        bank.save_session(session_id)
        print(f"  Session saved to {cfg.memory.session_dir}/{session_id}.json")
        print("  Run this script again — the session will still be there!")


# ── Demo 3: Memory-aware chat ─────────────────────────────────────────────────

def demo_memory_chat() -> None:
    """Interactive chat that uses memory for context."""
    print("\n--- DEMO 3: Memory-Aware Chat ---")
    print("  The bot remembers what you said earlier in the conversation.")
    print("  Commands: 'memory' to see current memory, 'quit' to exit\n")

    bank = MemoryBank(
        session_dir=cfg.memory.session_dir,
        buffer_size=cfg.memory.buffer_size,
        enable_summarization=False,  # set True to enable LLM summarization
    )
    llm = ChatOllama(
        model=cfg.models.generator,
        base_url=cfg.models.ollama_base_url,
        temperature=0.3,
    )
    session_id = "lesson07-chat"

    # WHY inject memory into the prompt?
    # The LLM has no state between calls. We must explicitly pass
    # the conversation history in the prompt every time.
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Use the conversation history below.\n\n"
         "CONVERSATION HISTORY:\n{history}\n\n"
         "KNOWN FACTS:\n{facts}"),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()

    while True:
        user_input = input("\n  You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            bank.delete_session(session_id)
            print("  Session cleared. Goodbye!")
            break

        if user_input.lower() == "memory":
            ctx = bank.get_context(session_id)
            print(f"\n  --- Current Memory ---")
            print(f"  History: {ctx['history'][:200] or '(empty)'}")
            print(f"  Facts:   {ctx['facts'][:100] or '(none)'}")
            print(f"  Summary: {ctx['summary'][:100] or '(none)'}")
            continue

        ctx = bank.get_context(session_id)
        print("\n  AI: ", end="", flush=True)
        response = chain.invoke({
            "history": ctx["history"] or "No prior conversation.",
            "facts": ctx["facts"] or "None.",
            "question": user_input,
        })
        print(response)

        # Save this exchange to memory
        bank.add_message(session_id, "user", user_input)
        bank.add_message(session_id, "assistant", response)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("LESSON 07: MEMORY BANK")
    print("=" * 60)
    print(f"""
Three-layer persistent memory (memory/memory_bank.py):
  Layer 1 — Buffer:  last {cfg.memory.buffer_size} messages (fast, in-memory)
  Layer 2 — Summary: LLM-compressed older messages (compressed)
  Layer 3 — Facts:   extracted key facts (structured)

Sessions saved to: {cfg.memory.session_dir}/
""")

    demo_memory_layers()
    demo_persistence()
    demo_memory_chat()


if __name__ == "__main__":
    main()
