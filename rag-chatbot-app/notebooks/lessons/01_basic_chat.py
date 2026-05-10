"""
LESSON 01: Basic Chat with Ollama via LangChain
================================================
CONCEPT: The LangChain chain pattern — prompt | llm | parser

WHAT THIS DOES:
  Connects to Ollama running locally and sends questions to Mistral.
  This is the simplest possible AI interaction and the foundation
  for everything else in this project.

WHY THIS MATTERS:
  Every RAG pipeline, every evaluation, every guardrail check is
  built on this same three-step chain. Master this and you understand
  the skeleton of the entire system.

BEFORE RUNNING:
  1. Install Ollama: https://ollama.ai
  2. Pull the model:  ollama pull mistral
  3. Activate venv:   source .venv/bin/activate  (Linux/Mac)
                      .venv\\Scripts\\activate     (Windows)
  4. Install deps:    pip install -r requirements.txt

RUN (from project root):
  python notebooks/lessons/01_basic_chat.py
"""

# ── Imports ───────────────────────────────────────────────────────────────────
# ChatOllama: bridges LangChain to Ollama's local model server.
# It does NOT download anything — Ollama must already be running.
from langchain_ollama import ChatOllama

# ChatPromptTemplate: a reusable prompt format with named variables.
# {question} is a placeholder filled at runtime.
from langchain_core.prompts import ChatPromptTemplate

# StrOutputParser: extracts the plain text string from the LLM's response object.
# Without it, you'd get an AIMessage object instead of a string.
from langchain_core.output_parsers import StrOutputParser


def demo_single_question() -> None:
    """Show the simplest possible chain: one question, one answer."""
    print("\n--- DEMO 1: Single Question ---")

    # Step 1: Create the LLM connection.
    # temperature=0.7 → balanced between creative and deterministic.
    # 0.0 = always the same answer; 1.0 = very random.
    llm = ChatOllama(model="mistral", temperature=0.7)

    # Step 2: Define a prompt template.
    # "system" message tells the AI HOW to behave.
    # "human" message is the actual question.
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer clearly and concisely."),
        ("human", "{question}"),
    ])

    # Step 3: Build the chain using the pipe (|) operator.
    # Data flows LEFT → RIGHT:
    #   prompt formats the dict → llm generates a response → parser extracts text
    # This is THE core LangChain pattern. Everything in this project uses it.
    chain = prompt | llm | StrOutputParser()

    # Step 4: Invoke the chain.
    # .invoke() sends the input through every step and returns the final output.
    response = chain.invoke({"question": "What is RAG in AI? Answer in two sentences."})
    print(f"Answer: {response}")


def demo_multiple_questions() -> None:
    """Show how the same chain handles different questions."""
    print("\n--- DEMO 2: Multiple Questions ---")

    llm = ChatOllama(model="mistral", temperature=0.3)
    # Lower temperature (0.3) for factual questions → more consistent answers.

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise assistant. Give short, direct answers."),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()

    questions = [
        "What does LLM stand for?",
        "Name three vector databases.",
        "What is the difference between RAG and fine-tuning in one sentence?",
    ]

    for q in questions:
        answer = chain.invoke({"question": q})
        print(f"\nQ: {q}")
        print(f"A: {answer}")


def demo_interactive_chat() -> None:
    """Interactive chat loop — type questions, get answers."""
    print("\n--- DEMO 3: Interactive Chat ---")
    print("Type 'quit' to exit.\n")

    llm = ChatOllama(model="mistral", temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer clearly and concisely."),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not question:
            continue
        print("AI: ", end="", flush=True)
        # WHY print with end=""? So the answer appears on the same line as "AI: "
        response = chain.invoke({"question": question})
        print(response)


def main() -> None:
    print("=" * 60)
    print("LESSON 01: BASIC CHAT WITH OLLAMA + LANGCHAIN")
    print("=" * 60)
    print("""
KEY CONCEPT: The LangChain chain pattern
  chain = prompt | llm | output_parser
  result = chain.invoke({"variable": "value"})

The | operator connects steps. Data flows left to right.
This same pattern is used in EVERY lesson that follows.
""")

    demo_single_question()
    demo_multiple_questions()
    demo_interactive_chat()


if __name__ == "__main__":
    main()
