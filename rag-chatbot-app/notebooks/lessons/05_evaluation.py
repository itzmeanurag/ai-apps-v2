"""
LESSON 05: LLM-as-Judge Evaluation
====================================
CONCEPT: Using an LLM to score the quality of another LLM's answers

WHAT THIS DOES:
  Evaluates RAG answers on two metrics:
    1. Relevance   — Does the answer address the question? (0.0–1.0)
    2. Groundedness — Is every claim supported by the retrieved documents? (0.0–1.0)
  If quality is below threshold, it refines the query and retries.

WHY THIS MATTERS:
  You can't manually review every answer in production.
  LLM-as-judge automates quality control at scale.
  Groundedness specifically catches hallucination — when the model
  invents facts not present in the retrieved documents.

BEFORE RUNNING:
  python notebooks/lessons/03_ingest_documents.py

RUN (from project root):
  python notebooks/lessons/05_evaluation.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import cfg


# ── Evaluation prompts ────────────────────────────────────────────────────────

RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an evaluation judge. Score whether the answer addresses the question.

Score 0.0–1.0:
  1.0 = Directly and completely answers the question
  0.7 = Mostly answers with minor gaps
  0.5 = Partially answers
  0.3 = Tangentially related
  0.0 = Completely off-topic

Respond ONLY with valid JSON: {{"score": 0.0, "reasoning": "brief explanation"}}"""),
    ("human", "Question: {question}\n\nAnswer: {answer}\n\nEvaluate relevance:"),
])

GROUNDEDNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an evaluation judge. Score whether the answer is supported by the context.

Score 0.0–1.0:
  1.0 = Every claim is directly supported by the context
  0.7 = Most claims supported, minor extrapolations
  0.5 = About half the claims are supported
  0.3 = Few claims are supported
  0.0 = Answer is entirely hallucinated (not in context)

Respond ONLY with valid JSON: {{"score": 0.0, "reasoning": "brief explanation"}}"""),
    ("human", "Context:\n{context}\n\nAnswer: {answer}\n\nEvaluate groundedness:"),
])

REFINE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a query optimization expert. Rewrite the query to be more specific. Return ONLY the improved query."),
    ("human", "Original query: {query}\nProblem: {reasoning}\n\nImproved query:"),
])

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Answer the question based ONLY on the provided context. Cite your sources.\n\nContext:\n{context}"),
    ("human", "{question}"),
])


# ── Evaluation functions ──────────────────────────────────────────────────────

def parse_json_score(text: str, fallback: float = 0.5) -> dict:
    """Extract JSON score from LLM output, with fallback."""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            return {
                "score": float(data.get("score", fallback)),
                "reasoning": data.get("reasoning", ""),
            }
    except (json.JSONDecodeError, ValueError):
        pass
    return {"score": fallback, "reasoning": f"Parse error: {text[:80]}"}


def evaluate_relevance(llm, question: str, answer: str) -> dict:
    """Score how well the answer addresses the question."""
    chain = RELEVANCE_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"question": question, "answer": answer})
    return parse_json_score(result)


def evaluate_groundedness(llm, answer: str, context: str) -> dict:
    """Score how well the answer is supported by the retrieved documents."""
    chain = GROUNDEDNESS_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"context": context[:3000], "answer": answer})
    return parse_json_score(result)


def refine_query(llm, original_query: str, reasoning: str) -> str:
    """Generate a better query when quality is low."""
    chain = REFINE_PROMPT | llm | StrOutputParser()
    return chain.invoke({"query": original_query, "reasoning": reasoning}).strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("LESSON 05: LLM-AS-JUDGE EVALUATION")
    print("=" * 60)

    quality_threshold = cfg.evaluation.quality_threshold  # 0.6
    max_retries = cfg.evaluation.max_retries              # 2

    print(f"\nQuality threshold : {quality_threshold}")
    print(f"Max retries       : {max_retries}")

    # Load components
    print("\nLoading models and vector store...")
    embeddings = OllamaEmbeddings(model=cfg.models.embedder, base_url=cfg.models.ollama_base_url)
    vector_store = Chroma(
        persist_directory=cfg.ingestion.persist_directory,
        embedding_function=embeddings,
        collection_name=cfg.ingestion.collection_name,
    )
    if vector_store._collection.count() == 0:
        print("ERROR: No documents. Run 03_ingest_documents.py first.")
        return

    retriever = vector_store.as_retriever(search_kwargs={"k": cfg.retrieval.top_k})
    generator = ChatOllama(model=cfg.models.generator, base_url=cfg.models.ollama_base_url, temperature=0.3)
    evaluator = ChatOllama(model=cfg.models.generator, base_url=cfg.models.ollama_base_url, temperature=0.1)
    # WHY lower temperature for evaluator?
    # Evaluation needs consistency — the same answer should always get the same score.
    # Lower temperature = more deterministic = more consistent scoring.

    rag_chain = RAG_PROMPT | generator | StrOutputParser()

    # Test questions — last one is intentionally unanswerable
    test_questions = [
        "How many days of annual leave do employees get?",
        "What is the API rate limit?",
        "Can employees work from home?",
        "What is the company's policy on quantum computing?",  # Not in docs!
    ]

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    for question in test_questions:
        print(f"\n{'─' * 70}")
        print(f"QUESTION: {question}")

        current_query = question
        attempt = 0

        while attempt <= max_retries:
            attempt += 1
            label = f"[Attempt {attempt}] " if attempt > 1 else ""

            # Retrieve
            docs = retriever.invoke(current_query)
            context = "\n\n".join(
                f"[{Path(d.metadata.get('source','?')).name}]: {d.page_content}"
                for d in docs
            )

            # Generate
            answer = rag_chain.invoke({"context": context, "question": current_query})
            print(f"\n{label}ANSWER: {answer[:200]}{'...' if len(answer) > 200 else ''}")

            # Evaluate
            relevance    = evaluate_relevance(evaluator, question, answer)
            groundedness = evaluate_groundedness(evaluator, answer, context)
            avg = (relevance["score"] + groundedness["score"]) / 2

            print(f"{label}RELEVANCE:    {relevance['score']:.2f} — {relevance['reasoning']}")
            print(f"{label}GROUNDEDNESS: {groundedness['score']:.2f} — {groundedness['reasoning']}")
            print(f"{label}AVERAGE:      {avg:.2f}  (threshold: {quality_threshold})")

            if avg >= quality_threshold:
                print(f"{label}✅ PASSED")
                break
            elif attempt <= max_retries:
                print(f"{label}❌ BELOW THRESHOLD — refining query...")
                current_query = refine_query(evaluator, current_query, relevance["reasoning"])
                print(f"{label}Refined query: \"{current_query}\"")
            else:
                print(f"{label}❌ FAILED after {max_retries} retries")

    print(f"\n{'=' * 70}")
    print("KEY TAKEAWAY:")
    print("  Relevance   = 'Did we answer the right question?'")
    print("  Groundedness = 'Did we make up anything?'")
    print("  Both must be high for a trustworthy RAG system.")
    print("=" * 70)


if __name__ == "__main__":
    main()
