"""
src/chatbot.py
RAGChatbot – main orchestrator.

Public API (spec-required):
  ingest_documents(directory)
  ask(question, session_id, persona)
  ask_stream(question, session_id, persona)  -> Iterator[str]
  clean_query(query) -> str
  evaluate_answer(question, answer, context) -> dict
  _retrieve_with_confidence(query) -> tuple[list, float]
  _init_vectorstore() -> Chroma
  _init_hybrid_retriever() -> HybridRetriever
"""
from __future__ import annotations

import re
import sys
import unicodedata
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import cfg
from guardrails.content_safety import Guardrails, GuardrailConfig
from guardrails.model_governance import validate_and_sanitize_input
from retrieval.hybrid import HybridRetriever, Document
from generation.prompts import prompts as _prompts
from memory.memory_bank import MemoryBank
from evaluation.feedback import FeedbackCollector
from evaluation.rag_monitor import RAGQualityMonitor, SemanticCache


class RAGChatbot:
    """
    Full RAG pipeline: ingest → retrieve → generate → evaluate.

    Supports terminal mode (ask / ask_stream) and Gradio web mode.
    """

    def __init__(self) -> None:
        self._cfg = cfg
        self._llm = self._init_llm()
        self._embeddings = self._init_embeddings()
        self._vector_store = self._init_vectorstore()
        self._retriever = self._init_hybrid_retriever()
        self._guardrails = Guardrails(GuardrailConfig())
        self._memory = MemoryBank()
        self._feedback = FeedbackCollector()
        self._monitor = RAGQualityMonitor()
        self._cache = SemanticCache(embedder=self._embeddings)
        print("[chatbot] Initialized.")

    # ──────────────────────────────────────────────────────────────────────────
    # Spec-required init helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _init_vectorstore(self) -> Any:
        """Initialize and return the ChromaDB vector store."""
        from langchain_chroma import Chroma  # type: ignore
        persist_dir = self._cfg.ingestion.persist_directory
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name=self._cfg.ingestion.collection_name,
            embedding_function=self._embeddings,
            persist_directory=persist_dir,
        )

    def _init_hybrid_retriever(self) -> HybridRetriever:
        """Initialize and return the HybridRetriever (BM25 + vector + RRF)."""
        return HybridRetriever(
            vector_store=self._vector_store,
            cfg=self._cfg,
        )

    def _init_llm(self) -> Any:
        try:
            from langchain_ollama import ChatOllama  # type: ignore
            return ChatOllama(
                model=self._cfg.models.generator,
                base_url=self._cfg.models.ollama_base_url,
                temperature=self._cfg.models.temperature,
                num_predict=self._cfg.models.max_tokens,
            )
        except ImportError:
            from langchain_community.llms import Ollama  # type: ignore
            return Ollama(
                model=self._cfg.models.generator,
                base_url=self._cfg.models.ollama_base_url,
                temperature=self._cfg.models.temperature,
            )

    def _init_embeddings(self) -> Any:
        try:
            from langchain_ollama import OllamaEmbeddings  # type: ignore
            return OllamaEmbeddings(
                model=self._cfg.models.embedder,
                base_url=self._cfg.models.ollama_base_url,
            )
        except ImportError:
            from langchain_community.embeddings import OllamaEmbeddings  # type: ignore
            return OllamaEmbeddings(
                model=self._cfg.models.embedder,
                base_url=self._cfg.models.ollama_base_url,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Spec-required public methods
    # ──────────────────────────────────────────────────────────────────────────

    def ingest_documents(self, directory: str = "./data/documents") -> int:
        """
        Load, chunk, embed, and index documents from *directory*.
        Returns the number of chunks indexed.
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import DirectoryLoader, TextLoader

        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        loader = DirectoryLoader(
            str(dir_path),
            glob="**/*",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            silent_errors=True,
        )
        raw_docs = loader.load()
        if not raw_docs:
            print(f"[ingest] No documents found in {directory}")
            return 0

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._cfg.ingestion.chunk_size,
            chunk_overlap=self._cfg.ingestion.chunk_overlap,
        )
        chunks = splitter.split_documents(raw_docs)

        for i, chunk in enumerate(chunks):
            chunk.metadata["id"] = f"chunk-{i:06d}"
            chunk.metadata.setdefault("source", "unknown")

        self._vector_store.add_documents(chunks)

        bm25_docs = [
            Document(id=c.metadata["id"], content=c.page_content, metadata=c.metadata)
            for c in chunks
        ]
        self._retriever.build_bm25_index(bm25_docs)

        print(f"[ingest] Indexed {len(chunks)} chunks from {len(raw_docs)} documents.")
        return len(chunks)

    def clean_query(self, query: str) -> str:
        """
        Sanitize and normalize a user query:
        - Remove null bytes and control characters
        - NFKC Unicode normalization
        - Collapse whitespace
        - Strip leading/trailing punctuation noise
        """
        result = validate_and_sanitize_input(query)
        if not result["valid"]:
            return ""
        text = result["sanitized"]
        # Collapse whitespace
        text = " ".join(text.split())
        # Strip leading/trailing punctuation noise (keep ? ! .)
        text = text.strip("\"'`")
        return text

    def _retrieve_with_confidence(
        self, query: str, top_k: Optional[int] = None
    ) -> tuple[list[Document], float]:
        """
        Retrieve documents and return (docs, confidence_score).
        Confidence is the average score of the top results (0–1).
        Returns ([], 0.0) when nothing is found.
        """
        docs = self._retriever.search(query, top_k=top_k)
        if not docs:
            return [], 0.0
        # Normalise scores: CrossEncoder scores can be negative; clamp to [0,1]
        scores = [max(0.0, min(1.0, d.score)) for d in docs]
        confidence = sum(scores) / len(scores)
        return docs, confidence

    def evaluate_answer(self, question: str, answer: str, context: str) -> dict:
        """
        LLM-as-judge evaluation.
        Returns dict with keys: relevance, groundedness, overall, passed.
        """
        if not self._cfg.evaluation.enable_auto_eval:
            return {"relevance": 1.0, "groundedness": 1.0, "overall": 1.0, "passed": True}

        import json as _json

        eval_prompt = _prompts.get("eval_combined").format(
            question=question,
            answer=answer,
            context=context[:3000],
        )
        try:
            response = self._llm.invoke(eval_prompt)
            text = response.content if hasattr(response, "content") else str(response)
            start, end = text.find("{"), text.rfind("}") + 1
            if start >= 0 and end > start:
                data = _json.loads(text[start:end])
                relevance = float(data.get("relevance", 0.5))
                groundedness = float(data.get("groundedness", 0.5))
                overall = (relevance + groundedness) / 2
                return {
                    "relevance": relevance,
                    "groundedness": groundedness,
                    "overall": overall,
                    "passed": overall >= self._cfg.evaluation.quality_threshold,
                    "explanation": data.get("explanation", ""),
                }
        except Exception as exc:
            print(f"[eval] Evaluation failed: {exc}")

        return {"relevance": 0.5, "groundedness": 0.5, "overall": 0.5, "passed": True}

    def ask(
        self,
        question: str,
        session_id: Optional[str] = None,
        persona: str = "default",
    ) -> dict:
        """
        Full RAG pipeline (non-streaming).
        Returns dict: answer, session_id, sources, quality, cached.
        """
        session_id = session_id or str(uuid.uuid4())

        # 1. Clean + guardrail check
        clean = self.clean_query(question)
        guard = self._guardrails.check_input(clean)
        if not guard["safe"]:
            return {
                "answer": f"I can't help with that. {guard.get('message', '')}",
                "session_id": session_id,
                "sources": [],
                "cached": False,
                "blocked": True,
            }
        effective = guard.get("cleaned_text") or clean

        # 2. Semantic cache
        cached_entry = self._cache.lookup(effective)
        if cached_entry:
            self._memory.add_exchange(session_id, question, cached_entry.answer)
            return {
                "answer": cached_entry.answer,
                "session_id": session_id,
                "sources": [],
                "cached": True,
            }

        # 3. Retrieve with confidence
        docs, confidence = self._retrieve_with_confidence(effective)
        if confidence < self._cfg.retrieval.similarity_threshold and docs:
            # Low confidence – try once with rewritten query
            rewritten = self._rewrite_query(effective)
            docs2, conf2 = self._retrieve_with_confidence(rewritten)
            if conf2 > confidence:
                docs, confidence = docs2, conf2

        context = "\n\n---\n\n".join(d.content for d in docs) if docs else ""

        # 4. Build prompt
        mem_ctx = self._memory.get_context(session_id)
        if mem_ctx.get("history"):
            prompt = _prompts.get("rag_with_history").format(
                context=context or "No relevant documents found.",
                question=effective,
                history=mem_ctx["history"],
                summary=mem_ctx.get("summary", ""),
                facts=mem_ctx.get("facts", ""),
            )
        else:
            prompt = _prompts.get("rag_simple").format(
                context=context or "No relevant documents found.",
                question=effective,
            )

        # 5. Generate
        response = self._llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        # 6. Output guardrail
        out_guard = self._guardrails.check_output(answer)
        if not out_guard["safe"]:
            answer = out_guard.get("cleaned_text") or answer

        # 7. Evaluate
        quality = None
        if context:
            quality = self.evaluate_answer(effective, answer, context)
            if not quality["passed"]:
                answer = (
                    "I wasn't able to find a reliable answer. "
                    "Please try rephrasing your question."
                )

        # 8. Memory + cache
        self._memory.add_exchange(session_id, question, answer)
        self._cache.store(effective, answer, context)

        sources = [
            {"id": d.id, "source": d.metadata.get("source", ""), "score": round(d.score, 4)}
            for d in docs
        ]
        return {
            "answer": answer,
            "session_id": session_id,
            "sources": sources,
            "quality": quality,
            "cached": False,
            "confidence": round(confidence, 3),
        }

    def ask_stream(
        self,
        question: str,
        session_id: Optional[str] = None,
        persona: str = "default",
    ) -> Iterator[str]:
        """
        Streaming RAG pipeline. Yields answer tokens one by one.
        """
        session_id = session_id or str(uuid.uuid4())

        clean = self.clean_query(question)
        guard = self._guardrails.check_input(clean)
        if not guard["safe"]:
            yield f"I can't help with that. {guard.get('message', '')}"
            return

        effective = guard.get("cleaned_text") or clean
        docs, _ = self._retrieve_with_confidence(effective)
        context = "\n\n---\n\n".join(d.content for d in docs) if docs else ""

        prompt = _prompts.get("rag_simple").format(
            context=context or "No relevant documents found.",
            question=effective,
        )

        full_answer = ""
        for chunk in self._llm.stream(prompt):
            text = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_answer += text
            yield text

        self._memory.add_exchange(session_id, question, full_answer)
        self._cache.store(effective, full_answer, context)

    async def ask_stream_async(
        self,
        question: str,
        session_id: Optional[str] = None,
        persona: str = "default",
    ) -> AsyncIterator[str]:
        """Async wrapper around ask_stream for FastAPI StreamingResponse."""
        for chunk in self.ask_stream(question, session_id=session_id, persona=persona):
            yield chunk

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _rewrite_query(self, query: str) -> str:
        """Use LLM to rewrite a query for better retrieval."""
        try:
            prompt = _prompts.get("refine_query").format(query=query)
            response = self._llm.invoke(prompt)
            return (response.content if hasattr(response, "content") else str(response)).strip()
        except Exception:
            return query

    # ──────────────────────────────────────────────────────────────────────────
    # Gradio web mode
    # ──────────────────────────────────────────────────────────────────────────

    def launch_gradio(self, share: bool = False) -> None:
        """Launch the Gradio web interface."""
        import gradio as gr  # type: ignore

        def chat_fn(message: str, history: list, session_id: str) -> str:
            result = self.ask(message, session_id=session_id or "gradio-default")
            return result["answer"]

        with gr.Blocks(title="RAG Chatbot") as demo:
            gr.Markdown("# RAG Chatbot\nPowered by Mistral + ChromaDB + LangChain")
            session_box = gr.Textbox(value=str(uuid.uuid4()), label="Session ID", interactive=True)
            chatbot_ui = gr.ChatInterface(
                fn=lambda msg, hist: chat_fn(msg, hist, session_box.value),
                title="",
            )
        demo.launch(server_name="0.0.0.0", server_port=7860, share=share)

    # ──────────────────────────────────────────────────────────────────────────
    # Convenience accessors (used by API server)
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def feedback(self) -> FeedbackCollector:
        return self._feedback

    @property
    def memory(self) -> MemoryBank:
        return self._memory
