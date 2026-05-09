"""
src/chatbot.py
Main RAG orchestrator: ingest, retrieve, generate, evaluate, stream.
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Optional

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import cfg
from guardrails.content_safety import ContentSafetyFilter
from guardrails.model_governance import ModelGovernance
from retrieval.hybrid import Document, HybridRetriever
from generation.prompts import PromptAssembler
from memory.memory_bank import MemoryBank
from evaluation.feedback import FeedbackStore
from evaluation.rag_monitor import RAGMonitor


class RAGChatbot:
    """
    Full RAG pipeline orchestrator.

    Capabilities:
    - Document ingestion with chunking
    - Hybrid retrieval (BM25 + vector + RRF + CrossEncoder)
    - Memory-aware generation (3-layer memory)
    - Content safety guardrails (input + output)
    - Model governance (checksums, sanitization)
    - Quality evaluation and semantic caching
    - Streaming generation
    - Human feedback collection
    """

    def __init__(self) -> None:
        self._cfg = cfg
        self._setup_components()

    # ── Initialization ────────────────────────────────────────────────────────

    def _setup_components(self) -> None:
        """Initialize all pipeline components."""
        # LLM and embeddings (Ollama via LangChain)
        self._llm = self._init_llm()
        self._embeddings = self._init_embeddings()

        # Vector store (ChromaDB)
        self._vector_store = self._init_vector_store()

        # Hybrid retriever
        self._retriever = HybridRetriever(
            vector_store=self._vector_store,
            cfg=self._cfg,
        )

        # Guardrails
        self._safety = ContentSafetyFilter(
            enable_llm_classification=self._cfg.guardrails.enable_llm_classification
        )
        self._governance = ModelGovernance(self._cfg.guardrails.checksum_file)

        # Memory
        self.memory = MemoryBank(
            session_dir=self._cfg.memory.session_dir,
            buffer_size=self._cfg.memory.buffer_size,
            summary_threshold=self._cfg.memory.summary_threshold,
            facts_max=self._cfg.memory.facts_max,
            llm=self._llm,
            enable_summarization=self._cfg.memory.enable_summarization,
        )

        # Prompt assembly
        self._prompt_assembler = PromptAssembler()

        # Evaluation
        self.feedback_store = FeedbackStore("./data/feedback.jsonl")
        self._monitor = RAGMonitor(
            llm=self._llm,
            embedder=self._embeddings,
            quality_threshold=self._cfg.evaluation.quality_threshold,
            truthfulness_threshold=self._cfg.evaluation.truthfulness_threshold,
            semantic_cache_threshold=self._cfg.evaluation.semantic_cache_threshold,
        )

        print("[chatbot] All components initialized.")

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
            try:
                from langchain_community.llms import Ollama  # type: ignore
                return Ollama(
                    model=self._cfg.models.generator,
                    base_url=self._cfg.models.ollama_base_url,
                    temperature=self._cfg.models.temperature,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "langchain-ollama or langchain-community is required. "
                    "Run: pip install langchain-ollama"
                ) from exc

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

    def _init_vector_store(self) -> Any:
        from langchain_chroma import Chroma  # type: ignore
        persist_dir = self._cfg.ingestion.persist_directory
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name=self._cfg.ingestion.collection_name,
            embedding_function=self._embeddings,
            persist_directory=persist_dir,
        )

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(self, directory: str = "./data/documents") -> int:
        """
        Load, chunk, and index documents from a directory.
        Returns the number of chunks ingested.
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import DirectoryLoader, TextLoader

        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Load documents
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

        # Chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._cfg.ingestion.chunk_size,
            chunk_overlap=self._cfg.ingestion.chunk_overlap,
        )
        chunks = splitter.split_documents(raw_docs)

        # Add unique IDs to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["id"] = f"chunk-{i:06d}"
            chunk.metadata["source"] = chunk.metadata.get("source", "unknown")

        # Index in vector store
        self._vector_store.add_documents(chunks)

        # Build BM25 index
        bm25_docs = [
            Document(
                id=c.metadata["id"],
                content=c.page_content,
                metadata=c.metadata,
            )
            for c in chunks
        ]
        self._retriever.build_bm25_index(bm25_docs)

        print(f"[ingest] Indexed {len(chunks)} chunks from {len(raw_docs)} documents.")
        return len(chunks)

    # ── Core ask pipeline ─────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        session_id: Optional[str] = None,
        persona: str = "default",
        extra_context: str = "",
    ) -> dict:
        """
        Full RAG pipeline:
        1. Sanitize + safety check input
        2. Check semantic cache
        3. Retrieve relevant documents
        4. Build memory-aware prompt
        5. Generate answer
        6. Safety check output
        7. Evaluate quality
        8. Update memory
        9. Return result
        """
        session_id = session_id or str(uuid.uuid4())

        # 1. Sanitize and safety check
        clean_question = self._governance.sanitize(question)
        safety_result = self._safety.check_input(clean_question)
        if not safety_result.is_safe:
            return {
                "answer": self._prompt_assembler.build(
                    "safety_refusal",
                    categories=", ".join(c.value for c in safety_result.flagged_categories),
                ),
                "session_id": session_id,
                "sources": [],
                "cached": False,
                "safety_blocked": True,
            }

        # Use anonymized text if PII was detected
        effective_question = safety_result.anonymized_text or clean_question

        # 2. Semantic cache lookup
        cached = self._monitor.cache.lookup(effective_question)
        if cached:
            self.memory.add_message(session_id, "user", question)
            self.memory.add_message(session_id, "assistant", cached.answer)
            return {
                "answer": cached.answer,
                "session_id": session_id,
                "sources": [],
                "cached": True,
            }

        # 3. Retrieve documents
        retrieved_docs = self._retriever.retrieve(effective_question)
        context_parts = [doc.content for doc in retrieved_docs]
        if extra_context:
            context_parts.append(extra_context)
        context = "\n\n---\n\n".join(context_parts)

        # 4. Build prompt with memory
        mem_ctx = self.memory.get_context(session_id)
        self._prompt_assembler.set_persona(persona)

        if mem_ctx["summary"] or mem_ctx["history"]:
            prompt = self._prompt_assembler.build(
                "memory",
                context=context or "No relevant documents found.",
                question=effective_question,
                summary=mem_ctx["summary"] or "No prior summary.",
                facts=mem_ctx["facts"],
                history=mem_ctx["history"] or "No prior conversation.",
            )
        else:
            prompt = self._prompt_assembler.build(
                "rag",
                context=context or "No relevant documents found.",
                question=effective_question,
            )

        # 5. Generate
        response = self._llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        # 6. Output safety check
        output_safety = self._safety.check_output(answer)
        if not output_safety.is_safe:
            answer = output_safety.anonymized_text or answer

        # 7. Evaluate quality
        quality_metrics = None
        if self._cfg.evaluation.enable_auto_eval and context:
            metrics = self._monitor.evaluate(effective_question, answer, context)
            self._monitor.record_metrics(metrics, effective_question)
            quality_metrics = metrics.to_dict()

            if self._monitor.should_degrade(metrics):
                answer = self._monitor.graceful_degradation_response(effective_question)

        # 8. Update memory
        self.memory.add_message(session_id, "user", question)
        self.memory.add_message(session_id, "assistant", answer)

        # 9. Cache the result
        self._monitor.cache.store(effective_question, answer, context)

        sources = [
            {"id": doc.id, "source": doc.metadata.get("source", ""), "score": round(doc.score, 4)}
            for doc in retrieved_docs
        ]

        return {
            "answer": answer,
            "session_id": session_id,
            "sources": sources,
            "quality": quality_metrics,
            "cached": False,
        }

    # ── Streaming ─────────────────────────────────────────────────────────────

    def stream(
        self,
        question: str,
        session_id: Optional[str] = None,
        persona: str = "default",
    ) -> Iterator[str]:
        """Synchronous streaming generator."""
        session_id = session_id or str(uuid.uuid4())
        clean_question = self._governance.sanitize(question)
        safety_result = self._safety.check_input(clean_question)

        if not safety_result.is_safe:
            yield self._prompt_assembler.build(
                "safety_refusal",
                categories=", ".join(c.value for c in safety_result.flagged_categories),
            )
            return

        effective_question = safety_result.anonymized_text or clean_question
        retrieved_docs = self._retriever.retrieve(effective_question)
        context = "\n\n---\n\n".join(doc.content for doc in retrieved_docs)

        self._prompt_assembler.set_persona(persona)
        prompt = self._prompt_assembler.build(
            "rag",
            context=context or "No relevant documents found.",
            question=effective_question,
        )

        full_answer = ""
        for chunk in self._llm.stream(prompt):
            text = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_answer += text
            yield text

        self.memory.add_message(session_id, "user", question)
        self.memory.add_message(session_id, "assistant", full_answer)
        self._monitor.cache.store(effective_question, full_answer, context)

    async def astream(
        self,
        question: str,
        session_id: Optional[str] = None,
        persona: str = "default",
    ) -> AsyncIterator[str]:
        """Async streaming generator for FastAPI StreamingResponse."""
        for chunk in self.stream(question, session_id=session_id, persona=persona):
            yield chunk

    # ── Utility ───────────────────────────────────────────────────────────────

    def rewrite_query(self, question: str) -> str:
        """Rewrite a query for better retrieval using the LLM."""
        prompt = self._prompt_assembler.build("query_rewrite", question=question)
        response = self._llm.invoke(prompt)
        return response.content.strip() if hasattr(response, "content") else str(response).strip()

    def get_cache_stats(self) -> dict:
        return self._monitor.cache.stats()

    def get_quality_stats(self) -> dict:
        return self._monitor.get_average_quality()
