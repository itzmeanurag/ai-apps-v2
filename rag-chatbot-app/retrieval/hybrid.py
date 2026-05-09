"""
retrieval/hybrid.py
BM25 (pure Python) + vector search + Reciprocal Rank Fusion + CrossEncoder re-ranking.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

# ── BM25 implementation (pure Python, no external deps) ──────────────────────

@dataclass
class Document:
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


class BM25:
    """
    Okapi BM25 retriever.
    k1=1.5, b=0.75 (standard defaults).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[Document] = []
        self._tf: list[dict[str, int]] = []
        self._df: dict[str, int] = defaultdict(int)
        self._avg_dl: float = 0.0
        self._N: int = 0

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def index(self, documents: list[Document]) -> None:
        """Build BM25 index from a list of Documents."""
        self._docs = documents
        self._tf = []
        self._df = defaultdict(int)
        total_len = 0

        for doc in documents:
            tokens = self._tokenize(doc.content)
            total_len += len(tokens)
            tf: dict[str, int] = defaultdict(int)
            for token in tokens:
                tf[token] += 1
            self._tf.append(dict(tf))
            for token in set(tokens):
                self._df[token] += 1

        self._N = len(documents)
        self._avg_dl = total_len / max(self._N, 1)

    def search(self, query: str, top_k: int = 5) -> list[Document]:
        """Return top_k documents ranked by BM25 score."""
        if not self._docs:
            return []

        query_tokens = self._tokenize(query)
        scores: list[float] = []

        for i, doc in enumerate(self._docs):
            dl = sum(self._tf[i].values())
            score = 0.0
            for token in query_tokens:
                if token not in self._tf[i]:
                    continue
                tf = self._tf[i][token]
                df = self._df.get(token, 0)
                idf = math.log((self._N - df + 0.5) / (df + 0.5) + 1)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
                score += idf * (numerator / denominator)
            scores.append(score)

        ranked = sorted(
            range(len(self._docs)), key=lambda i: scores[i], reverse=True
        )
        results = []
        for idx in ranked[:top_k]:
            doc = self._docs[idx]
            results.append(
                Document(
                    id=doc.id,
                    content=doc.content,
                    metadata=doc.metadata,
                    score=scores[idx],
                )
            )
        return results


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.
    RRF score = Σ 1 / (k + rank_i)
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            rrf_scores[doc.id] += 1.0 / (k + rank)
            doc_map[doc.id] = doc

    merged = sorted(doc_map.values(), key=lambda d: rrf_scores[d.id], reverse=True)
    for doc in merged:
        doc.score = rrf_scores[doc.id]
    return merged


# ── CrossEncoder re-ranker ────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Re-ranks documents using a CrossEncoder model.
    Falls back to original order if sentence-transformers is unavailable.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._model: Any = None
        self._available = False
        self._load()

    def _load(self) -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            self._model = CrossEncoder(self._model_name)
            self._available = True
        except ImportError:
            print(
                "[reranker] sentence-transformers not installed. "
                "CrossEncoder re-ranking disabled."
            )

    def rerank(self, query: str, documents: list[Document], top_k: int = 3) -> list[Document]:
        """Return top_k documents re-ranked by cross-encoder score."""
        if not self._available or not documents:
            return documents[:top_k]

        pairs = [(query, doc.content) for doc in documents]
        scores: list[float] = self._model.predict(pairs).tolist()

        for doc, score in zip(documents, scores):
            doc.score = score

        reranked = sorted(documents, key=lambda d: d.score, reverse=True)
        return reranked[:top_k]


# ── Hybrid retriever ──────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Combines BM25 + vector search via RRF, then optionally re-ranks with CrossEncoder.

    Usage:
        retriever = HybridRetriever(vector_store, cfg)
        docs = retriever.retrieve("What is the vacation policy?")
    """

    def __init__(
        self,
        vector_store: Any,  # ChromaDB collection or LangChain vectorstore
        cfg: Any,
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> None:
        self._vector_store = vector_store
        self._cfg = cfg
        self._bm25 = BM25()
        self._reranker = reranker or CrossEncoderReranker(cfg.models.reranker)
        self._indexed = False

    def build_bm25_index(self, documents: list[Document]) -> None:
        """Index documents for BM25 search."""
        self._bm25.index(documents)
        self._indexed = True

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[Document]:
        """
        Full hybrid retrieval pipeline:
        1. BM25 search
        2. Vector search
        3. RRF fusion
        4. CrossEncoder re-ranking
        """
        top_k = top_k or self._cfg.retrieval.top_k
        rerank_k = self._cfg.retrieval.rerank_top_k

        # 1. BM25
        bm25_results: list[Document] = []
        if self._indexed:
            bm25_results = self._bm25.search(query, top_k=top_k)

        # 2. Vector search
        vector_results = self._vector_search(query, top_k=top_k)

        # 3. RRF fusion
        if bm25_results and vector_results:
            fused = reciprocal_rank_fusion(
                [bm25_results, vector_results],
                k=self._cfg.retrieval.rrf_k,
            )
        elif vector_results:
            fused = vector_results
        else:
            fused = bm25_results

        # 4. Re-rank
        if fused:
            return self._reranker.rerank(query, fused, top_k=rerank_k)
        return []

    def _vector_search(self, query: str, top_k: int) -> list[Document]:
        """Perform vector similarity search using the configured vector store."""
        try:
            results = self._vector_store.similarity_search_with_score(query, k=top_k)
            docs = []
            for lc_doc, score in results:
                docs.append(
                    Document(
                        id=lc_doc.metadata.get("id", lc_doc.metadata.get("source", str(id(lc_doc)))),
                        content=lc_doc.page_content,
                        metadata=lc_doc.metadata,
                        score=float(score),
                    )
                )
            return docs
        except Exception as exc:
            print(f"[hybrid] Vector search error: {exc}")
            return []
