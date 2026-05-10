<div align="center">

# 🧠 AI Apps — Python Engineering Lab

### *Turning complexity into code, one breakthrough at a time.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

---

**A production-grade AI engineering repository** — built for developers who want to go beyond tutorials and actually ship autonomous, high-scale AI systems.

Every application in this repo is fully documented, tested, and deployable.

</div>

---

## 👤 About the Author

```
Engineering AI  |  Architecting Clouds  |  Coding Full-Stack Systems

Built for: Delta  |  JPMC  |  GE  |  Elevance Health  |  Fidelity
```

With **more than a decade of mastery** in the technology sector, I don't just build architectures — I engineer digital ecosystems. My career has been defined by delivering robust, scalable solutions for industry titans across aviation, finance, healthcare, and industrial sectors.

| Company | Domain | What I Built |
|---|---|---|
| ✈️ **Delta Air Lines** | Aviation | High-scale reservation, operations, and real-time data systems |
| 🏦 **JPMorgan Chase & Co.** | Finance | High-stakes financial infrastructure, risk platforms, trading systems |
| ⚙️ **General Electric (GE)** | Industrial | Industrial-grade digital ecosystems, IoT data pipelines |
| 🏥 **Elevance Health** | Healthcare | Mission-critical healthcare infrastructure, claims processing |
| 📈 **Fidelity Investments** | Fintech | Large-scale investment platforms, portfolio analytics |

> **Corporate architect by day, digital revolutionist by night.**
> If you're ready to break free from the guesswork and actually start building autonomous, high-scale systems — you're in the right place.
>
> *Let's build something legendary together.*

---

## 🗺️ What This Repository Covers

This is not a collection of toy demos. Every project here is engineered with **production patterns** — security, observability, scalability, and maintainability baked in from day one.

### 🔬 AI Application Tracks

```
ai-apps/
│
├── 📦 RAG Applications          ← Retrieval-Augmented Generation
│   └── rag-chatbot-app          ← ✅ LIVE — Full production RAG chatbot
│
├── 🖼️  Multimodal Applications   ← Vision + Language + Audio (coming soon)
│   ├── image-qa-app             ← Visual question answering
│   ├── doc-intelligence-app     ← PDF/image document understanding
│   └── audio-transcribe-app     ← Speech-to-text + summarization
│
├── 🤖 Autonomous Agents         ← Tool-using, self-directing AI (coming soon)
│   ├── research-agent           ← Web search + synthesis agent
│   ├── code-review-agent        ← Automated PR review agent
│   └── data-analyst-agent       ← SQL + chart generation agent
│
├── 🔧 Fine-Tuning Lab           ← QLoRA, RLHF, domain adaptation (coming soon)
│   ├── qlora-finetune           ← Fine-tune 7B models on consumer hardware
│   └── feedback-loop            ← Human feedback → training data pipeline
│
└── 🏗️  Production Patterns       ← Enterprise AI infrastructure (coming soon)
    ├── llm-gateway              ← Multi-model routing, caching, rate limiting
    ├── vector-db-benchmark      ← ChromaDB vs pgvector vs Qdrant comparison
    └── ai-observability         ← Tracing, evaluation, drift detection
```

---

## 🚀 Application 01 — `rag-chatbot-app`

> **Status: ✅ Complete & Production-Ready**

The first application in this repo is a **full-stack, production-grade RAG (Retrieval-Augmented Generation) chatbot** — the kind of system that powers enterprise knowledge bases, HR assistants, and internal documentation search at scale.

### What It Does

Instead of relying on an LLM's training data (which may be outdated or wrong), this chatbot **retrieves answers directly from your documents** — grounded, cited, and verifiable.

```
Your Question
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  GUARDRAILS  →  HYBRID SEARCH  →  LLM GENERATION        │
│  (safety +      (BM25 + vector    (Mistral 7B via        │
│   PII check)     + RRF + rerank)   Ollama, local)        │
└─────────────────────────────────────────────────────────┘
     │
     ▼
Grounded Answer + Source Citations + Quality Score
```

### Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **LLM** | Mistral 7B via Ollama | Local, private, no API costs |
| **Embeddings** | nomic-embed-text | 768-dim, fast, accurate |
| **Vector DB** | ChromaDB | Zero-config, file-based, production-upgradeable |
| **Retrieval** | BM25 + Vector + RRF + CrossEncoder | +25–35% accuracy over vector-only |
| **Orchestration** | LangChain 0.3 | Industry-standard, extensible |
| **API** | FastAPI + JWT + RBAC | Production-ready, Swagger docs |
| **UI** | Gradio | Instant web interface, zero frontend code |
| **Memory** | 3-layer (buffer + summary + facts) | Persistent across restarts |
| **Safety** | 6-category guardrails + PII detection | Enterprise-grade content filtering |
| **Fine-tuning** | QLoRA (PEFT + bitsandbytes) | Adapt to your domain on consumer GPU |

### Key Features

- **🔍 Hybrid Search** — BM25 keyword matching + semantic vector search, merged with Reciprocal Rank Fusion, re-ranked by CrossEncoder
- **🧠 3-Layer Memory** — Buffer (recent), Summary (compressed), Facts (extracted) — survives restarts
- **🛡️ Enterprise Guardrails** — 6 content safety categories, PII detection (email/phone/SSN/credit card/address/passport), prompt injection defence
- **🔐 Production Auth** — JWT tokens, BCrypt passwords, role-based access (employee / hr_admin / admin)
- **📊 Full Observability** — Audit logging (JSONL), metrics dashboard, quality scoring, semantic cache
- **🌊 Streaming** — Token-by-token output for instant perceived response
- **🎓 18-Lesson Curriculum** — Every concept taught step-by-step in `notebooks/lessons/`
- **🔧 Fine-Tuning Pipeline** — QLoRA skeleton + Google Colab notebook + training data validator

### Project Structure

```
rag-chatbot-app/
├── src/
│   ├── chatbot.py          # RAGChatbot — ingest, retrieve, generate, evaluate
│   └── config.py           # Config class with dot-access (cfg.models.generator)
├── api/
│   ├── server.py           # FastAPI: /login /health /ask /metrics /audit /dashboard
│   ├── auth.py             # JWT + BCrypt + local user store
│   ├── audit.py            # Thread-safe JSONL audit logger
│   └── monitoring.py       # Metrics tracker with dashboard
├── guardrails/
│   ├── content_safety.py   # GuardrailConfig + Guardrails (6 categories + PII)
│   └── model_governance.py # SHA-256 checksums, pickle detection, supply chain
├── retrieval/
│   └── hybrid.py           # BM25 + CrossEncoderReranker + HybridRetriever (RRF)
├── generation/
│   └── prompts.py          # PromptAssembler — 12 named templates + global instance
├── memory/
│   └── memory_bank.py      # MemoryBank — 3-layer persistent memory
├── evaluation/
│   ├── feedback.py         # FeedbackCollector — ratings → training data
│   └── rag_monitor.py      # RAGQualityMonitor, TruthfulnessScorer, GracefulDegradation
├── notebooks/
│   ├── lessons/            # 18 standalone Python teaching scripts
│   └── colab_finetune.ipynb # QLoRA fine-tuning on Google Colab (free GPU)
├── scripts/
│   ├── finetune.py         # QLoRA fine-tuning (local GPU)
│   └── training_validator.py
├── data/documents/         # Sample docs (company policy + technical guide)
├── docs/                   # 10 documentation files
├── config.yaml             # All settings — change models, thresholds, ports
├── app.py                  # Gradio web UI
└── mcp_client.py           # MCP server client (Java interop)
```

### Quick Start

```bash
# 1. Pull AI models (one-time)
ollama pull mistral
ollama pull nomic-embed-text

# 2. Set up environment
cd rag-chatbot-app
python -m venv .venv && source .venv/bin/activate   # Linux/Mac
python -m venv .venv && .venv\Scripts\activate       # Windows
pip install -r requirements-prod.txt

# 3. Configure
cp .env.example .env   # set SECRET_KEY

# 4. Ingest sample documents
python -c "from src.chatbot import RAGChatbot; RAGChatbot().ingest_documents('./data/documents')"

# 5. Launch the web UI
python app.py          # → http://localhost:7860

# 6. Or launch the production API
uvicorn api.server:app --port 8000   # → http://localhost:8000/docs
```

### Default API Users

| Username | Password | Role | Access |
|---|---|---|---|
| `admin` | `admin123` | admin | Everything |
| `hr_manager` | `hr123` | hr_admin | Metrics + dashboard |
| `employee1` | `emp123` | employee | Ask questions |

### 18-Lesson Learning Path

| # | File | Concept |
|---|---|---|
| 01 | `01_basic_chat.py` | LangChain chain pattern — `prompt \| llm \| parser` |
| 02 | `02_embeddings.py` | Text → vectors, cosine similarity |
| 03 | `03_ingest_documents.py` | Chunking + ChromaDB ingestion |
| 04 | `04_rag_chatbot.py` | Full retrieval + generation pipeline |
| 05 | `05_evaluation.py` | LLM-as-judge: relevance + groundedness |
| 06 | `06_huggingface.py` | HuggingFace models + zero-shot classification |
| 07 | `07_memory_bank.py` | 3-layer persistent memory |
| 08 | `08_prompt_engineering.py` | PromptAssembler + 12 templates |
| 09 | `09_guardrails.py` | 6-category content safety |
| 10 | `10_pii_detection.py` | PII detection + anonymization |
| 11 | `11_input_sanitization.py` | Null bytes, homoglyphs, pickle detection |
| 12 | `12_hybrid_search.py` | BM25 + vector + RRF + CrossEncoder |
| 13 | `13_streaming.py` | Token-by-token streaming output |
| 14 | `14_fine_tuning.py` | QLoRA concepts + training data format |
| 15 | `15_human_feedback.py` | Ratings → fine-tuning training data |
| 16 | `16_rag_monitoring.py` | Quality monitor + semantic cache |
| 17 | `17_api_and_auth.py` | FastAPI + JWT + RBAC |
| 18 | `18_configuration.py` | config.yaml dot-access system |

---

## 🔬 AI Research Coverage

This repository is grounded in current AI research. Here's what each application area covers and the papers/techniques behind it:

### Retrieval-Augmented Generation (RAG)

| Technique | Paper / Source | Implemented In |
|---|---|---|
| RAG baseline | *Lewis et al., 2020 — "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"* | `rag-chatbot-app` |
| BM25 ranking | *Robertson & Zaragoza, 2009 — "The Probabilistic Relevance Framework: BM25 and Beyond"* | `retrieval/hybrid.py` |
| Reciprocal Rank Fusion | *Cormack et al., 2009 — "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"* | `retrieval/hybrid.py` |
| Cross-encoder re-ranking | *Nogueira & Cho, 2019 — "Passage Re-ranking with BERT"* | `retrieval/hybrid.py` |
| HyDE (Hypothetical Document Embeddings) | *Gao et al., 2022 — "Precise Zero-Shot Dense Retrieval without Relevance Labels"* | `generation/prompts.py` |
| LLM-as-judge evaluation | *Zheng et al., 2023 — "Judging LLM-as-a-Judge with MT-Bench"* | `evaluation/rag_monitor.py` |
| Semantic caching | Industry pattern — embedding similarity for cache lookup | `evaluation/rag_monitor.py` |

### Fine-Tuning & Alignment

| Technique | Paper / Source | Implemented In |
|---|---|---|
| QLoRA | *Dettmers et al., 2023 — "QLoRA: Efficient Finetuning of Quantized LLMs"* | `scripts/finetune.py` |
| LoRA | *Hu et al., 2021 — "LoRA: Low-Rank Adaptation of Large Language Models"* | `scripts/finetune.py` |
| RLHF (simplified) | *Ouyang et al., 2022 — "Training language models to follow instructions with human feedback"* | `evaluation/feedback.py` |
| SFT (Supervised Fine-Tuning) | Standard practice | `scripts/finetune.py` |

### Safety & Alignment

| Technique | Paper / Source | Implemented In |
|---|---|---|
| Prompt injection defence | *Perez & Ribeiro, 2022 — "Ignore Previous Prompt: Attack Techniques For Language Models"* | `guardrails/content_safety.py` |
| PII detection & anonymization | GDPR / HIPAA compliance patterns | `guardrails/content_safety.py` |
| Model governance | NIST AI RMF, EU AI Act patterns | `guardrails/model_governance.py` |
| Graceful degradation | Production reliability engineering | `evaluation/rag_monitor.py` |

### Embeddings & Vector Search

| Technique | Paper / Source | Implemented In |
|---|---|---|
| Dense retrieval | *Karpukhin et al., 2020 — "Dense Passage Retrieval for Open-Domain Question Answering"* | `retrieval/hybrid.py` |
| Sentence embeddings | *Reimers & Gurevych, 2019 — "Sentence-BERT"* | `notebooks/lessons/06` |
| Cosine similarity | Standard linear algebra | `notebooks/lessons/02` |

---

## 🗓️ Roadmap

| Quarter | Application | Description |
|---|---|---|
| ✅ Q1 2026 | `rag-chatbot-app` | Production RAG chatbot — complete |
| 🔨 Q2 2026 | `multimodal-app` | Vision + language: image Q&A, PDF intelligence |
| 🔨 Q2 2026 | `research-agent` | Autonomous web research agent with tool use |
| 🔜 Q3 2026 | `llm-gateway` | Multi-model routing, semantic caching, cost tracking |
| 🔜 Q3 2026 | `code-review-agent` | Automated PR review with security scanning |
| 🔜 Q4 2026 | `data-analyst-agent` | Natural language → SQL → charts pipeline |
| 🔜 Q4 2026 | `ai-observability` | LLM tracing, evaluation drift, production monitoring |

---

## 🏗️ Engineering Philosophy

Every application in this repo is built on five non-negotiable principles:

```
1. PRODUCTION-FIRST     No toy demos. Every pattern is deployable.
2. SECURITY BY DEFAULT  Auth, guardrails, audit logging from day one.
3. OBSERVABLE           Metrics, tracing, and quality scoring built in.
4. TEACHABLE            Every concept explained with comments and lessons.
5. UPGRADEABLE          Local → cloud swap requires changing 2-3 lines.
```

---

## 📋 Prerequisites

```bash
# Required
Python 3.10+
Ollama (https://ollama.ai)

# Pull models
ollama pull mistral           # 4GB — generation
ollama pull nomic-embed-text  # 274MB — embeddings

# Optional (for fine-tuning)
NVIDIA GPU with 6GB+ VRAM
Google Colab account (free T4 GPU)
```

---

## 📄 License

MIT — see [LICENSE](LICENSE)

---

<div align="center">

**Built with precision. Engineered for scale. Documented for humans.**

*If this repo helps you ship something great — star it, fork it, build on it.*

</div>
