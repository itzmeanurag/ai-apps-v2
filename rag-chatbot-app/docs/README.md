# RAG Chatbot App

A production-grade Retrieval-Augmented Generation chatbot that answers questions from your documents — running 100% locally with no API keys or cloud costs.

---

## What It Does

- **Answers questions from your documents** — not from the LLM's training data
- **Cites sources** — every answer references which document it came from
- **Remembers conversations** — 3-layer persistent memory survives restarts
- **Blocks harmful content** — 6-category guardrails + PII detection
- **Improves over time** — human feedback → fine-tuning → better answers
- **Production-ready** — JWT auth, RBAC, rate limiting, audit logging

---

## Quick Start

```bash
# 1. Install Ollama and pull models
ollama pull mistral
ollama pull nomic-embed-text

# 2. Set up Python environment
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements-prod.txt

# 4. Configure
cp .env.example .env
# Edit .env — set SECRET_KEY at minimum

# 5. Ingest sample documents
python -c "
from src.chatbot import RAGChatbot
bot = RAGChatbot()
bot.ingest('./data/documents')
print('Done!')
"

# 6. Start the Gradio UI
python app.py
# Open http://localhost:7860

# 7. (Optional) Start the production API
uvicorn api.server:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000/docs
```

---

## 18-Lesson Curriculum

Work through these in order. Each file is standalone and runnable.

```bash
python notebooks/lessons/01_basic_chat.py
```

| # | File | Concept |
|---|------|---------|
| 01 | `01_basic_chat.py` | ChatOllama + LangChain chain pattern |
| 02 | `02_embeddings.py` | OllamaEmbeddings + cosine similarity |
| 03 | `03_ingest_documents.py` | TextLoader + chunking + ChromaDB |
| 04 | `04_rag_chatbot.py` | Retrieval + generation pipeline |
| 05 | `05_evaluation.py` | LLM-as-judge: relevance + groundedness |
| 06 | `06_huggingface.py` | Transformers + alternative models |
| 07 | `07_memory_bank.py` | 3-layer persistent memory |
| 08 | `08_prompt_engineering.py` | PromptAssembler + 12 templates |
| 09 | `09_guardrails.py` | 6-category content safety filter |
| 10 | `10_pii_detection.py` | PII anonymize/block demo |
| 11 | `11_input_sanitization.py` | Null bytes, homoglyphs, pickle detection |
| 12 | `12_hybrid_search.py` | BM25 + vector + RRF + CrossEncoder |
| 13 | `13_streaming.py` | Token-by-token streaming output |
| 14 | `14_fine_tuning.py` | QLoRA concepts + training data format |
| 15 | `15_human_feedback.py` | Collect + export training data |
| 16 | `16_rag_monitoring.py` | Quality monitor + semantic cache |
| 17 | `17_api_and_auth.py` | FastAPI + JWT + RBAC demo |
| 18 | `18_configuration.py` | config.yaml dot-access usage |

---

## Project Structure

```
rag-chatbot-app/
├── src/
│   ├── chatbot.py              # Main RAG orchestrator
│   └── config.py               # YAML config loader (dot-access)
├── api/
│   ├── server.py               # FastAPI (JWT, RBAC, rate limiting)
│   ├── auth.py                 # JWT + BCrypt + user store
│   ├── audit.py                # Thread-safe JSONL audit logger
│   └── monitoring.py           # Metrics tracker
├── guardrails/
│   ├── content_safety.py       # 6-category filter + PII detection
│   └── model_governance.py     # Checksums + pickle detection
├── retrieval/
│   └── hybrid.py               # BM25 + vector + RRF + CrossEncoder
├── generation/
│   └── prompts.py              # PromptAssembler (12 templates)
├── memory/
│   └── memory_bank.py          # 3-layer persistent memory
├── evaluation/
│   ├── feedback.py             # Human feedback + training data export
│   └── rag_monitor.py          # Quality monitor + semantic cache
├── notebooks/
│   └── lessons/                # 18 standalone teaching scripts
├── scripts/
│   ├── finetune.py             # QLoRA fine-tuning
│   └── training_validator.py   # Training data validation
├── data/
│   └── documents/              # Your source documents (2 samples included)
├── docs/                       # All documentation
├── app.py                      # Gradio web UI
├── mcp_client.py               # MCP server client
├── config.yaml                 # All configuration
├── .env.example                # Environment variable template
├── requirements.txt            # Base dependencies
├── requirements-prod.txt       # + FastAPI/uvicorn/jose/passlib
└── requirements-finetune.txt   # + transformers/peft/bitsandbytes
```

---

## Default Users

| Username | Password | Role |
|----------|----------|------|
| `admin` | `admin123` | admin |
| `hr_user` | `hr_password` | hr_admin |
| `employee1` | `emp_password` | employee |

**Change all passwords before production deployment.**

---

## Documentation

| File | Contents |
|------|----------|
| `docs/README.md` | This file |
| `docs/COMPLETE_GUIDE.md` | Full walkthrough from install to production |
| `docs/ARCHITECTURE.md` | 12-step request flow + component map |
| `docs/SECURITY.md` | 15 AI threats + 8 security layers |
| `docs/LESSON_PLAN.md` | Detailed curriculum for all 18 lessons |
| `docs/TEACHER_NOTES.md` | Every concept explained with analogies |
| `docs/PRODUCTION_ARCHITECTURE.md` | 50K employee deployment guide |
| `docs/HARDWARE_REQUIREMENTS.md` | Local, Colab, and production specs |
| `docs/TRAINING_DATA_GUIDE.md` | 13 data sources with code examples |
| `docs/QA.md` | 29 Q&As from the learning journey |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Mistral 7B via Ollama |
| Embeddings | nomic-embed-text via Ollama |
| Vector DB | ChromaDB |
| Orchestration | LangChain |
| API | FastAPI |
| UI | Gradio |
| Auth | JWT (python-jose) + BCrypt (passlib) |
| Fine-tuning | PEFT + bitsandbytes (QLoRA) |

---

## License

MIT
