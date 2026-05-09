# Architecture — Local RAG Chatbot

## System Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                       LOCAL RAG CHATBOT                     │
│                                                             │
│   ┌──────┐   ┌────────────────┐   ┌─────────────────────┐   │
│   │ User │──▶│ Gradio Web UI  │──▶│ RAGChatbot Class    │   │
│   └──────┘   │ or Terminal    │   │ (src/chatbot)       │   │
│              └────────────────┘   └─────────────────────┘   │
│                                              │              │
│       ┌───────────────────────┬──────────────┼────────┐     │
│       ▼                       ▼              ▼        │     │
│   ┌───────────────┐   ┌────────────────┐   ┌────────┐ │     │
│   │ Content Safety│   │ Query Cleaning │   │ Prompt │ │     │
│   │ (Guardrails)  │   │ & PII Anonymize│   │ Assembler│ │     │
│   └───────────────┘   └────────────────┘   └────────┘ │     │
│           │                                           │     │
│           ▼                                           │     │
│   ┌───────────────────────────────────────────────────│─┐   │
│   │                   RAG PIPELINE                    │ │   │
│   │                                                   │ │   │
│   │   ┌──────────┐   ┌────────────┐   ┌─────────────┐ │ │   │
│   │   │ ChromaDB │   │ Ollama     │   │ Memory Bank │ │ │   │
│   │   │ (Vector  │   │(Mistral 7B │   │ (Persistent │ │ │   │
│   │   │ Store)   │   │ or custom) │   │ Sessions)   │ │ │   │
│   │   └──────────┘   └────────────┘   └─────────────┘ │ │   │
│   │        │               ▲                 │        │ │   │
│   │        ▼               │                 ▼        │ │   │
│   │   ┌──────────┐   ┌────────────┐   ┌─────────────┐ │ │   │
│   │   │ Retrieve │──▶│ Generate   │◀──│ Conversation│ │ │   │
│   │   │ Top 3    │   │ Answer     │   │ History +   │ │ │   │
│   │   │ Chunks   │   └────────────┘   │ Facts +     │ │ │   │
│   │   └──────────┘         │          │ Summary     │ │ │   │
│   │                        ▼          └─────────────┘ │ │   │
│   │                  ┌────────────┐                   │ │   │
│   │                  │ Evaluate   │                   │ │   │
│   │                  │(LLM Judge) │                   │ │   │
│   │                  │ Relevance +│                   │ │   │
│   │                  │ Groundedness                   │ │   │
│   │                  └────────────┘                   │ │   │
│   │                        │                          │ │   │
│   │                        ▼                          │ │   │
│   │                  ┌────────────┐                   │ │   │
│   │                  │ Pass?      │──No──▶ Refine &   │ │   │
│   │                  └────────────┘        Retry      │ │   │
│   │                        │                          │ │   │
│   │                       Yes                         │ │   │
│   │                        ▼                          │ │   │
│   │               Return Answer + Sources + Scores    │ │   │
│   └───────────────────────────────────────────────────┴─┘   │
└─────────────────────────────────────────────────────────────┘
```

## Component Map

### 1. Models (Ollama)

| Model | Role | Size | Dimensions |
|---|---|---|---|
| `mistral` (7B) | Answer generation + Evaluation judge | ~4GB | N/A |
| `nomic-embed-text` | Document & query embeddings | ~274MB | 768 |
| Custom fine-tuned (optional) | Domain-specialized generation | Varies | N/A |

All run locally via Ollama. NO API keys, no cloud, no cost.
Custom models can be created via QLoRA fine-tuning (Lesson 11).

### 2. Vector Store (ChromaDB)

- Stores document chunks as 768-dimension vectors
- Persisted to disk in `chroma_db/` folder
- Cosine similarity search returns top 3 most relevant chunks per query
- Local file-based - no database server needed

### 3. Prompt Assembler (`src/generation/prompts.py`)

Central module that owns all 12 prompt templates. Assembles each from reusable parts:

```text
Final Prompt = Persona + Guardrails + Task Instructions + Context Block
```

| Building Block | Purpose | Example |
|---|---|---|
| Persona | Who the AI is | "You are a helpful document assistant" |
| Guardrails | Safety rules | "Do NOT invent information" |
| Task Instructions | What to do | "Score from 0.0 to 1.0" |
| Context Blocks | Dynamic placeholders | `{context}`, `{history}` |
| Format Rules | Output style | "Respond ONLY with valid JSON" |

Change a persona or guardrail once -> every prompt using it updates automatically.

### 4. Memory Bank (`src/memory/memory_bank.py`)

Three-layer persistent conversation memory:

```text
Layer 1: BUFFER (last 6 exchanges, full detail)
Layer 2: SUMMARY (LLM-compressed older exchanges)
Layer 3: KEY FACTS (extracted important info)
     │
     ▼
Saved to disk as JSON: memory_bank/{session_id}.json
Survives restarts. Supports multiple sessions.
```

When the buffer overflows, oldest exchanges are summarized by the LLM and merged into the running summary. Key facts are extracted every 3 exchanges.

### 5. Content Safety - Guardrails (`src/guardrails/content_safety.py`)

Full Bedrock-equivalent guardrail system with input AND output scanning:

| Filter | Level | What It Catches |
|---|---|---|
| SEXUAL | HIGH | Sexual content, explicit material |
| VIOLENCE | HIGH | Violence, threats, self-harm |
| HATE | HIGH | Hate speech, discrimination, slurs |
| INSULTS | HIGH | Personal insults, demeaning language |
| MISCONDUCT | HIGH | Hacking, fraud, malware, injection |
| PROMPT_ATTACK | HIGH | Jailbreaking, instruction override, system prompt extraction |

Two detection layers:
- Layer 1: Regex pattern matching (fast, <1ms)
- Layer 2: LLM-based classification (optional, ~3s, catches context-dependent attacks)

PII handling:
- Anonymize: EMAIL, PHONE, SSN, ADDRESS, NAME -> replaced with `[EMAIL]`, `[PHONE]`, etc.
- Block: CREDIT_CARD, PASSPORT, BANK_ACCOUNT -> query rejected entirely

Output scanning:
- Content safety check on model responses
- PII leakage detection (model might leak PII from documents)
- System prompt leakage detection

### 6. Model Governance (`src/guardrails/model_governance.py`)

Enterprise AI governance compliance:

| Feature | What It Does |
|---|---|
| SHA-256 checksums | Verify model file integrity, detect tampering |
| Model registry | Track versions, hashes, metadata for every release |
| Pickle blocking | Detect pickle files by extension AND magic bytes, block automatically |
| Supply chain validation | Approved sources list for models and datasets |
| Input sanitization | Null byte removal, unicode normalization, homoglyph defense, control char stripping |
| Governance report | Compliance status for all 12 enterprise policy requirements |

### 7. Quality Evaluation (LLM-as-Judge)

Every answer is scored on two metrics:

| Metric | What It Measures | Scale |
|---|---|---|
| Relevance | Does the answer address the question? | 0.0 - 1.0 |
| Groundedness | Is the answer supported by retrieved documents? | 0.0 - 1.0 |

If the average score falls below 0.6 (configurable in config.yaml), the system refines the query and retries once.

### 8. Advanced Retrieval (`src/retrieval/hybrid.py`)

Three-stage retrieval pipeline for +25-35% accuracy over vector-only search:

```text
Query
  │
  ├─ Vector Search (semantic similarity, 60% weight)
  │
  └─ BM25 Keyword Search (exact word matching, 40% weight)
  ▼
Reciprocal Rank Fusion (merge + deduplicate)
  │
  ▼
Cross-Encoder Re-Ranking (reads query + document together)
  │
  ▼
Top K most relevant documents
```

| Stage | What It Does | Accuracy Gain | Latency |
|---|---|---|---|
| Vector search | Finds documents by meaning | Baseline | ~50ms |
| + BM25 hybrid | Adds exact keyword matching | +15-20% | +10ms |
| + Re-ranking | Re-scores with cross-encoder | +10-20% | +100-200ms |

Configurable in `config.yaml`: `retrieval.use_hybrid`, `retrieval.use_reranker`.

### 9. Configuration (`src/config.py` + `config.yaml`)

All settings centralized in one YAML file. No hardcoded values in code.

```yaml
# config.yaml - change any setting without editing Python code
models:
  generator: "mistral"          # Swap to fine-tuned model here
  embeddings: "nomic-embed-text"
retrieval:
  top_k: 3
  confidence_threshold: 0.4
  use_hybrid: true
  use_reranker: true
evaluation:
  quality_threshold: 0.6
guardrails:
  sexual: "HIGH"
  violence: "HIGH"
  # ... all 6 filters configurable
```

### 10. Document Ingestion Pipeline

```text
data/documents/ folder
       │
       ▼
Load (.txt, .md, .pdf, .csv, .doc, .docx)
       │
       ▼
Chunk (1000 chars, 200 overlap, recursive splitting)
       │
       ▼
Embed (nomic-embed-text -> 768-dim vectors)
       │
       ▼
Store (ChromaDB -> chroma_db/ folder)
```

### 11. Fine-Tuning Pipeline (QLoRA)

```text
Training Data (local JSONL or Hugging Face dataset)
       │
       ▼
Load Base Model in 4-bit quantization (saves VRAM)
       │
       ▼
Add LoRA adapter layers (~0.1% of parameters)
       │
       ▼
Train adapters on your data (SFT)
       │
       ▼
Save adapters (~50-200MB)
       │
       ▼
Merge + Convert to GGUF -> Import into ollama
       │
       ▼
Use in RAG chatbot: ChatOllama(model="my-finetuned-model")
```

Supports local GPU training or Google Colab (free T4/L4 GPU).
See `notebooks/colab_finetune.ipynb` for cloud-based training of larger models.

## File Structure

```text
local-rag-chatbot/
├── src/                     # Core chatbot engine
│   ├── chatbot.py           # Main RAG chatbot (all features)
│   ├── config.py            # Configuration loader (reads config.yaml)
│   ├── ingestion/           # Document ingestion pipeline
│   │   └── pipeline.py      # Ingestion logic
│   ├── retrieval/           # Search and retrieval
│   │   └── hybrid.py        # Hybrid search (vector + BM25) + cross-encoder re-ranking
│   ├── generation/          # LLM generation and prompts
│   │   └── prompts.py       # Centralized prompt management (12 templates)
│   ├── guardrails/          # Content safety and governance
│   │   ├── content_safety.py # Content safety, PII, prompt attack (Bedrock-equivalent)
│   │   └── model_governance.py # Model integrity, pickle blocking, supply chain
│   ├── evaluation/          # Quality evaluation and feedback
│   │   ├── feedback.py      # Human feedback collection + fine-tuning data export
│   │   └── rag_monitor.py   # Quality monitoring, truthfulness, semantic cache
│   ├── memory/              # Conversation memory
│   │   └── memory_bank.py   # Persistent conversation memory
│   └── api/                 # Production API
│       ├── server.py        # FastAPI with auth + audit + monitoring
│       ├── auth.py          # JWT authentication + RBAC
│       ├── audit.py         # Audit logging
│       └── monitoring.py    # Metrics and monitoring
│
├── data/
│   ├── documents/           # Your source documents
│   └── training/            # Fine-tuning training data
│
├── notebooks/
│   └── lessons/             # Teaching scripts (run in order)
│       ├── 01_basic_chat.py # Lesson 3: LangChain + Ollama basics
│       ├── 02_embeddings.py # Lesson 4: Embeddings & vector similarity
│       ├── 03_ingest_documents.py # Lesson 5: Document ingestion pipeline
│       ├── 04_rag_chatbot.py # Lesson 5: Basic RAG chatbot
│       ├── 05_evaluation.py # Lesson 6: LLM-as-judge evaluation
│       ├── 06_huggingface.py # Lesson 7: Hugging Face ecosystem
│       ├── 07-18: Memory, Prompts, Guardrails, PII, Sanitization, Hybrid Search,
│       │          Streaming, Fine-Tuning, Feedback, Monitoring, API, Config
│       └── colab_finetune.ipynb # Lesson 11: QLoRA fine-tuning (Google Colab)
│
├── scripts/
│   ├── finetune.py          # Lesson 11: QLoRA fine-tuning (local)
│   └── training_validator.py# Data filtering, validation split, benchmarking
│
├── tests/                   # Test suite
├── docs/                    # Documentation
│   ├── COMPLETE_GUIDE.md    # Full walkthrough: local -> Colab -> production
│   ├── LESSON_PLAN.md       # Curriculum (18 lessons)
│   ├── ARCHITECTURE.md      # This file
│   ├── PRODUCTION_ARCHITECTURE.md # Production deployment guide
│   ├── SECURITY.md          # Cybersecurity & AI threat analysis
│   └── QA.md                # Frequently asked questions
│
├── pyproject.toml           # Project metadata
├── requirements.txt         # Core dependencies
├── requirements-finetune.txt# Fine-tuning dependencies
├── requirements-prod.txt    # Production API dependencies
├── config.yaml              # Central configuration (all settings)
└── README.md                # Project overview
```

## Data Flow - Full Request Lifecycle

```text
1. User types question
     │
2. Input Sanitization (model_governance: length, null bytes, unicode, homoglyphs)
     │
3. Input Guardrails (guardrails: 6 content filters + PII + prompt attack)
     ├── Blocked? -> Return safety message
     └── (passed)
4. Query Cleaning (whitespace normalization)
     │
5. Hybrid Search (Vector + BM25 -> Reciprocal Rank Fusion -> Cross-Encoder Re-Rank -> Top K)
     │
6. Prompt Assembly (Prompt Assembler: persona + guardrails + task + context + history)
     │
7. LLM Generation (Mistral 7B or fine-tuned model via Ollama)
     │
8. Output Guardrails (guardrails: content safety + PII leakage + prompt leakage scan)
     ├── Unsafe? -> Block response
     └── (passed)
9. Quality Evaluation (LLM-as-judge: relevance + groundedness)
     ├── Below 0.6? -> Refine query -> Go to step 5 (once)
     └── (passed)
10. Save to Memory Bank (buffer + periodic summarization + fact extraction)
     │
11. Audit Log (user, question, answer, scores, timing)
     │
12. Return: answer + source files + evaluation scores + PII flags
```

## AWS Reference Mapping

| AWS Reference Component | Our Local Equivalent | File |
|---|---|---|
| Amazon Bedrock Knowledge Base | ChromaDB + nomic-embed-text | notebooks/lessons/03, src/chatbot |
| Claude 3 Haiku (generation) | Mistral 7B via Ollama (or fine-tuned) | src/chatbot |
| Mistral 7B Instruct (evaluation) | Mistral 7B via Ollama (eval prompt) | notebooks/lessons/05, src/chatbot |
| Aurora PostgreSQL pgvector (1024d) | ChromaDB (768d) | notebooks/lessons/03, src/chatbot |
| TITAN_EMBED_TEXT_V2_1024 | nomic-embed-text (768d) | notebooks/lessons/02, notebooks/lessons/03 |
| S3 Document Storage | Local `data/documents/` folder | notebooks/lessons/03 |
| S3-backed Session Storage | JSON files in `memory_bank/` | src/memory/memory_bank |
| Bedrock Guardrails (all HIGH) | 6-category guardrails + PII + prompt attack | src/guardrails/content_safety |
| Bedrock PII Anonymization | Regex PII detection (anonymize + block modes) | src/guardrails/content_safety |
| Bedrock Prompt Management | Prompt Assembler (12 templates) | src/generation/prompts |
| SageMaker Model Registry | Model registry with SHA-256 hashes | src/guardrails/model_governance |
| DynamoDB Session TTL | cleanup_old_sessions() (30-day) | src/memory/memory_bank |
| Bedrock PII Anonymization | Regex-based PII replacement | src/chatbot |
| Lambda Document Processing | Direct Python function calls | notebooks/lessons/03 |
| Next.js + AppSync Frontend | Gradio web interface / Terminal | src/chatbot |
| Ping Identity OIDC Auth | Not needed (local single-user) | - |
| Bedrock Prompt Management | Prompt Assembler (12 templates) | src/generation/prompts |
| DynamoDB Session TTL | cleanup_old_sessions() (30-day) | src/memory/memory_bank |
| SageMaker Fine-Tuning | QLoRA via local GPU or Google Colab | scripts/finetune, notebooks/colab_finetune |

## Technology Stack

| Layer | Technology | Version |
|---|---|---|
| AI Runtime | Ollama | Latest |
| Chat Model | Mistral 7B (or fine-tuned custom) | Latest via ollama |
| Embedding Model | nomic-embed-text | Latest via ollama |
| Orchestration | LangChain | 0.3.7 |
| Vector Database | ChromaDB | 0.5.18 |
| HF Embeddings | sentence-transformers | 3.3.1 |
| HF Classification | transformers | 4.46.3 |
| Fine-Tuning | PEFT + TRL + bitsandbytes | Latest |
| Web Interface | Gradio | 5.6.0 |
| Language | Python | 3.10+ |
| OS | Any (Windows, macOS, Linux) | - |
| Cloud Training | Google Colab (free T4/L4 GPU) | Optional |
