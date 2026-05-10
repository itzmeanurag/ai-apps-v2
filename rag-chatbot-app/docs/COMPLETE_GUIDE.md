# The Complete Guide — From Zero to Production RAG Chatbot

A comprehensive reference covering every concept, decision, and implementation
in this project. Written as a learning resource — explains not just WHAT we built,
but WHY we built it that way and WHAT ALTERNATIVES exist.

---

## Table of Contents

### Part 1: Foundation Concepts
1. [What Is an LLM?](#1-what-is-an-llm)
2. [What Is RAG?](#2-what-is-rag)
3. [What Is LangChain?](#3-what-is-langchain)
4. [What Are Embeddings and Vector Databases?](#4-what-are-embeddings-and-vector-databases)

### Part 2: The Journey - How We Built This
5. [Phase 1: Learning (Lessons 0-8)](#5-phase-1-learning-lessons-0-8)
6. [Phase 2: Enhancement (Lessons 9-10)](#6-phase-2-enhancement-lessons-9-10)
7. [Phase 3: Fine-Tuning (Lesson 11)](#7-phase-3-fine-tuning-lesson-11)
8. [Phase 4: Production (Lesson 12)](#8-phase-4-production-lesson-12)
9. [Phase 5: Engineering Optimization](#9-phase-5-engineering-optimization)

### Part 3: Deep Dives
10. [Guardrails — Content Safety In Depth](#10-guardrails--content-safety-in-depth)
11. [Memory Bank — How Conversation Memory Works](#11-memory-bank--how-conversation-memory-works)
12. [Prompt Assembler — Centralized Prompt Management](#12-prompt-assembler--centralized-prompt-management)
13. [Model Governance — Enterprise AI Compliance](#13-model-governance--enterprise-ai-compliance)
14. [Hybrid Search and Re-Ranking](#14-hybrid-search-and-re-ranking)
15. [Feedback and Refinement](#15-feedback-and-refinement)
16. [RAG Monitoring and Truthfulness](#16-rag-monitoring-and-truthfulness)
17. [Graceful Degradation](#17-graceful-degradation)
18. [Semantic Caching](#18-semantic-caching)

### Part 4: Security
19. [The 15 AI Security Threats](#19-the-15-ai-security-threats)
20. [How We Protect Against Each Threat](#20-how-we-protect-against-each-threat)
21. [Bedrock Guardrails Equivalent](#21-bedrock-guardrails-equivalent)

### Part 5: Models and Training
22. [Models — What We Use and Alternatives](#22-models--what-we-use-and-alternatives)
23. [Fine-Tuning In Depth](#23-fine-tuning-in-depth)
24. [Training Data Sources](#24-training-data-sources)

### Part 6: Production and Beyond
25. [Production Architecture](#25-production-architecture)
26. [Project Structure and Configuration](#26-project-structure-and-configuration)
27. [Quick Reference — All Commands](#27-quick-reference--all-commands)

---

# Part 1: Foundation Concepts

## 1. What Is an LLM?

A Large Language Model (LLM) is a program trained on massive text data that generates
human-like text. Think of it as a very smart autocomplete — given some text, it predicts
what comes next, one token at a time.

### How It Works (Simplified)

```text
Input: "The capital of France is"
Model: Predicts next token -> "Paris"
       Predicts next token -> "."
       Predicts next token -> "It"
       Predicts next token -> "is"
       ... and so on
```

The model doesn't "know" facts. It learned statistical patterns from training data.
When it says "Paris," it's because in its training data, "capital of France" was
overwhelmingly followed by "Paris."

### Key Concepts

| Term | Meaning |
|------|---------|
| Parameters | The model's learned knowledge — billions of numbers encoding patterns |
| Tokens | Chunks of text (~3/4 of a word). Models process tokens, not words |
| Context Window | Max text the model can "see" at once (Mistral: 32K tokens) |
| Temperature | Randomness control. 0.0 = deterministic, 1.0 = creative |
| Inference | Running the model to get an answer |
| Hallucination | When the model confidently says something false |

### Models We Use

| Model | Parameters | Size | Role in Our Project |
|-------|------------|------|---------------------|
| Mistral 7B | 7 billion | ~4GB | Answer generation + evaluation |
| nomic-embed-text | 137 million | ~274MB | Document embeddings (text -> vectors) |

### Alternative Models

| Model | Parameters | Strengths | When to Use Instead |
|-------|------------|-----------|---------------------|
| Llama 3.1 8B | 8B | Meta's latest, strong reasoning | Better general quality |
| Gemma 2 9B | 9B | Google's model, very capable | Better instruction following |
| Phi 3.5 | 3.8B | Small but capable (Microsoft) | Limited hardware (8GB RAM) |
| Qwen 2.5 | 7B | Alibaba's model, multilingual | Non-English documents |
| DeepSeek V2 | 16B (MoE) | Mixture of Experts, efficient | Better quality with similar speed |

All available via ollama: `ollama pull llama3.1`, `ollama pull gemma2`, etc.

---

## 2. What Is RAG?

RAG (Retrieval Augmented Generation) is a technique that makes LLMs answer from
YOUR documents instead of their general training data.

### The Problem RAG Solves

```text
WITHOUT RAG:
User: "What is our company's leave policy?"
LLM:  "I don't know your specific policy." (or worse, makes one up)

WITH RAG:
User: "What is our company's leave policy?"
System: [searches your documents -> finds leave-policy.pdf -> extracts relevant section]
LLM:  "According to your policy, employees get 20 days annual leave..." (grounded answer)
```

### How RAG Works

```text
Step 1: INGEST (one-time)
    Your documents -> split into chunks -> convert to vectors -> store in vector DB

Step 2: RETRIEVE (every question)
    User question -> convert to vector -> find similar document vectors -> get top 3 chunks

Step 3: GENERATE (every question)
    Top 3 chunks + user question -> send to LLM -> LLM answers using the chunks
```

### Why RAG Instead of Fine-Tuning?

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| Updates | Add/remove documents anytime | Must retrain the model |
| Source attribution | Can cite which document | Cannot cite sources |
| Hallucination | Grounded in actual documents | Can still hallucinate |
| Cost | Free (just add documents) | Needs GPU time |
| Best for | "What does document X say?" | "Answer in our company's style" |

Best approach: Use BOTH. Fine-tune for style, RAG for content.

### Alternatives to RAG

| Approach | How It Works | When to Use |
|----------|--------------|-------------|
| RAG (what we use) | Retrieve documents, feed to LLM | Most use cases |
| Long-context LLMs | Put ALL documents in the prompt | Small document sets (<100 pages) |
| Fine-tuning only | Train the model on your data | When style matters more than accuracy |
| Knowledge graphs | Structured relationships between concepts | Hierarchical data (org charts, regulations) |
| Hybrid RAG + KG | RAG for text + knowledge graph for relationships | Complex enterprise systems |

---

## 3. What Is LangChain?

LangChain is a Python framework that CONNECTS things together. It doesn't do AI
itself — it's the plumbing between your model, your data, and your logic.

### Analogy

```text
Ollama      = the engine (runs the AI model)
LangChain   = the car frame, steering wheel, pedals (connects everything)
Your code   = the driver (decides where to go)
ChromaDB    = the GPS database (stores and searches locations/documents)
```

### The Chain Pattern

Everything in LangChain uses the pipe (`|`) operator to chain steps:

```python
chain = prompt_template | llm | output_parser
result = chain.invoke({"question": "What is the leave policy?"})
```

Data flows left to right: template formats the question -> LLM generates answer -> parser extracts text.

### What LangChain Provides in Our Project

| Component | LangChain Class | What It Does |
|-----------|-----------------|--------------|
| LLM connection | `ChatOllama` | Talks to Ollama models |
| Embeddings | `OllamaEmbeddings` | Converts text to vectors |
| Vector store | `Chroma` | Stores and searches vectors |
| Document loaders | `TextLoader`, `PyPDFLoader`, etc. | Reads files |
| Text splitter | `RecursiveCharacterTextSplitter` | Chunks documents |
| Prompt templates | `ChatPromptTemplate` | Formats prompts |
| Output parser | `StrOutputParser` | Extracts text from responses |

### Alternatives to LangChain

| Framework | Language | Strengths | When to Use |
|-----------|----------|-----------|-------------|
| LangChain (what we use) | Python | Largest ecosystem, most integrations | Default choice for Python |
| LangChain4j | Java | Same concepts, Java ecosystem | Java/Spring Boot shops |
| LlamaIndex | Python | Better for pure RAG (simpler API) | RAG-focused projects |
| Haystack | Python | Production-focused, modular | Enterprise deployments |
| Semantic Kernel | C#/Python | Microsoft's framework | Azure-heavy environments |
| Direct API calls | Any | No framework overhead | Simple projects |

---

## 4. What Are Embeddings and Vector Databases?

### Embeddings

An embedding converts text into a list of numbers (a vector) that captures its MEANING.
Similar texts get similar numbers.

```text
"The cat sat on the mat"       -> [0.2, 0.8, 0.1, 0.5, ...] (768 numbers)
"A kitten rested on the rug"   -> [0.3, 0.7, 0.2, 0.4, ...] (similar numbers!)
"Python is a programming language" -> [0.9, 0.1, 0.8, 0.2, ...] (very different numbers)
```

### How Similarity Search Works

```text
User asks: "How many vacation days do I get?"
    ↓
Convert question to vector: [0.4, 0.6, 0.3, ...]
    ↓
Compare against all document chunk vectors using cosine similarity
    ↓
Closest match: "Employees get 20 days of annual leave" (vector: [0.4, 0.5, 0.3, ...])
    ↓
Return this chunk to the LLM
```

The embedding model understood that "vacation days" and "annual leave" mean the same thing,
even though the exact words are different. This is the power of semantic search.

### Embedding Models We Use

| Model | Dimensions | Size | Provider |
|-------|------------|------|----------|
| nomic-embed-text (what we use) | 768 | ~274MB | ollama |

### Alternative Embedding Models

| Model | Dimensions | Strengths | How to Use |
|-------|------------|-----------|------------|
| all-MiniLM-L6-v2 | 384 | Fast, small, good quality | Hugging Face sentence-transformers |
| BGE-large-en-v1.5 | 1024 | High accuracy, English | Hugging Face |
| E5-large-v2 | 1024 | Microsoft, strong on benchmarks | Hugging Face |
| TITAN Embed V2 (AWS) | 1024 | AWS Bedrock native | AWS only |
| text-embedding-3-large (OpenAI) | 3072 | Highest quality | OpenAI API (paid) |
| Cohere embed-v3 | 1024 | Multilingual | Cohere API (paid) |

### Vector Databases

| Database | Type | Strengths | When to Use |
|----------|------|-----------|-------------|
| ChromaDB (what we use) | File-based | Simple, no server needed | Learning, small projects |
| PostgreSQL + pgvector | SQL database | ACID, concurrent access, backup | Production |
| Qdrant | Purpose-built | Fast, scalable, filtering | Large-scale production |
| Pinecone | Cloud service | Managed, serverless | Cloud-native projects |
| Weaviate | Purpose-built | GraphQL API, hybrid search | Complex queries |
| Milvus | Purpose-built | Billion-scale vectors | Very large datasets |
| FAISS (Facebook) | In-memory library | Fastest search | Research, benchmarking |

---

# Part 2: The Journey - How We Built This

## 5. Phase 1: Learning (Lessons 0-8)

### What We Built

Started from zero knowledge of AI. Built progressively:

| Step | What We Did | File Created | Concept Learned |
|------|-------------|--------------|-----------------|
| Lesson 0 | Learned vocabulary | - | 40+ AI/ML terms |
| Lesson 1 | Installed ollama, pulled Mistral | - | Running AI locally |
| Lesson 2 | Set up Python environment | `requirements.txt` | Virtual environments, dependencies |
| Lesson 3 | First LLM chat via LangChain | `notebooks/lessons/01_basic_chat.py` | Chains, prompts, output parsing |
| Lesson 4 | Explored embeddings | `notebooks/lessons/02_embeddings.py` | Text -> vectors, cosine similarity |
| Lesson 5a | Document ingestion | `notebooks/lessons/03_ingest_documents.py` | Chunking, ChromaDB, metadata |
| Lesson 5b | Basic RAG chatbot | `notebooks/lessons/04_rag_chatbot.py` | Retrieval + generation pipeline |
| Lesson 6 | Quality evaluation | `notebooks/lessons/05_evaluation.py` | LLM-as-judge, relevance, groundedness |
| Lesson 7 | Hugging Face models | `notebooks/lessons/06_huggingface.py` | Transformers, classification, embeddings |
| Lesson 8 | Combined everything | `src/chatbot.py` | Full RAG chatbot with all features |

### Key Decision: notebooks/lessons/ vs src/

`notebooks/lessons/` = practice exercises. Each teaches ONE concept in isolation.
`src/` = the actual product. Combines everything into a working system.

If you delete `notebooks/lessons/`, the chatbot still works. Lessons are training wheels.

### Setup Commands

```bash
ollama pull mistral && ollama pull nomic-embed-text
cd local-rag-chatbot
python -m venv venv && source venv/Scripts/activate
pip install -r requirements.txt
python notebooks/lessons/01_basic_chat.py   # Start here
```

---

## 6. Phase 2: Enhancement (Lessons 9-10)

### Lesson 9: Memory Bank (`src/memory/memory_bank.py`)

**Problem:** The chatbot forgot everything on restart. Conversations were stored
in a Python dictionary that died when the process ended.

**Solution:** Three-layer persistent memory:

| Layer | What It Stores | Analogy |
|-------|----------------|---------|
| Buffer | Last 6 exchanges in full detail | Short-term memory |
| Summary | LLM-compressed older exchanges | Long-term memory |
| Key Facts | Important info extracted by LLM | Notes you write down |

Saved to `memory_bank/{session_id}.json`. Survives restarts.

### Lesson 10: Prompt Assembler (`src/generation/prompts.py`)

**Problem:** 8+ prompts hardcoded across 5 files. Changing one behavior meant
editing multiple files.

**Solution:** Central prompt factory. 12 templates assembled from reusable parts:

```text
Final Prompt = Persona + Guardrails + Task Instructions + Context Block
```

Change a persona once -> every prompt using it updates automatically.

---

## 7. Phase 3: Fine-Tuning (Lesson 11)

### What We Built

| File | Purpose |
|------|---------|
| `scripts/finetune.py` | QLoRA fine-tuning on local GPU |
| `scripts/training_validator.py` | Data quality filtering, validation split, before/after benchmarking |
| `notebooks/colab_finetune.ipynb` | QLoRA fine-tuning on Google Colab (free GPU) |

### How QLoRA Works

```text
Base Model (7B params, 14GB) -> Quantize to 4-bit (3.5GB) -> Add LoRA adapters (0.1%)
    -> Train ONLY adapters -> Save adapters (~100MB) -> Merge -> Convert to GGUF -> Ollama
```

### Current State

The fine-tuning scripts are READY but not yet executed. The chatbot currently uses
stock Mistral 7B. After fine-tuning, change one line in `src/chatbot.py` — or
better, change `models.generator` in `config.yaml`.

---

## 8. Phase 4: Production (Lesson 12)

### What We Built

| Module | Purpose | File |
|--------|---------|------|
| API Server | FastAPI with all features | `src/api/server.py` |
| Authentication | JWT + bcrypt + 3 roles | `src/api/auth.py` |
| Audit Logging | Every Q&A recorded to JSONL | `src/api/audit.py` |
| Monitoring | Request counts, latency, quality | `src/api/monitoring.py` |

### API Endpoints

| Method | Endpoint | Auth | Role | Purpose |
|--------|----------|------|------|---------|
| POST | /login | No | - | Get JWT token |
| GET | /health | No | - | Health check |
| POST | /ask | Yes | employee+ | Ask a question |
| GET | /metrics | Yes | hr_admin+ | View metrics |
| GET | /audit | Yes | admin | Query audit logs |
| POST | /ingest | Yes | admin | Re-ingest documents |
| GET | /dashboard | Yes | hr_admin+ | Combined view |

---

## 9. Phase 5: Engineering Optimization

After the senior AI engineer review (scored 6.5/10 initially), we fixed 5 issues:

| Fix | Problem | Solution | Score Impact |
|-----|---------|----------|--------------|
| #1 | Everything hardcoded | Created `config.yaml` + `src/config.py` | Configuration: 3->9 |
| #2 | Dead code in chatbot.py | Removed old guardrails functions | Code Quality: 7->8 |
| #3 | No retrieval confidence | Added similarity threshold, refuses low-confidence | Performance: 5->7 |
| #4 | No streaming | Added `ask_stream()`, token-by-token output | Performance: 7->8 |
| #5 | No hybrid search | Added BM25 + cross-encoder re-ranking | Performance: 8->9 |

Final score: 8.5/10.

---

# Part 3: Deep Dives

## 10. Guardrails — Content Safety In Depth

### What Guardrails Do

Guardrails are safety filters that check BOTH input (user's question) and output
(model's answer) for harmful content. Our implementation mirrors AWS Bedrock Guardrails.

### Architecture (4 Layers)

---

Layer 1: REGEX (fast, <1ms)
    Pattern matching for each of 6 content categories
    Catches obvious violations: "hack", "kill", "ignore previous instructions"

Layer 2: LLM CLASSIFIER (accurate, ~3s, optional)
    Mistral classifies the text into categories
    Catches context-dependent issues regex misses

Layer 3: PII HANDLER
    Anonymize: EMAIL, PHONE, SSN, ADDRESS, NAME -> [EMAIL], [PHONE], etc.
    Block: CREDIT_CARD, PASSPORT, BANK_ACCOUNT -> reject entire query

Layer 4: OUTPUT SCANNER (post-generation)
    Same content safety check on the model's RESPONSE
    Plus: PII leakage detection, system prompt leakage detection
---

### The 6 Content Filter Categories

| Category | Level | What It Catches | Example Blocked Query |
|----------|-------|-----------------|-----------------------|
| SEXUAL | HIGH | Sexual content, explicit material | "Write an erotic story" |
| VIOLENCE | HIGH | Violence, threats, self-harm | "How to make a weapon" |
| HATE | HIGH | Hate speech, discrimination | "Why are [group] inferior" |
| INSULTS | HIGH | Personal insults, harassment | "You're an idiot" |
| MISCONDUCT | HIGH | Hacking, fraud, illegal activity | "How to hack a system" |
| PROMPT_ATTACK | HIGH | Jailbreaking, instruction override | "Ignore all previous rules" |

### Alternatives to Our Guardrails

| Approach | Accuracy | Speed | Cost |
|----------|----------|-------|------|
| Regex (what we have, Layer 1) | Low-Medium | <1ms | Free |
| LLM Classifier (our Layer 2) | Medium-High | ~3s | Free (uses same model) |
| Meta Llama Guard 3 | High | ~500ms | Free (open source, needs GPU) |
| NVIDIA NeMo Guardrails | High | ~200ms | Free (open source) |
| AWS Bedrock Guardrails | Very High | ~100ms | Paid (AWS) |
| OpenAI Moderation API | High | ~200ms | Free (but sends data to OpenAI) |

---

## 11. Memory Bank — How Conversation Memory Works

### The Problem

Without memory, every question is independent. The chatbot can't understand
"How many days exactly?" because it doesn't know you were asking about leave policy.

### Three-Layer Architecture

---
Exchange 1-6:   Stored in BUFFER (full detail)
Exchange 7:     Buffer overflows -> oldest exchanges SUMMARIZED by LLM
                Summary saved, buffer keeps only recent ones
Exchange 9:     LLM EXTRACTS KEY FACTS from recent conversation
Exchange 10+:   Cycle continues: buffer -> summary -> facts
---

### What Gets Injected into the Prompt

---
KEY FACTS FROM PREVIOUS CONVERSATIONS:
  - User is asking about HR leave policy
  - Company allows 20 days annual leave
  - User prefers concise answers

SUMMARY OF EARLIER CONVERSATION:
  User asked about leave policy, remote work rules, and expense limits.
  They were particularly interested in carry-over rules.

RECENT EXCHANGES:
  User: Can I carry over unused days?
  Assistant: Yes, up to 5 days can be carried over to the next year.
  User: What about sick leave?
  Assistant: Sick leave is separate — 10 days per year with medical certificate.
---

### Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| Our Memory Bank (buffer + summary + facts) | Persistent, compressed, intelligent | Complex, LLM calls for summarization |
| LangChain ConversationBufferMemory | Simple, built-in | No compression, fills context window |
| LangChain ConversationSummaryMemory | Compressed | No key facts, no persistence |
| Redis-backed memory | Fast, distributed | Needs Redis server |
| Database-backed (PostgreSQL) | Queryable, scalable | Needs database server |

---

## 12. Prompt Assembler — Centralized Prompt Management

### 12 Registered Prompts

| Name | Used By | Purpose |
|------|---------|---------|
| `basic_chat` | notebooks/lessons/01 | Simple conversation |
| `rag_simple` | notebooks/lessons/04 | RAG without history |
| `rag_with_history` | src/chatbot | RAG with conversation context |
| `rag_for_evaluation` | notebooks/lessons/05 | RAG for evaluation testing |
| `eval_combined` | src/chatbot | Relevance + groundedness scoring |
| `eval_relevance` | notebooks/lessons/05 | Relevance-only scoring |
| `eval_groundedness` | notebooks/lessons/05 | Groundedness-only scoring |
| `refine_query` | notebooks/lessons/05 | Query optimization |
| `memory_summarize_new` | src/memory/memory_bank | Summarize new conversation |
| `memory_summarize_update` | src/memory/memory_bank | Update existing summary |
| `memory_extract_facts` | src/memory/memory_bank | Extract key facts |
| `memory_demo_chat` | src/memory/memory_bank | Demo chat with history |

### How Assembly Works

```python
# Each prompt is built from reusable parts:
system_message = Persona + Guardrails + Task Instructions + Context Block

# Example: rag_with_history prompt assembles:
# Persona:      "You are a helpful document assistant..."
# Guardrails:   "SAFETY RULES: Do NOT invent information..."
# Task:         "ONLY use information from context documents..."
# Context:      "{context}" + "{history}" placeholders
```

---

## 13. Model Governance — Enterprise AI Compliance

### What It Implements

| Requirement | Implementation |
|-------------|----------------|
| SHA-256 checksums | Verify model file integrity, detect tampering |
| Model registry | Track versions, hashes, metadata for every release |
| Pickle blocking | Detect by extension AND magic bytes, block automatically |
| Supply chain validation | Approved sources list for models and datasets |
| Input sanitization | Null bytes, unicode normalization, homoglyph defense |
| Governance report | Compliance status for all 12 enterprise policy requirements |

### Pickle Blocking — Why It Matters

Pickle is Python's serialization format. It can contain EXECUTABLE CODE.
Loading a malicious pickle file = running the attacker's code on your machine.

```python
# Our detection (src/guardrails/model_governance.py):
# 1. Check file extension (.pkl, .pickle -> BLOCKED)
# 2. Check magic bytes (catches RENAMED pickle files)
# 3. Recommend SafeTensors format instead
```

### Input Sanitization - What It Catches

| Attack | How It Works | Our Defense |
|--------|--------------|-------------|
| Null byte injection | `\x00` in input can truncate strings | Removed before processing |
| Homoglyph attack | Cyrillic 'a' looks like Latin 'a', bypasses regex | Unicode NFKC normalization |
| Control characters | Hidden chars that manipulate display | Stripped (except newline/tab) |
| Context overflow | Very long input pushes guardrails out of context | Length limit (2000 chars) |

---

## 14. Hybrid Search and Re-Ranking

### Why Vector Search Alone Isn't Enough

| Query | Vector Search Result | Problem |
|-------|----------------------|---------|
| "What is Policy 4.2?" | Random policy chunks | "4.2" is an identifier, not a meaning |
| "What does FMLA mean?" | Insurance-related chunks | FMLA is an acronym, needs exact match |
| "Section 3.1.2 requirements" | General requirements chunks | Section numbers need keyword matching |

### Our Three-Stage Pipeline (`src/retrieval/hybrid.py`)

---
Stage 1: VECTOR SEARCH (60% weight)
    Finds documents by MEANING
    "vacation" -> finds "annual leave" (synonym understanding)

Stage 2: BM25 KEYWORD SEARCH (40% weight)
    Finds documents by EXACT WORDS
    "Policy 4.2" -> finds "Policy 4.2" (exact match)

Stage 3: RECIPROCAL RANK FUSION
    Merges results from both searches
    Uses rank positions (not raw scores) — scale-independent

Stage 4: CROSS-ENCODER RE-RANKING
    Reads FULL query + FULL document TOGETHER
    Much more accurate than comparing compressed vectors
    Re-scores and re-orders the merged results
---

### Accuracy Impact

| Approach | Accuracy vs Baseline |
|----------|----------------------|
| Vector only (original) | Baseline |
| + BM25 hybrid | +15-20% |
| + Cross-encoder re-ranking | +10-20% on top |
| All combined | +25-35% total |

---

## 15. Feedback and Refinement

### What We Have: Automatic Refinement (Machine Feedback)

Our chatbot has a built-in quality loop — when the LLM-as-judge scores an answer
below the threshold (0.6), the system automatically refines the query and retries:

---
User asks question
    |
    v
Generate answer
    |
    v
LLM-as-judge scores: relevance + groundedness
    |
    |-- Score >= 0.6 -> PASS -> Return answer to user
    |
    |-- Score < 0.6 -> FAIL -> Refine query -> Retrieve again -> Generate again
                            -> Re-score -> Return (pass or fail)
---

This is AUTOMATIC refinement — no human involved. The system self-corrects.

**Where it lives:** `src/chatbot.py` -> `ask()` method, Step 6.

### What We DON'T Have: Human Feedback

There is no way for users to say:
- "This answer was helpful" (thumbs up)
- "This answer was wrong" (thumbs down)
- "The answer should have been X" (correction)

Without human feedback, we can't:
- Know which answers are actually good in the real world
- Improve the system based on user experience
- Generate training data from real usage for fine-tuning
- Track user satisfaction over time

### The Two Types of Feedback

---
TYPE 1: AUTOMATIC (what we have)

LLM judges its own answers -> refines if low quality
Happens: Every response (or sampled)
Speed: Instant (no human needed)
Accuracy: Medium (model might not catch its own mistakes)
Used for: Real-time quality improvement

TYPE 2: HUMAN (what we're missing)

Users rate answers -> feedback stored -> used to improve system
Happens: When user chooses to give feedback
Speed: Slow (depends on user participation)
Accuracy: High (humans know if the answer actually helped)
Used for: Long-term system improvement, fine-tuning data
---

### How Human Feedback Would Work

---
User asks: "How many leave days do I get?"
Bot answers: "You get 20 days per year."

    |-- User clicks 👍 -> Logged as positive feedback
    |   -> This Q&A pair becomes a fine-tuning candidate
    |
    |-- User clicks 👎 -> Logged as negative feedback
    |   -> Flagged for review
    |   -> User can optionally provide the correct answer
    |   -> Used to identify weak areas in retrieval/generation
    |
    |-- User provides correction: "Actually it's 25 days for senior employees"
        -> Stored as a correction
        -> Can be used to update documents or fine-tuning data
---
### The Feedback Loop (Full Cycle)

---
+-------------------------------------------------------------+
|                        FEEDBACK LOOP                        |
|                                                             |
|  1. User asks question -> Bot answers                       |
|  2. User gives feedback (👍/👎/correction)                  |
|  3. Feedback stored in feedback_log.jsonl                   |
|  4. Periodically analyze feedback:                          |
|     - Which topics get most 👎?                             |
|     - Which documents are cited in wrong answers?           |
|     - What corrections do users provide?                    |
|  5. Use insights to:                                        |
|     a. Update documents (fix source content)                |
|     b. Improve prompts (adjust instructions)                |
|     c. Generate fine-tuning data (👍 pairs = training)      |
|     d. Adjust retrieval (tune confidence threshold)         |
|  6. System improves -> better answers -> repeat             |
+-------------------------------------------------------------+
---

### How Feedback Connects to Fine-Tuning

This is the key insight: human feedback generates training data.

---
Step 1: Collect 500+ 👍 rated Q&A pairs from real usage
Step 2: Review and clean them (remove duplicates, verify accuracy)
Step 3: Format as training data (JSONL with conversations)
Step 4: Fine-tune the model on these real-world Q&A pairs
Step 5: Deploy the fine-tuned model
Step 6: Collect more feedback -> repeat the cycle
---

This is called RLHF (Reinforcement Learning from Human Feedback) at a simplified level.
It's how ChatGPT, Claude, and every major AI product improves over time.

### Alternatives for Feedback Collection

| Approach | Complexity | Quality | Best For |
|----------|------------|---------|----------|
| Thumbs up/down (binary) | Low | Low (no detail) | Quick sentiment tracking |
| Star rating (1-5) | Low | Medium | Satisfaction measurement |
| Thumbs + optional comment | Medium | High | Actionable feedback |
| Correction field | Medium | Very High | Training data generation |
| A/B testing (two answers) | High | Very High | Comparing model versions |
| Implicit feedback (click tracking) | High | Medium | Large-scale systems |

---

## 16. RAG Monitoring and Truthfulness

### The Critical Question: How Do You KNOW Your System Is Truthful?

You can't just ask the LLM "are you telling the truth?" — it will say yes.
Truthfulness must be measured through multiple independent signals:

---
Signal 1: GROUNDEDNESS SCORE (LLM-as-judge)
    "Is every claim in the answer supported by the retrieved documents?"
    Score 0.0-1.0. Above 0.7 = well-grounded. Below 0.4 = hallucination suspect.

Signal 2: RETRIEVAL CONFIDENCE
    "Did we actually find relevant documents?"
    If the best document match scores 0.2, the answer is probably made up.

Signal 3: SOURCE ATTRIBUTION
    "Can we trace every claim to a specific document?"
    If the answer cites sources, humans can verify.

Signal 4: GROUND TRUTH TESTING
    "Compare chatbot answers against KNOWN correct answers."
    Create a test set: 50 questions with verified answers.
    Run periodically. Track accuracy over time.

Signal 5: HUMAN FEEDBACK CORRELATION
    "Do users agree with the LLM's quality scores?"
    If LLM says "good answer" but users say "bad" -> LLM judge is unreliable.
---

### RAG Quality Monitor (`src/evaluation/rag_monitor.py`)

Tracks quality metrics over a rolling window:

---
Health Report Example:
{
    "status": "HEALTHY",
    "truthfulness": {
        "avg_groundedness": 0.78,     <- Above 0.7 = good
        "hallucination_rate": 0.03,   <- Below 5% = good
        "quality_pass_rate": 0.92     <- Above 90% = good
    },
    "retrieval": {
        "avg_confidence": 0.65,       <- Above 0.5 = finding relevant docs
        "low_confidence_rate": 0.08   <- Below 10% = good
    },
    "performance": {
        "avg_latency_ms": 4200,
        "cache_hit_rate": 0.45
    }
}

Health thresholds:
- HEALTHY: groundedness >= 0.7 AND hallucination rate < 5%
- WARNING: groundedness >= 0.5 AND hallucination rate < 15%
- CRITICAL: anything worse
---

### Ground Truth Testing (`TruthfulnessScorer`)

The most objective way to measure accuracy:

```python
from core.rag_monitor import truthfulness

# Define known correct answers
truthfulness.add_ground_truth(
    "How many leave days?", "20 days per year", category="leave"
)
truthfulness.add_ground_truth(
    "Remote work limit?", "3 days per week", category="remote"
)
truthfulness.add_ground_truth(
    "Expense receipt threshold?", "$50", category="expenses"
)

# Run evaluation against the chatbot
results = truthfulness.run_evaluation(chatbot)
# Returns: accuracy per category, list of failures, detailed scores
```

Run this weekly. Track accuracy over time. If accuracy drops after adding
new documents or changing the model, you know something went wrong.

---

## 17. Graceful Degradation

### The Problem

When the system encounters a query it can't answer confidently, what happens?

Bad: Crash with a Python traceback
Bad: Hallucinate a confident-sounding wrong answer
Bad: Generic "I don't know"
Good: Explain WHY it can't answer and suggest what to do instead

### Failure Types and Responses (`GracefulDegradation`)

| Failure | Trigger | Response |
|---------|---------|----------|
| LOW_CONFIDENCE | Retrieval score below threshold | "I found some documents but none are closely related. Try rephrasing or contact HR directly." |
| HALLUCINATION_DETECTED | Groundedness score < 0.4 | "I generated an answer but it wasn't well-supported by documents. Rather than risk incorrect info, please check policy X." |
| GUARDRAIL_BLOCKED | Content safety filter triggered | "Your question was flagged by content safety. Please rephrase to focus on HR topics." |
| NO_DOCUMENTS | Vector store is empty | "No documents have been ingested yet. Add documents and run ingestion." |
| MODEL_ERROR | Ollama connection failed | "Error generating response. Check that ollama is running." |
| TIMEOUT | Response took too long | "Response took too long. Try a shorter question." |

### Why This Matters

Every failure response:
1. Explains WHAT went wrong (not just "error")
2. Explains WHY (retrieval score, groundedness score)
3. Suggests WHAT TO DO (rephrase, contact HR, check ollama)
4. Returns the same response format as successful answers (no special handling needed)

---

## 18. Semantic Caching

### The Problem with Our Current Cache

Our production API has an exact-match cache (MD5 hash of the question text).

---
"How many leave days?"           -> cached ✓
"How many vacation days?"        -> CACHE MISS (different text, same meaning)
"how many leave days"            -> CACHE MISS (different capitalization)
"What is the annual leave?"      -> CACHE MISS (different wording)
---

All four questions mean the same thing, but only the first one gets a cache hit.

### Semantic Cache (`SemanticCache`)

Instead of hashing the text, we embed the question into a vector and find the
nearest cached question by cosine similarity:

---
"How many leave days?"           -> vector [0.4, 0.6, ...] -> cached ✓
"How many vacation days?"        -> vector [0.4, 0.5, ...] -> SEMANTIC HIT (similar vector!)
"how many leave days"            -> vector [0.4, 0.6, ...] -> SEMANTIC HIT
"What is the annual leave?"      -> vector [0.3, 0.6, ...] -> SEMANTIC HIT
---

### Impact on Cache Hit Rate

| Cache Type | Expected Hit Rate | Latency |
|------------|-------------------|---------|
| Exact match (MD5) | 40-50% | <1ms |
| Semantic (embeddings) | 60-80% | ~50ms (embedding computation) |

The 50ms embedding cost is negligible compared to the 5-15 seconds saved by
avoiding a full LLM generation.

### Configuration

```python
# Similarity threshold controls how "similar" a question must be
# 0.85 = very similar (safe, fewer false hits)
# 0.75 = somewhat similar (more hits, risk of wrong cached answer)
# 0.95 = nearly identical (very safe, fewer hits)
```

### Do We Need Redis?

For our local setup: NO. The in-memory semantic cache works fine.

For production (50K employees): YES. Redis provides:
- Persistence (cache survives restarts)
- Distributed access (multiple API servers share one cache)
- TTL management (automatic expiry)
- Memory management (eviction policies)

Redis doesn't do semantic matching natively, but you can store vectors in Redis
and use Redis Vector Search (RediSearch) for semantic cache lookups.

---

# Part 4: Security

## 19. The 15 AI Security Threats

These are specific to AI/LLM applications (OWASP Top 10 for LLMs + enterprise threats):

| # | Threat | Severity | What It Is |
|---|--------|----------|------------|
| 1 | Prompt Injection | CRITICAL | Attacker tricks LLM into ignoring its instructions |
| 2 | Malicious Usability | HIGH | Using the AI for unintended harmful purposes |
| 3 | Jailbreaking | CRITICAL | Bypassing the model's safety training |
| 4 | Hallucinations | HIGH | Model confidently generates false information |
| 5 | Serialization Attacks | MEDIUM | Malicious code in model/document files |
| 6 | Bias | MEDIUM | Systematically unfair outputs for certain groups |
| 7 | Model DoS | HIGH | Overwhelming the model with requests |
| 8 | Excessive Agency | LOW | AI takes actions beyond its intended scope |
| 9 | Model Inversion | MEDIUM | Extracting training data from the model |
| 10 | Guardrail Evasion | HIGH | Techniques to bypass safety filters |
| 11 | Improper Output Handling | HIGH | Unsanitized model output causes XSS/injection |
| 12 | Model Theft | MEDIUM | Unauthorized access to model weights |
| 13 | Model Poisoning | MEDIUM | Corrupting the model via malicious training data |
| 14 | 3rd Party Data | MEDIUM | Risks from external/untrusted data sources |
| 15 | Supply Chain | HIGH | Vulnerabilities in libraries, models, tools |

---

## 20. How We Protect Against Each Threat

| # | Threat | Our Defense | What's Missing |
|---|--------|-------------|----------------|
| 1 | Prompt Injection | Guardrail regex + system prompt rules + optional LLM classifier | No canary tokens, no indirect injection scanning |
| 2 | Malicious Usability | 6-category content filter (all HIGH) | No AI-powered classifier (Llama Guard) |
| 3 | Jailbreaking | System prompt guardrails + prompt attack regex | No multi-layer defense, no output classifier |
| 4 | Hallucinations | RAG grounding + LLM-as-judge + iterative refinement + source attribution (4 layers) | Same model for generation and evaluation |
| 5 | Serialization | Pickle detection (extension + magic bytes) + SafeTensors recommendation | No document malware scanning |
| 6 | Bias | Not addressed | Need bias evaluation datasets |
| 7 | Model DoS | Rate limiting (30/min per user) + auth required | No token-level limiting |
| 8 | Excessive Agency | Zero risk — model has NO tool access (read-only RAG) | N/A |
| 9 | Model Inversion | PII anonymization on input | No output PII scanning |
| 10 | Guardrail Evasion | Unicode normalization (homoglyph defense) + input sanitization | No adversarial testing |
| 11 | Improper Output | Output guardrails + Pydantic response models | No HTML sanitization |
| 12 | Model Theft | Auth required for all endpoints | Model files unencrypted on disk |
| 13 | Model Poisoning | Not addressed | Need training data validation |
| 14 | 3rd Party Data | Documents are local, approved sources list | No document provenance tracking |
| 15 | Supply Chain | Pinned dependency versions | No `pip audit`, no model integrity verification |

### Our Strongest Defense: Anti-Hallucination (4 Layers)

---
Layer 1: RAG Grounding
    System prompt: "ONLY use information from context documents"
    Model only sees retrieved chunks, not general knowledge

Layer 2: LLM-as-Judge Evaluation
    Every answer scored on groundedness (0.0-1.0)
    "Is every claim supported by the retrieved context?"

Layer 3: Iterative Refinement
    If score < 0.6, automatically refine query and retry

Layer 4: Source Attribution
    Every answer includes which document(s) it came from
    Makes hallucinations auditable
---

### Alternative Security Approaches

| Threat | Our Approach | Better Alternative (Production) |
|--------|--------------|---------------------------------|
| Content safety | Regex + optional LLM | Meta Llama Guard 3 (purpose-built) |
| Prompt injection | System prompt rules | Input classifier + canary tokens |

| PII detection | Regex patterns | Microsoft Presidio (50+ PII types) |
| Bias | Not addressed | BBQ/WinoBias evaluation datasets |
| Supply chain | Pinned versions | `pip audit` + Dependabot + SBOM |

---

## 21. Bedrock Guardrails Equivalent

Our guardrails mirror the AWS Bedrock Guardrails minimum configuration:

---
AWS Bedrock Guardrails                 Our Local Equivalent
----------------------                 --------------------
Content Filters:                       src/guardrails/content_safety.py:
  SEXUAL     -> HIGH                     SEXUAL     -> HIGH
  VIOLENCE   -> HIGH                     VIOLENCE   -> HIGH
  HATE       -> HIGH                     HATE       -> HIGH
  INSULTS    -> HIGH                     INSULTS    -> HIGH
  MISCONDUCT -> HIGH                     MISCONDUCT -> HIGH
  PROMPT_ATTACK -> HIGH                  PROMPT_ATTACK -> HIGH

PII Filters:                           PII handling:
  Anonymize                              Anonymize: EMAIL, PHONE, SSN, ADDRESS, NAME
  Block                                  Block: CREDIT_CARD, PASSPORT, BANK_ACCOUNT

Applied to:                            Applied to:
  Input (before LLM)                     Input: guardrails.check_input()
  Output (after LLM)                     Output: guardrails.check_output()
---

All filter levels configurable in `config.yaml`.

---

# Part 5: Models and Training

## 22. Models — What We Use and Alternatives

### Our Model Stack

| Role | Model | Why We Chose It |
|------|-------|-----------------|
| Chat/Generation | Mistral 7B | Good quality, runs on 8GB RAM, fast |
| Evaluation | Mistral 7B (same, different prompt) | Simplicity — one model for everything |
| Embeddings | nomic-embed-text | 768 dimensions, good quality, small |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | Trained on MS MARCO, accurate |

### Alternative Model Stacks

| Scenario | Generator | Embeddings | Why |
|----------|-----------|------------|-----|
| Better quality | Llama 3.1 8B | BGE-large-en-v1.5 | Stronger reasoning + better embeddings |
| Limited hardware | Phi 3.5 (3.8B) | all-MiniLM-L6-v2 | Runs on 8GB RAM total |
| Multilingual | Qwen 2.5 7B | multilingual-e5-large | Non-English document support |
| Maximum quality | Gemma 2 9B | E5-large-v2 | Best open-source quality |
| After fine-tuning | Your custom model | nomic-embed-text | Domain-specialized |

### How to Switch Models

Edit `config.yaml` (no code changes needed):

```yaml
models:
  generator: "llama3.1"        # Change from "mistral"
  embeddings: "nomic-embed-text" # Keep or change
```

Then: `ollama pull llama3.1` and restart the chatbot.

---

## 23. Fine-Tuning In Depth

### What Fine-Tuning Is

Taking a general model and training it further on YOUR data to make it a domain expert.

---
General Mistral 7B -> Fine-tune on HR Q&A -> HR Policy Expert Mistral
---

### Three Types of Fine-Tuning

| Type | What Changes | VRAM | When to Use |
|------|--------------|------|-------------|
| Full Fine-Tuning | ALL 7B parameters | 80-160GB | Enterprise GPUs only |
| LoRA | Adds small adapters (~0.1%) | 12-24GB | Good GPU (RTX 3090) |
| QLoRA (what we use) | LoRA + 4-bit quantization | 6-12GB | Consumer GPU or Colab |

### QLoRA Step by Step

---
1. Load base model in 4-bit (14GB -> 3.5GB)
2. Add LoRA adapter layers (~7M trainable params out of 7B)
3. Train ONLY the adapters on your data
4. Save adapters (~50-200MB, not the full 4GB model)
5. Merge adapters + base model
6. Convert to GGUF format
7. Import into ollama
8. Change one line in config.yaml
---

### When to Fine-Tune vs When to Use RAG

| Situation | RAG | Fine-Tune | Both |
|-----------|-----|-----------|------|
| Answer from specific documents | YES | - | - |
| Change writing style/tone | - | YES | - |
| Teach domain terminology | - | YES | - |
| Keep answers up-to-date | YES | - | - |
| Reduce hallucination | - | - | YES |
| Format outputs consistently | - | YES | - |
| Handle company Q&A patterns | - | - | YES |

### Fine-Tuning for Different Scenarios

| Scenario | Training Data | Expected Result |
|----------|---------------|-----------------|
| HR policy chatbot (our project) | 500 HR Q&A pairs + Alpaca 52K | Answers in company tone, uses HR terminology |
| Medical Q&A | PubMedQA + 1000 clinical Q&A | Clinically accurate, uses medical terms |
| Legal document analysis | LegalBench + 500 contract Q&A | Understands legal language and structure |
| Customer support | Bitext dataset + 1000 support tickets | Matches support team's tone and process |
| Code assistant | GitHub Code + 2000 code Q&A | Writes better code, explains clearly |
| Financial analysis | FinGPT + 500 financial Q&A | Understands financial terminology |

### Hyperparameter Guide

| Parameter | Conservative | Balanced | Aggressive |
|-----------|--------------|----------|------------|
| LoRA rank (r) | 8 | 16 | 64 |
| LoRA alpha | 16 | 32 | 128 |
| Learning rate | 1e-4 | 2e-4 | 5e-4 |
| Epochs | 1-2 | 3 | 5-10 |
| Batch size | 1 | 2-4 | 8+ |

Start conservative. Increase if the model isn't learning enough.
Decrease epochs if the model memorizes instead of generalizing.

---

## 24. Training Data Sources

### Complete Source Comparison

| Source | Type | Size | Quality | Format Ready | Best For |
|--------|------|------|---------|--------------|----------|
| Hugging Face | Curated datasets | 1K-1M+ | High | YES | Fine-tuning |
| Kaggle | Structured data | Varies | Medium-High | NO | Tabular/niche |
| Common Crawl | Raw web crawl | Petabytes | Low | NO | Pre-training only |
| Project Gutenberg | Classic books | 70K books | High | NO | Literary style |
| Wikipedia | Encyclopedia | 6.7M articles | High | NO | Factual knowledge |
| Your own docs | Custom | 50-10K+ | Highest | NO | Domain-specific |
| SQuAD | Q&A benchmark | 150K | Gold standard | YES | Evaluate comprehension |
| MS MARCO | Search benchmark | 1M+ | Gold standard | YES | Evaluate retrieval |
| HotpotQA | Multi-hop Q&A | 113K | Gold standard | YES | Evaluate reasoning |
| BEIR | 18 benchmarks | Varies | Gold standard | YES | Cross-domain eval |
| PubMed | Medical literature | 35M+ | Professional | NO | Medical domain |
| arXiv | Scientific papers | 2.3M+ | Professional | NO | Scientific domain |
| SEC EDGAR | Financial filings | 20M+ | Professional | NO | Financial domain |

### How Data Gets Into the System

---
RAG DOCUMENTS (actual files)          TRAINING DATA (Python pulls)
----------------------------          ----------------------------
You place files in data/documents/    Python downloads from Hugging Face
  |- policy.pdf                       dataset = load_dataset("tatsu-lab/alpaca")
  |- handbook.docx                    # Data in RAM, used by trainer
  L- rules.txt                        # Or save: dataset.to_json("train.jsonl")
---

RAG documents = actual files you provide. Training data = pulled via Python.

Full details: [TRAINING_DATA_GUIDE.md](TRAINING_DATA_GUIDE.md)

---

# Part 6: Production and Beyond

## 25. Production Architecture

### What's Running Locally (Everything Works Today)

| Feature | Implementation | File |
|---------|----------------|------|
| JWT Authentication | Local user store, 3 roles | `src/api/auth.py` |
| Role-Based Access | employee < hr_admin < admin | `src/api/server.py` |
| Audit Logging | JSONL file, every Q&A recorded | `src/api/audit.py` |
| Monitoring Metrics | Request counts, latency, quality | `src/api/monitoring.py` |
| Response Caching | In-memory, 1-hour TTL | `src/api/server.py` |
| Rate Limiting | 30 req/min per user | `src/api/server.py` |
| Content Safety | 6 categories, all HIGH, input + output | `src/guardrails/content_safety.py` |
| PII Detection | Anonymize + block modes | `src/guardrails/content_safety.py` |
| Input Sanitization | Null bytes, homoglyphs, control chars | `src/guardrails/model_governance.py` |
| Model Integrity | SHA-256 checksums, pickle blocking | `src/guardrails/model_governance.py` |
| Supply Chain | Approved sources validation | `src/guardrails/model_governance.py` |
| Hybrid Search | Vector + BM25 + re-ranking | `src/retrieval/hybrid.py` |
| Streaming | Token-by-token output | `src/chatbot.py` |
| Retrieval Confidence | Refuses low-confidence answers | `src/chatbot.py` |
| Central Config | All settings in config.yaml | `config.yaml` |

### What Changes for Real Production

| Local Component | Production Replacement |
|-----------------|------------------------|
| Ollama (single request) | vLLM on GPU servers (concurrent batching) |
| ChromaDB (file-based) | PostgreSQL + pgvector on AWS RDS |
| In-memory cache | Redis cluster |
| JSON user store | Corporate SSO/OIDC (Okta, Azure AD) |
| JSONL audit file | PostgreSQL or Elasticsearch |
| In-memory metrics | Prometheus + Grafana |
| Single process | Multi-worker (Python) or Spring Boot (Java) |
| Single machine | AWS EC2 Auto Scaling + ALB |

Full details: [PRODUCTION_ARCHITECTURE.md](PRODUCTION_ARCHITECTURE.md)

---

## 26. Project Structure and Configuration

### File Structure

```text
local-rag-chatbot/
├── src/
│   ├── chatbot.py                 # Core chatbot engine
│   ├── config.py                  # Configuration loader (reads config.yaml)
│   ├── ingestion/                 # Document ingestion pipeline
│   │   └── pipeline.py            # Ingestion logic
│   ├── retrieval/                 # Search and retrieval
│   │   └── hybrid.py              # Hybrid search (vector + BM25) + re-ranking
│   ├── generation/                # LLM generation and prompts
│   │   └── prompts.py             # Centralized prompt management (12 templates)
│   ├── guardrails/                # Content safety and governance
│   │   ├── content_safety.py      # Content safety, PII, prompt attack
│   │   └── model_governance.py    # Model integrity, pickle blocking, supply chain
│   ├── evaluation/                # Quality evaluation and feedback
│   │   ├── feedback.py            # Human feedback collection + training data export
│   │   └── rag_monitor.py         # Quality monitoring, truthfulness, semantic cache
│   ├── memory/                    # Conversation memory
│   │   └── memory_bank.py         # Persistent conversation memory
│   └── api/                       # Production API
│       ├── server.py              # FastAPI with auth + audit + monitoring
│       ├── auth.py                # JWT authentication + RBAC
│       ├── audit.py               # Audit logging
│       └── monitoring.py          # Metrics and monitoring
├── data/
│   ├── documents/                 # Your source documents
│   └── training/                  # Fine-tuning training data
├── notebooks/
│   ├── lessons/                   # 18 teaching scripts (run in order)
│   │   ├── 01_basic_chat.py ... 06_huggingface.py   # Foundation (Lessons 1-6)
│   │   ├── 07_memory_bank.py ... 08_prompt_engineering.py # Intelligence (7-8)
│   │   ├── 09_guardrails.py ... 11_input_sanitization.py  # Security (9-11)
│   │   ├── 12_hybrid_search.py ... 13_streaming.py      # Performance (12-13)
│   │   ├── 14_fine_tuning.py ... 15_human_feedback.py   # Improvement (14-15)
│   │   ├── 16_rag_monitoring.py ... 18_configuration.py # Production (16-18)
│   └── colab_finetune.ipynb       # QLoRA (Google Colab)
├── scripts/
│   ├── finetune.py                # QLoRA (local GPU)
│   └── training_validator.py      # Data filtering, validation, benchmarking
├── tests/                         # Test suite
├── docs/                          # Documentation
│   ├── COMPLETE_GUIDE.md          # This file
│   ├── LESSON_PLAN.md             # Curriculum (18 lessons)
│   ├── ARCHITECTURE.md            # System architecture
│   ├── PRODUCTION_ARCHITECTURE.md # Production deployment guide
│   ├── SECURITY.md                # Cybersecurity & AI threats
│   ├── ENGINEERING_REVIEW.md      # Code audit & optimization
│   ├── HARDWARE_REQUIREMENTS.md   # Hardware specs
│   ├── TRAINING_DATA_GUIDE.md     # Training data sources
│   └── QA.md                      # 29 frequently asked questions
├── pyproject.toml                 # Project metadata
├── config.yaml                    # Central configuration
├── requirements.txt               # Core dependencies
├── requirements-finetune.txt      # Fine-tuning dependencies
└── requirements-prod.txt          # Production API dependencies
```

### Full Request Flow (12 Steps)

---
1. User types question
2. Input Sanitization (null bytes, homoglyphs, control chars, length limit)
3. Input Guardrails (6 content filters + PII + prompt attack detection)
     └── Blocked? -> Return safety message
4. Query Cleaning (whitespace normalization)
5. Hybrid Retrieval (Vector + BM25 -> Rank Fusion -> Cross-Encoder Re-Rank)
     └── Low confidence? -> "I don't have relevant documents"
6. Prompt Assembly (Persona + Guardrails + Task + Context + History)
7. LLM Generation (Mistral 7B or fine-tuned model, with streaming)
8. Output Guardrails (content safety + PII leakage + prompt leakage)
     └── Unsafe? -> Block response
9. Quality Evaluation (LLM-as-judge: relevance + groundedness)
     └── Below 0.6? -> Refine query -> Go to step 5 (once)
10. Save to Memory Bank (buffer + periodic summarization + fact extraction)
11. Audit Log (user, question, answer, scores, timing)
12. Return: answer + sources + scores + PII flags + retrieval confidence
---

## 27. Quick Reference — All Commands

```bash
# --- SETUP --------------------------------------------------
ollama pull mistral && ollama pull nomic-embed-text
python -m venv venv && source venv/Scripts/activate
pip install -r requirements.txt

# --- LESSONS (18 lessons, run in order) ---------------------
python notebooks/lessons/01_basic_chat.py
python notebooks/lessons/02_embeddings.py
python notebooks/lessons/03_ingest_documents.py
python notebooks/lessons/04_rag_chatbot.py
python notebooks/lessons/05_evaluation.py
python notebooks/lessons/06_huggingface.py
python notebooks/lessons/07_memory_bank.py
python notebooks/lessons/08_prompt_engineering.py
python notebooks/lessons/09_guardrails.py
python notebooks/lessons/10_pii_detection.py
python notebooks/lessons/11_input_sanitization.py
python notebooks/lessons/12_hybrid_search.py
python notebooks/lessons/13_streaming.py
python notebooks/lessons/14_fine_tuning.py
python notebooks/lessons/15_human_feedback.py
python notebooks/lessons/16_rag_monitoring.py
python notebooks/lessons/17_api_and_auth.py
python notebooks/lessons/18_configuration.py

# --- CORE CHATBOT -------------------------------------------
python -m src.chatbot              # Terminal (streaming by default)
python -m src.chatbot --web        # Web interface (Gradio)

# --- DEMOS --------------------------------------------------
python -m src.memory.memory_bank          # Memory Bank demo
python -m src.generation.prompts          # Prompt Assembler demo

# --- FINE-TUNING --------------------------------------------
pip install -r requirements-finetune.txt
python scripts/finetune.py         # Local GPU
# OR: Upload notebooks/colab_finetune.ipynb to Google Colab

# --- PRODUCTION API -----------------------------------------
pip install -r requirements-prod.txt
python -m src.api.server           # http://localhost:8000/docs

# --- CONFIGURATION ------------------------------------------
# Edit config.yaml to change any setting:
#   models, retrieval, evaluation, memory, guardrails, API, auth
```

---

## Documentation Index

| Document | Pages | What It Covers |
|----------|-------|----------------|
| This file (COMPLETE_GUIDE.md) | ~500 lines | Everything from concepts to production |
| [LESSON_PLAN.md](LESSON_PLAN.md) | ~860 lines | Step-by-step curriculum, 18 lessons |
| [ARCHITECTURE.md](ARCHITECTURE.md) | ~320 lines | System design, components, data flow |
| [SECURITY.md](SECURITY.md) | ~1200 lines | 15 AI threats, 8 security layers, governance |
| [ENGINEERING_REVIEW.md](ENGINEERING_REVIEW.md) | ~500 lines | Code audit, 5 fixes, scorecard |
| [TRAINING_DATA_GUIDE.md](TRAINING_DATA_GUIDE.md) | ~800 lines | 13 data sources, benchmarks, comparison |
| [HARDWARE_REQUIREMENTS.md](HARDWARE_REQUIREMENTS.md) | ~200 lines | Local, Colab, production specs |
| [PRODUCTION_ARCHITECTURE.md](PRODUCTION_ARCHITECTURE.md) | ~300 lines | 50K employee deployment guide |
| [QA.md](QA.md) | ~1800 lines | 29 detailed Q&A from the learning journey |
| [TEACHER_NOTES.md](TEACHER_NOTES.md) | ~500 lines | Every lesson explained with analogies |
