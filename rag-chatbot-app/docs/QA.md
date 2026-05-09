# Q&A — Frequently Asked Questions

A collection of real questions asked during the learning process,
with detailed answers that explain the concepts behind this project.

---

## Table of Contents

1. [What is the use of Hugging Face in this project?](#1-what-is-the-use-of-hugging-face-in-this-project)
2. [Do we have a Memory Bank in our project?](#2-do-we-have-a-memory-bank-in-our-project)
3. [Do we have a Prompt Assembler in our project?](#3-do-we-have-a-prompt-assembler-in-our-project)
4. [What is fine-tuning? What does QLoRA mean?](#4-what-is-fine-tuning-what-does-qlora-mean)
5. [How to fine-tune to the best possible way?](#5-how-to-fine-tune-to-the-best-possible-way)
6. [If I pull a large dataset from Hugging Face, what fine-tuning do I need?](#6-if-i-pull-a-large-dataset-from-hugging-face-what-fine-tuning-do-i-need)
7. [Can we use Google Colab to train larger models like Llama or Gemma?](#7-can-we-use-google-colab-to-train-larger-models-like-llama-or-gemma)
8. [Is this project ready for a 50K employee MNC getting 5000 questions/month?](#8-is-this-project-ready-for-a-50k-employee-mnc-getting-5000-questionsmonth)
9. [What is the difference between src/ and notebooks/lessons/?](#9-what-is-the-difference-between-src-and-notebookslessons)
10. [We trained on Colab but it didn't contain LangChain, RAG, or Memory Bank - how does this work?](#10-we-trained-on-colab-but-it-didnt-contain-langchain-rag-or-memory-bank-how-does-this-work)
11. [Colab and local are separate systems - how does it work?](#11-colab-and-local-are-separate-systems-how-does-it-work)
12. [What is vLLM?](#12-what-is-vllm)
13. [Can we use Java to create this project? Is there a LangChain for Java?](#13-can-we-use-java-to-create-this-project-is-there-a-langchain-for-java)
14. [What is a Context Gatherer? Do we have one in our project?](#14-what-is-a-context-gatherer-do-we-have-one-in-our-project)
15. [Is the context gathering system helpful in multimodal?](#15-is-the-context-gathering-system-helpful-in-multimodal)
16. [What are the hardware requirements for local, Colab, and production?](#16-what-are-the-hardware-requirements-for-local-colab-and-production)
17. [What dataset source should I use - Hugging Face, Kaggle, Common Crawl, Gutenberg, or Wikipedia?](#17-what-dataset-source-should-i-use)
18. [Do I pull data using Python or use actual files?](#18-do-i-pull-data-using-python-or-use-actual-files)
19. [What about SQuAD, MS MARCO, HotpotQA, BEIR, PubMed, arXiv, SEC EDGAR?](#19-what-about-squad-ms-marco-hotpotqa-beir-pubmed-arxiv-sec-edgar)
20. [Which datasets run locally vs need Colab?](#20-which-datasets-run-locally-vs-need-colab)
21. [What kind of model do we get after training? Do we have Fine-Tuned Mistral + RAG?](#21-what-kind-of-model-do-we-get-after-training)
22. [What are hybrid search and re-ranking?](#22-what-are-hybrid-search-and-re-ranking)
23. [What are OpenClaw and NemoClaw?](#23-what-are-openclaw-and-nemoclaw)
24. [What are the 4 levels of AI systems?](#24-what-are-the-4-levels-of-ai-systems)
25. [Can this system handle 1M documents?](#25-can-this-system-handle-1m-documents)
26. [Can the fine-tuning pipeline handle 1M training examples?](#26-can-the-fine-tuning-pipeline-handle-1m-training-examples)
27. [What do we have for observability?](#27-what-do-we-have-for-observability)
28. [Why LangChain and not LlamaIndex?](#28-why-langchain-and-not-llamaindex)
29. [Are we using LangGraph to manage the RAG pipeline?](#29-are-we-using-langgraph-to-manage-the-rag-pipeline)

---

## ## 1. What is the use of Hugging Face in this project?

Hugging Face plays THREE roles in this project:

### ### Role 1: Alternative Embedding Models

In `notebooks/lessons/02_embeddings.py`, we use Ollama's `nomic-embed-text` to convert text into vectors.
Hugging Face gives you another option — the `sentence-transformers` library with models like
`all-MiniLM-L6-v2`. Same job, different tool.

Why care? Hugging Face gives you hundreds of embedding models to choose from. Some are faster,
some are more accurate, some are specialized for specific languages or domains. Ollama gives
you maybe 5-10 options.

### ### Role 2: Content Safety / Classification (Guardrails)

In the AWS reference system, Bedrock Guardrails filters harmful content. Locally, we can't use
Bedrock. So in `notebooks/lessons/06_huggingface.py`, we use Hugging Face's pre-trained classifiers:

- `distilbert` for sentiment analysis — detecting negative/toxic content
- `bart-large-mnli` for zero-shot classification — routing queries to categories like
  "HR Policy" vs "Technical Documentation" without any training

This is our local version of content safety filtering.

### ### Role 3: The Model Zoo

Hugging Face hosts 500,000+ free models. Think of it as a grocery store for AI. Ollama is the
convenience store — fewer options, but dead simple. When you need something specialized (named
entity recognition, summarization, translation, code generation), Hugging Face is where you go.

### ### Summary: Who Does What

| Task | Who Does It |
| :--- | :--- |
| Chat / Answer generation | Ollama (Mistral) |
| Document embeddings | Ollama (nomic-embed-text) |
| Vector storage & search | ChromaDB |
| Content safety | Hugging Face classifiers |
| Specialized NLP tasks | Hugging Face models |
| Orchestration | LangChain |

If you removed Hugging Face entirely, the chatbot would still work. You'd just lose the content
classification and the option to swap in different embedding models. It's the toolkit expansion,
not the foundation.

---

## ## 2. Do we have a Memory Bank in our project?

Yes. The Memory Bank (`src/memory/memory_bank.py`) is a three-layer persistent conversation memory system.

### ### What It Replaced

Originally, the chatbot stored conversations in a simple Python dictionary:

```python
self.conversations: dict[str, list] = {}
```

This had three fatal flaws:
1. Died on restart — close the app, memory is gone forever
2. No compression — long conversations eat up the entire context window
3. No learning — doesn't extract or remember key facts

### ### The Three Layers

| Layer | What It Stores | How It Works |
| :--- | :--- | :--- |
| Buffer | Last 6 exchanges in full detail | Like your short-term memory |
| Summary | LLM-compressed older exchanges | Like your long-term memory |
| Key Facts | Important facts extracted by LLM | Like notes you write down |

### ### How It Works

---

Exchange 1-6: Stored in buffer (full detail)
Exchange 7: Oldest exchanges get SUMMARIZED by the LLM
Summary is saved, buffer keeps only recent ones
Exchange 9: LLM extracts key facts from recent conversation
Exchange 10+: Cycle continues — buffer -> summary -> facts

---

Everything is saved to `memory_bank/` as JSON files. Restart the app? Memory is still there.

### ### AWS Reference Mapping

| AWS Reference | Our Local Version |
| :--- | :--- |
| S3-backed session storage | JSON files in `memory_bank/` folder |
| DynamoDB session management | File-based session per user |
| AppSync session context | Memory Bank's `get_context()` method |
| Session TTL / cleanup | `cleanup_old_sessions()` (30-day max) |

---

## ## 3. Do we have a Prompt Assembler in our project?

Yes. The Prompt Assembler (`src/generation/prompts.py`) centralizes all 12 prompt templates
into ONE module.

### ### The Problem It Solves

Before the Prompt Assembler, prompts were scattered across 5 files — 8+ prompts, all hardcoded.
Want to change how the AI cites sources? Hunt through every file. Want to add a safety rule?
Edit 5 places and hope you don't miss one.

### ### How It Works

Every prompt is assembled from reusable building blocks:

---

Final Prompt = Persona + Guardrails + Task Instructions + Context Block

---

| Building Block | What It Is | Example |
| :--- | :--- | :--- |
| Persona | Who the AI is | "You are a helpful document assistant" |
| Guardrails | Safety rules | "Do NOT invent information" |
| Task Instructions | What to do | "Score from 0.0 to 1.0" |
| Context Blocks | Dynamic content | `{context}`, `{history}` placeholders |
| Format Rules | Output style | "Be concise", "Respond with JSON only" |

Change a persona once -> every prompt using it updates automatically.
Add a guardrail once -> it's injected into every guarded prompt.

### ### Available Prompts

| Name | Used By | Purpose |
| :--- | :--- | :--- |
| `basic_chat` | notebooks/lessons/01 | Simple chat |
| `rag_simple` | notebooks/lessons/04 | RAG without history |
| `rag_with_history` | src/chatbot | RAG with memory |
| `rag_for_evaluation` | notebooks/lessons/05 | RAG for eval testing |
| `eval_combined` | src/chatbot | Relevance + groundedness |
| `eval_relevance` | notebooks/lessons/05 | Relevance only |
| `eval_groundedness` | notebooks/lessons/05 | Groundedness only |
| `refine_query` | notebooks/lessons/05 | Query improvement |
| `memory_summarize_new` | src/memory/memory_bank | Summarize conversation |
| `memory_summarize_update` | src/memory/memory_bank | Update existing summary |
| `memory_extract_facts` | src/memory/memory_bank | Extract key facts |
| `memory_demo_chat` | src/memory/memory_bank | Demo chat with history |

### ### Usage

```python
from generation.prompts import PromptAssembler

assembler = PromptAssembler(persona="default")

# Get any prompt by name
rag_prompt = assembler.get_template("rag")

# Build a prompt with variables
prompt = assembler.build("rag", context="...", question="How many leave days?")

# Change persona — all prompts using it update automatically
assembler.set_persona("hr")
```

---

## ## 4. What is fine-tuning? What does QLoRA mean?

### ### What Is Fine-Tuning?

When you downloaded Mistral 7B, you got a model trained on general internet text. It knows a
little about everything. Fine-tuning is taking that general model and training it further on
YOUR specific data so it becomes an expert in YOUR domain.

---

General Model (knows everything vaguely)
|
▼ Fine-tune on your data
|
Domain Expert Model (knows your stuff deeply)

---

It's like hiring a smart generalist and then training them specifically for your job.

### ### The Three Types of Fine-Tuning

| Type | What It Does | VRAM Needed | When to Use |
| :--- | :--- | :--- | :--- |
| Full Fine-Tuning | Updates ALL model parameters | 80-160GB+ | Enterprise GPUs (A100s). Almost nobody does this locally. |
| LoRA | Adds small trainable layers, freezes the rest | 12-24GB | Decent GPU (RTX 3090/4090) |
| QLoRA | LoRA but with 4-bit quantized base model | 6-12GB | Consumer GPU (RTX 3060+). This is what YOU should use. |

### ### What is QLoRA? (Broken Down)

**Q = Quantization**

The original Mistral 7B uses 16-bit numbers for each parameter.
That's 7 billion x 2 bytes = ~14GB just to load the model.
Quantization compresses those 16-bit numbers down to 4-bit numbers.
Now it's 7 billion x 0.5 bytes = ~3.5GB.
The model gets slightly less precise but fits in way less memory.

**LoRA = Low-Rank Adaptation**

Instead of updating all 7 billion parameters (impossible on your hardware), LoRA adds tiny
"adapter" layers — typically only 0.1-1% of the total parameters. Only these adapters get
trained. The original model stays frozen.

---

Original Mistral 7B (FROZEN, not changed)
7 billion parameters
___________________________________
|                                 |
| LoRA Adapters (TRAINABLE)        |
| ~7 million parameters (0.1%)     |
| These are the ONLY thing that   |
| gets updated during training    |
|_________________________________|

---

**QLoRA = Quantization + LoRA**

Load the base model in 4-bit (saves memory) + train only the LoRA adapters.
This is how you fine-tune a 7B model on a single consumer GPU.

---

Original Model: 7B params x 16-bit = 14GB VRAM needed
|
Quantize to 4-bit: 7B params x 4-bit = 3.5GB  <- fits on your GPU
|
Add LoRA adapters: ~7M trainable params (0.1% of total)
|
Train ONLY adapters on your data
|
Save adapters: ~50-200MB (not the full 4GB model)

---

## ## 5. How to fine-tune to the best possible way?

Three rules:

### ### Rule 1: Data Quality Over Quantity

500 perfect, domain-specific Q&A pairs beat 50,000 noisy generic ones.
Clean your data ruthlessly — remove duplicates, fix formatting, verify accuracy.

### ### Rule 2: Fine-Tune for Style, RAG for Content

Fine-tuning teaches the model HOW to answer (format, tone, terminology).
RAG provides WHAT to answer with (your actual documents).
Use both together for the best results.

| Situation | Use RAG | Use Fine-Tuning | Use Both |
| :--- | :--- | :--- | :--- |
| Answer from specific documents | YES | - | - |
| Change the model's writing style | - | YES | - |
| Teach domain-specific terminology | - | YES | - |
| Keep answers up-to-date | YES | - | - |
| Reduce hallucination on your domain | - | - | YES |
| Format outputs consistently | - | YES | - |

### ### Rule 3: Start Conservative, Iterate

Begin with LoRA rank 8, learning rate 2e-4, 3 epochs. If the model isn't learning enough,
increase rank. If it's memorizing instead of generalizing, reduce epochs.

| Parameter | Conservative | Balanced | Aggressive |
| :--- | :--- | :--- | :--- |
| LoRA rank (r) | 8 | 16 | 64 |
| LoRA alpha | 16 | 32 | 128 |
| Learning rate | 1e-4 | 2e-4 | 5e-4 |
| Epochs | 1-2 | 3 | 5-10 |

### ### Common Mistakes

1. **Too many epochs**: Model memorizes training data instead of learning patterns.
   Symptom: Training loss near 0, but weird answers on new questions. Fix: Use 1-3 epochs.
2. **Learning rate too high**: Training is unstable, loss spikes.
   Fix: Start at 2e-4, reduce if loss is erratic.
3. **Too little data**: Model doesn't learn enough.
   Fix: Need at least 100 examples for simple tasks, 1000+ for complex ones.
4. **Wrong format**: Training data format doesn't match how you'll use the model.
   Fix: If you'll use chat format, train on chat format.

### ### Dataset Size Guidelines

| Task | Minimum | Recommended | Diminishing Returns |
| :--- | :--- | :--- | :--- |
| Style/format change | 50-100 | 200-500 | 1,000+ |
| Domain Q&A | 200-500 | 1,000-5,000 | 10,000+ |
| Instruction following | 1,000 | 5,000-20,000 | 50,000+ |
| General capability | 10,000+ | 50,000-200,000 | 500,000+ |

---

## ## 6. If I pull a large dataset from Hugging Face, what fine-tuning do I need?

**Supervised Fine-Tuning (SFT) with QLoRA.** That's the answer for 90% of cases.

### ### What SFT Means

You give the model input/output pairs:
- Input: "What is the leave policy?"
- Output: "Employees get 20 days of annual leave per year..."

The model learns to produce outputs that look like your training examples.

### ### Popular Hugging Face Datasets

| Dataset | Size | Good For |
| :--- | :--- | :--- |
| `tatsu-lab/alpaca` | 52K examples | General instruction following |
| `teknium/OpenHermes-2.5` | 1M+ examples | High-quality conversations |
| `HuggingFaceH4/ultrachat_200k` | 200K chats | Conversational ability |
| `medalpace/medical_meadow_medqa` | Medical Q&A | Medical domain |
| `b-mc2/sql-create-context` | SQL generation | Database queries |

### ### How to Use

```python
from datasets import load_dataset
```
```python
# Load any dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train[:5000]")

# Use first 5000 examples (you don't need all of them)
```

The fine-tuning script (`scripts/finetune.py`) and Colab notebook
(`notebooks/colab_finetune.ipynb`) handle the rest automatically.

---

## 7. Can we use Google Colab to train larger models like Llama or Gemma?

Yes. Here's what fits on each Colab tier:

| Model | Parameters | Colab Free (T4 16GB) | Colab Pro (L4 24GB) | Colab Pro+ (A100 40GB) |
| :--- | :--- | :--- | :--- | :--- |
| Phi 3.5 | 3.8B | Easy | Easy | Easy |
| Mistral 7B | 7B | YES | YES | YES |
| Llama 3.1 8B | 8B | YES | YES | YES |
| Gemma 2 9B | 9B | Tight (reduce rank to 8) | YES | YES |
| Llama 3.1 70B | 70B | NO | NO | Tight |

### How to Use

1. Upload `notebooks/colab_finetune.ipynb` to https://colab.research.google.com
2. Go to Runtime -> Change runtime type -> T4 GPU
3. In Cell 3, change `BASE_MODEL` to your chosen model:
   ```python
   BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Llama
   BASE_MODEL = "google/gemma-2-9b-it"                  # Gemma
   ```
4. For Llama/Gemma: accept the license on Hugging Face and provide your HF token (Cell 4)
5. Run all cells
6. Download the GGUF file and import into ollama

### Expected Training Times on Colab T4

| Examples | Epochs | Time |
| :--- | :--- | :--- |
| 1,000 | 3 | ~15-30 minutes |
| 5,000 | 3 | ~1-2 hours |
| 50,000 | 3 | ~8-15 hours |

The free T4 GPU (16GB VRAM) handles 7B-8B models comfortably with QLoRA.
For Gemma 2 9B, reduce the LoRA rank from 16 to 8. For anything above 13B, you need Colab Pro.

---

## 8. Is this project ready for a 50K employee MNC getting 5000 questions/month?

### Short Answer: No. But the gap is fixable.

The current project is a learning tool. It works for 1 person on 1 machine. An MNC with
50,000 employees needs fundamentally different infrastructure.

### Gap Analysis

| Area | Current State | MNC Requirement | Gap |
| :--- | :--- | :--- | :--- |
| Concurrency | Single user, single thread | 50-200 concurrent users | CRITICAL |
| Throughput | ~1 question/30 seconds | ~170 questions/day, bursts of 50+ | CRITICAL |
| Availability | Runs on 1 laptop | 99.9% uptime | CRITICAL |
| Vector DB | ChromaDB (file-based) | Concurrent reads, backup | HIGH |
| LLM Serving | Ollama (1 request at a time) | GPU server with queuing | HIGH |
| Auth | None (now added locally) | Corporate SSO for 50K employees | HIGH |
| Audit | None (now added locally) | Compliance-grade logging | HIGH |

### What We Built Locally (Lesson 12)

| Feature | Local Implementation |
| :--- | :--- |
| Authentication | JWT tokens, local user store |
| Audit Logging | JSONL file, every Q&A recorded |
| Monitoring | In-memory metrics, dashboard endpoint |
| Response Caching | In-memory cache, 1-hour TTL |
| Rate Limiting | Per-user, 30 req/min |
| Role-Based Access | employee, hr_admin, admin |

### What Changes for Production

| Local Component | Production Replacement |
| :--- | :--- |
| Ollama | vLLM on GPU servers |
| ChromaDB | PostgreSQL + pgvector on AWS RDS |
| In-memory cache | Redis cluster |
| JSON user store | Corporate SSO/OIDC |
| JSONL audit file | PostgreSQL or Elasticsearch |
| In-memory metrics | Prometheus + Grafana |
| Single process | Multi-worker or Java/Spring Boot |
| Single machine | AWS EC2 Auto Scaling + ALB |

### Hardware Cost Estimate

| Option | Monthly Cost | Handles |
| :--- | :--- | :--- |
| Minimal (1 GPU) | ~$1,100/month | 170 questions/day, 20 concurrent |
| Production (2 GPUs, HA) | ~$7,200/month | 1,000+ questions/day, 200 concurrent |
| On-premises | ~$52,000 one-time | Break-even at ~7 months |

Full details: [PRODUCTION_ARCHITECTURE.md](PRODUCTION_ARCHITECTURE.md)

---

## 9. What is the difference between src/ and notebooks/lessons/?

Think of it like a cooking class.

### notebooks/lessons/ = The Practice Exercises

Each file teaches you ONE concept in isolation. They're standalone scripts you run, learn from,
and move on. You don't need them to run the actual chatbot. They exist purely for YOUR education.

| File | Teaches You |
| :--- | :--- |
| `01_basic_chat.py` | How to talk to an LLM via LangChain |
| `02_embeddings.py` | How text becomes searchable vectors |
| `03_ingest_documents.py` | How to load documents into a vector store |
| `04_rag_chatbot.py` | How RAG works (basic version) |
| `05_evaluation.py` | How to judge answer quality with LLM-as-judge |
| `06_huggingface.py` | What Hugging Face models can do |

### src/ = The Actual Product

This is the real chatbot that combines everything from the lessons into one working system.
If you were deploying this for real, you'd ship `src/`. You'd never ship
`notebooks/lessons/`.

| File | What It Does |
| :--- | :--- |
| `chatbot.py` | The complete RAG chatbot (ingestion + retrieval + generation + evaluation + guardrails) |
| `memory/memory_bank.py` | Persistent conversation memory (3 layers: buffer + summary + facts) |
| `generation/prompts.py` | Centralized prompt management (12 templates) |

### The Key Point

If you delete the entire `notebooks/lessons/` folder, the chatbot still works perfectly.
Lessons are training wheels. `src/` is the bicycle.

---
notebooks/lessons/   ->   "Here's how each part works individually"   (education)
src/                 ->   "Here's everything working together"        (product)
---

---

## 10. We trained on Colab but it didn't contain LangChain, RAG, or Memory Bank - how does this work?

This is the most important question. Pay close attention.

### Fine-tuning and RAG are TWO SEPARATE things. They do DIFFERENT jobs.

---
FINE-TUNING (Colab)                  RAG SYSTEM (Local)
-------------------                  ------------------
Changes the MODEL's brain            Changes what the model SEES

Teaches HOW to answer                Provides WHAT to answer with
(style, format, terminology)         (your actual documents)

Happens ONCE (training)              Happens EVERY question (retrieval)

Output: a new model file             Output: an answer with sources
(.gguf file)

Runs on GPU (Colab)                  Runs on your machine (ollama)
---

### The Complete Flow

---
Step 1: You fine-tune on Colab
        -> Produces a .gguf model file (just the AI brain, nothing else)

Step 2: You download the .gguf file to your machine

Step 3: You import it into ollama
        -> ollama create my-model -f Modelfile

Step 4: You change ONE LINE in src/chatbot.py:
        self.generator = ChatOllama(model="my-model")

Step 5: Now when the RAG system runs:
        - LangChain orchestrates everything (same as before)
        - ChromaDB searches your documents (same as before)
        - Memory Bank tracks conversation (same as before)
        - Prompt Assembler builds the prompt (same as before)
        - BUT the model generating the answer is YOUR fine-tuned model
          instead of generic Mistral
---

### The Analogy

The fine-tuned model is a **drop-in replacement** for the brain. Everything else (LangChain,
RAG, Memory Bank, Prompt Assembler) stays exactly the same. They don't care which model is
generating the answer — they just need a model that can take text in and produce text out.

Think of it like a car:
- Fine-tuning = upgrading the engine
- RAG / LangChain / Memory Bank = the steering wheel, GPS, and dashboard
- You can swap the engine without redesigning the car

### What the Colab Notebook Does NOT Contain (and why)

| Component | In Colab? | Why Not? |
| :--- | :--- | :--- |
| LangChain | No | LangChain is the orchestrator. It calls the model, not the other way around. |
| RAG / ChromaDB | No | RAG retrieves documents at query time. Training doesn't need documents. |
| Memory Bank | No | Memory is a runtime feature. Training is a one-time process. |
| Prompt Assembler | No | Prompts are assembled when the user asks a question, not during training. |
| Guardrails | No | Safety filtering happens at query time, not training time. |

The Colab notebook ONLY does one thing: make the model smarter at generating text.
Everything else is handled by the local system when the model is actually used.

---

## 11. Colab and local are separate systems - how does it work?

They're separate machines but connected by ONE file — the model.

---
   GOOGLE COLAB                          YOUR MACHINE

   Training data                         data/documents/
         ↓                               ChromaDB
  Fine-tune model                        LangChain
         ↓                               Memory Bank
 model.gguf (4GB)  ---- download ---->   ollama
                                         src/chatbot.py
 (GPU does the
  heavy lifting)                         (runs the chatbot)
---

### Why Two Systems?

1. **Colab has a GPU.** Your machine probably doesn't (or has a small one).
2. **Training needs a GPU.** It's computationally intensive — billions of math operations.
3. **Running the trained model does NOT need a GPU.** Ollama handles it on CPU just fine.
4. **You train ONCE on Colab, use the result FOREVER locally.**

### The Analogy

It's like sending your clothes to a dry cleaner:
- The cleaning happens at their facility (Colab)
- You pick up the clean clothes (model.gguf)
- You wear them at home (Ollama)

The dry cleaner doesn't need to know about your wardrobe. Your wardrobe doesn't need to know
about the cleaning process. They're connected by the clothes themselves.

### Step-by-Step Transfer Process

```bash
# 1. On Colab: Training produces model.gguf
#    Download it (or save to Google Drive first)

# 2. On your machine: Create a Modelfile
echo "FROM ./model.gguf" > Modelfile
echo "PARAMETER temperature 0.3" >> Modelfile

# 3. Import into Ollama
ollama create my-finetuned-model -f Modelfile

# 4. Test it
ollama run my-finetuned-model

# 5. Use in the chatbot (change one line in src/chatbot.py)
# self.generator = ChatOllama(model="my-finetuned-model", temperature=0.3)
```

---

## 12. What is vLLM?

### The Problem with Ollama in Production

Ollama is great for 1 person. But in the 50K employee scenario, Ollama processes ONE request
at a time. If 50 people ask questions simultaneously, 49 of them wait in line.

### What vLLM Does

vLLM (Very Large Language Model serving) is a production LLM server that handles MANY requests
at once using a technique called **continuous batching**.

### Comparison

| Feature | Ollama | vLLM |
| :--- | :--- | :--- |
| Designed for | Single user, local dev | Production, many users |
| Concurrent requests | 1 (queued) | 50+ (batched) |
| Throughput | ~2 requests/min | ~30-100 requests/min |
| API format | Custom | OpenAI-compatible (drop-in) |
| Setup complexity | Very easy | Moderate |
| GPU utilization | Low | High |

### How Continuous Batching Works

Without batching (Ollama):
---
Request 1 -> process -> done -> Request 2 -> process -> done -> Request 3 -> process -> done
Total time: 3x
---

With continuous batching (vLLM):
---
Request 1, 2, 3 -> process ALL together on GPU -> done
Total time: ~1.2x (not 3x)
---

vLLM groups multiple requests together and processes them in one GPU pass. The GPU does
almost the same amount of work whether it's processing 1 request or 10, because the
bottleneck is memory bandwidth, not computation.

### When to Use What

| Scenario | Use |
| :--- | :--- |
| Learning, development, testing | Ollama |
| 1-5 users | Ollama |
| 10+ concurrent users | vLLM |
| Production deployment | vLLM |

### Code Change Required

The switch from Ollama to vLLM is minimal. vLLM exposes an OpenAI-compatible API,
so LangChain can talk to it the same way:

```python
# Before (Ollama)
from langchain_ollama import ChatOllama
llm = ChatOllama(model="mistral", temperature=0.3)

# After (vLLM)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    base_url="http://vllm-server:8000/v1",
    model="mistral",
    temperature=0.3,
)
```
Two lines change. Everything else (RAG, Memory Bank, Prompt Assembler, evaluation) stays
exactly the same.

---

## 13. Can we use Java to create this project? Is there a LangChain for Java?

### Yes and yes.

**LangChain4j** is the Java equivalent of LangChain. It's a real project, actively maintained,
and used in production by many companies.

### Component Mapping: Python -> Java
| Python (What We Built) | Java Equivalent |
| :--- | :--- |
| LangChain | LangChain4j |
| FastAPI | Spring Boot |
| ChromaDB | pgvector with Spring Data JPA |
| Ollama (Python client) | LangChain4j Ollama integration |
| Gradio | React / Angular frontend |
| JWT auth (python-jose) | Spring Security |
| Audit logging | Spring AOP + JPA |
| Prometheus metrics | Micrometer (Spring Boot Actuator) |
| In-memory cache | Spring Cache + Redis |

### What the Same RAG Chain Looks Like in Java

```java
// Java (LangChain4j) — same concept, different syntax

// 1. Connect to the LLM (same as ChatOllama in Python)
ChatLanguageModel model = OllamaChatModel.builder()
    .baseUrl("http://localhost:11434")
    .modelName("mistral")
    .temperature(0.3)
    .build();

// 2. Connect to the embedding model (same as OllamaEmbeddings in Python)
EmbeddingModel embeddingModel = OllamaEmbeddingModel.builder()
    .baseUrl("http://localhost:11434")
    .modelName("nomic-embed-text")
    .build();

// 3. Connect to the vector store (same as ChromaDB in Python)
EmbeddingStore<TextSegment> store = ChromaEmbeddingStore.builder()
    .baseUrl("http://localhost:8000")
    .collectionName("local_rag")
    .build();

// 4. Create the retriever (same as vectorstore.as_retriever() in Python)
ContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
    .embeddingStore(store)
    .embeddingModel(embeddingModel)
    .maxResults(3)
    .build();

// 5. Build the RAG chain (same as prompt | llm | parser in Python)
Assistant assistant = AiServices.builder(Assistant.class)
    .chatLanguageModel(model)
    .contentRetriever(retriever)
    .build();

// 6. Ask a question
String answer = assistant.chat("How many leave days do I get?");
```

### The Concepts Are Identical

Everything you learned in this project applies directly to Java:

| Concept | Python | Java |
| :--- | :--- | :--- |
| RAG pipeline | LangChain chain | LangChain4j AiServices |
| Embeddings | OllamaEmbeddings | OllamaEmbeddingModel |
| Vector search | ChromaDB retriever | EmbeddingStoreContentRetriever |
| Prompt templates | ChatPromptTemplate | PromptTemplate / SystemMessage |
| Memory | Custom Memory Bank | ChatMemory (built into LangChain4j) |
| Evaluation | Custom LLM-as-judge | Custom (same pattern) |

### When to Use Java vs Python

| Factor | Python | Java |
| :--- | :--- | :--- |
| ML/AI ecosystem | Dominant (all libraries are Python-first) | Growing (LangChain4j is solid) |
| Fine-tuning | Python only (PyTorch, Hugging Face) | Not available (train in Python, serve in Java) |
| Web API | FastAPI (async, lightweight) | Spring Boot (enterprise-grade, battle-tested) |
| Concurrency | Async/await, multi-worker | Virtual threads, thread pools (stronger) |
| Enterprise adoption | Growing | Dominant |
| Team familiarity | Data science teams | Enterprise dev teams |

### The Practical Approach

The fine-tuning part stays in Python regardless — that's the ML ecosystem. But the chatbot
application itself (API server, RAG pipeline, auth, audit, monitoring) can absolutely be Java.

Many production systems use both:
- Python for training and model management
- Java/Spring Boot for the application layer serving users

This is a perfectly valid and common architecture pattern.

---

## Summary: How Everything Connects

---
                      THE COMPLETE PICTURE
____________________________________________________________________
|                                                                    |
|  TRAINING (one-time)              APPLICATION (runs continuously)  |
|  -------------------              -------------------------------  |
|                                                                    |
|  Hugging Face Dataset             User asks a question             |
|         ↓                                ↓                         |
|  QLoRA Fine-Tuning                Content Safety Check             |
|  (Colab or local GPU)                    ↓                         |
|         ↓                         Query Cleaning + PII             |
|  model.gguf file                         ↓                         |
|         ↓                         Vector Search (ChromaDB)         |
|  Import into ollama                      ↓                         |
|         ↓                         Prompt Assembly                  |
|  Model ready to use                      ↓                         |
|         |                         LLM Generation (Ollama/vLLM)     |
|         |                                ↓                         |
|         `--- model powers ---->   Quality Evaluation               |
|              this step                   ↓                         |
|                                   Save to Memory Bank              |
|                                          ↓                         |
|                                   Return Answer + Sources          |
|                                                                    |
|  Python: training                 Python or Java: application      |
|  Colab: GPU                       Local or AWS: serving            |
|  Once: done                       Always: running                  |
|____________________________________________________________________|
---

---

## 14. What is a Context Gatherer? Do we have one in our project?

### What It Is

A context gatherer is an intelligent system that EXPLORES multiple sources before answering,
instead of doing a single search. Think of the difference between:

- **Single-shot retrieval** (what our chatbot does): Ask one question -> get top 3 chunks -> answer
- **Context gathering** (advanced): Analyze the question -> decide what info is needed ->
  search multiple times -> follow cross-references -> combine everything -> answer

### Analogy

Single-shot retrieval is like asking a librarian "find me a book about leave policy" —
they check one shelf and hand you the closest match.

Context gathering is like asking a research assistant "figure out our complete leave policy" —
they check the employee handbook, cross-reference with the benefits guide, look up recent
policy updates, check if there are department-specific exceptions, and THEN give you a
comprehensive answer.

### Do We Have it?

Partially. Our chatbot does context gathering at a basic level:

| Feature | Our Project | Full Context Gatherer |
| :--- | :--- | :--- |
| Vector search | YES (top 3 chunks) | YES (multiple queries) |
| Follow cross-references | NO | YES |
| Multi-step retrieval | NO | YES (search -> read -> search again) |
| Decide what to search for | NO (uses raw question) | YES (analyzes question first) |
| Combine multiple sources | NO (just concatenates chunks) | YES (synthesizes) |
| Memory of what was already found | NO | YES |

### How to Add It (LangChain Agents)

```python
# Conceptual — not in our project yet
from langchain.agents import create_react_agent

tools = [
    vector_search_tool,      # Search documents by meaning
    document_reader_tool,    # Read a full document
    metadata_search_tool,    # Search by title/date/author
]

agent = create_react_agent(llm, tools, prompt)
# The agent decides: "I need to search for X, then read document Y,
# then search for Z to get the full picture"
```

This is an advanced topic covered in the "Where to Go From Here" section of the lesson plan.

---

## 15. Is the context gathering system helpful in multimodal?

### Short Answer: It's not just helpful — it's ESSENTIAL.

### Why?

When your chatbot only handles text, finding context is straightforward — vector search on
text chunks. But when you add images, audio, video, PDFs with charts, spreadsheets — the
system needs to understand WHAT TYPE of content it's looking at and HOW to extract meaning
from each type.

### How It Works in Multimodal

---
TEXT-ONLY RAG (what we have)

Question -> Search text vectors -> Get text chunks -> Answer
(One path, one modality, simple)


MULTIMODAL RAG (with context gathering)

Question -> Context Gatherer decides:
  |
  |--- "This is about a chart" -> Search image embeddings
  |                               -> Use vision model to describe chart
  |                               -> Feed description to LLM
  |
  |--- "This is about a policy" -> Search text vectors (normal RAG)
  |
  |--- "This needs both" -> Search text + images
  |                         -> Combine context from both
  |                         -> Feed combined context to LLM
  |
  |--- "This is about data" -> Search spreadsheet embeddings
                            -> Extract relevant rows/columns
                            -> Feed structured data to LLM
---

The context gatherer becomes the ROUTER that decides:
1. Which modality to search (text? images? tables?)
2. Which models to use for understanding (text LLM? vision model? both?)
3. How to combine results from different sources

### Models for Each Modality

| Modality | Embedding Model | Understanding Model | Ollama Command |
| :--- | :--- | :--- | :--- |
| Text | nomic-embed-text (what we use) | Mistral (what we use) | Already installed |
| Images | CLIP or SigLIP | LLaVA, Llama 3.2 Vision | `ollama pull llava` |
| Audio | Whisper (transcribe first) | Then text pipeline | Whisper via Python |
| Tables/CSV | Text embedding on rows | LLM with structured prompt | Already works |
| PDF charts | Extract images separately | Vision model for charts | `ollama pull llava` |

### How It Connects to Our Project

If we added multimodal support, here's what changes in each module:

| Module | What Changes |
| :--- | :--- |
| `src/chatbot.py` | Add modality detection, route to appropriate pipeline |
| `src/guardrails/content_safety.py` | Add image content safety (NSFW detection) |
| `src/generation/prompts.py` | New templates for image-description prompts |
| `src/memory/memory_bank.py` | Track which modalities were used per exchange |
| `src/guardrails/model_governance.py` | Validate vision model sources |
| `notebooks/lessons/03_ingest_documents.py` | Add image extraction from PDFs |
| ChromaDB | Store image embeddings alongside text embeddings |

### Example: Multimodal HR Chatbot

Employee: "What does the org chart show about the HR department?"

Context Gatherer:
1. Detects: "org chart" -> needs IMAGE understanding
2. Searches image embeddings -> finds org_chart.png
3. Sends to LLaVA (vision model): "Describe this org chart"
4. LLaVA: "HR department has 3 teams: Recruitment (5), Benefits (3), Compliance (4)"
5. Also searches text docs -> finds HR department description
6. Combines image description + text context
7. Generates final answer with both sources

Answer: "According to the org chart, the HR department has three teams:
Recruitment with 5 members, Benefits with 3 members, and Compliance
with 4 members. The department reports to the VP of People Operations."

### The Bottom Line

| System | Text Only | Multimodal |
| :--- | :--- | :--- |
| Without context gathering | Works fine | Broken (can't route between modalities) |
| With context gathering | Works better (multi-step) | Works well (routes + combines) |

Context gathering goes from "nice to have" in text-only systems to "must have" in
multimodal systems. It's the brain that decides how to find and combine information
across different types of content.

---

## 16. What are the hardware requirements for local, Colab, and production?

### Quick Summary

| Environment | CPU | RAM | GPU | Disk | Cost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Local (minimum) | 4 cores | 8 GB | None | 10 GB | Free |
| Local (recommended) | 8 cores | 16 GB | NVIDIA 6GB+ | 20 GB | Free |
| Local (fine-tuning) | 8 cores | 16 GB | NVIDIA 8GB+ | 30 GB | Free |
| Colab Free | - | - | T4 15GB | 100 GB | Free |
| Colab Pro | - | - | L4 24GB | 100 GB | ~$10/month |
| Production (minimal) | 12 vCPU | 48 GB | L4 24GB | 200 GB | ~$1,080/month |
| Production (HA) | 36+ vCPU | 160+ GB | 2x A100 40GB | 500+ GB | ~$7,150/month |
| On-premises | 48+ cores | 192+ GB | 2x A100 80GB | 4+ TB | ~$57,000 one-time |

### Key Points

- **No GPU needed for local development.** Ollama runs on CPU. It's slower (10-30 seconds
  per answer vs 1-3 seconds with GPU) but fully functional for learning.

- **8GB RAM is the absolute minimum.** Mistral 7B takes ~4GB, leaving ~4GB for everything
  else. 16GB is where it gets comfortable.

- **Colab Free is enough for fine-tuning.** The T4 GPU (15GB VRAM) handles Mistral 7B and
  Llama 3.1 8B with QLoRA. No local GPU needed.

- **Production costs scale with concurrency.** 5,000 questions/month (~7/hour average) is
  modest. The cost comes from handling bursts (Monday mornings, policy changes) and
  maintaining high availability.

Full details: [HARDWARE_REQUIREMENTS.md](HARDWARE_REQUIREMENTS.md)

---

## 17. What dataset source should I use?

### Quick Answer

| Source | Use For | Don't Use For |
| :--- | :--- | :--- |
| Hugging Face | Fine-tuning (instruction following, domain Q&A, chat) | Pre-training from scratch |
| Kaggle | Structured/tabular data, niche domains | Direct fine-tuning (needs format conversion) |
| Common Crawl | Pre-training from scratch (NOT fine-tuning) | Fine-tuning (too noisy, too large) |
| Project Gutenberg | Literary style, historical text, clean long-form | Q&A or instruction format |
| Wikipedia | Factual knowledge, broad topic coverage | Direct fine-tuning (needs conversion) |
| Your own docs | Domain-specific fine-tuning (ALWAYS the best) | When you have fewer than 50 examples |
### For Our HR Policy Chatbot

1. RAG documents -> Your own HR policies (not from any public dataset)
2. Fine-tuning (general) -> Hugging Face: `tatsu-lab/alpaca` or `teknium/OpenHermes-2.5`
3. Fine-tuning (domain) -> Generate Q&A pairs from your own documents
4. Skip Common Crawl and Gutenberg for this use case

### Recommended Learning Path

Start with Hugging Face (easiest), then try generating from your own docs,
then explore Wikipedia, Kaggle, Gutenberg, and finally Common Crawl.

Full details with code examples for every source: [TRAINING_DATA_GUIDE.md](TRAINING_DATA_GUIDE.md)

---

## 18. Do I pull data using Python or use actual files?

**Both — but for different purposes.**

There are TWO different ways data enters the project:

| What | How It Gets In | Format | Used By |
| :--- | :--- | :--- | :--- |
| RAG documents | You manually place files in `documents/` folder | PDF, DOCX, TXT, MD, CSV | Ingestion script -> ChromaDB -> chatbot searches them |
| Training data | Python pulls from Hugging Face / Kaggle / Wikipedia | JSON, JSONL (in RAM or saved to disk) | Fine-tuning script -> trains the model |

### RAG Documents = Actual Files (Manual)

---
You copy/paste/download files into:
documents/
├── employee-handbook.pdf     <- You got this from HR
├── leave-policy.docx         <- You downloaded this
└── api-docs.md               <- You wrote this
---

The ingestion script reads them from disk. No Python pulling involved.

### Training Data = Pulled via Python (Automatic)

```python
from datasets import load_dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train[:5000]")
# 5000 Q&A examples now in RAM - used directly by the trainer
```

You CAN also save it as a file if you want to inspect or edit it:

```python
dataset["train"].to_json("finetune_data/alpaca.jsonl")
# Now it's a file on disk you can open and review
```

### Why the Difference?

- RAG documents are YOUR specific content. No public dataset has your company's HR policies.
  You must provide the actual files.

- Training data needs thousands of examples to change model behavior. Nobody creates 50,000
  Q&A pairs by hand. Python pulls them from curated sources.

Our fine-tuning scripts use the Python-pull approach by default — no manual file management needed.

---

## 19. What about SQuAD, MS MARCO, HotpotQA, BEIR, PubMed, arXiv, SEC EDGAR?

We didn't use any of these in the project. They fall into two categories we hadn't covered:

### Category 1: RAG Benchmark Datasets (for EVALUATING your system)

These aren't for training — they're for TESTING whether your RAG pipeline works well.
They have known correct answers so you can measure accuracy objectively.

| Dataset | What It Tests | Questions | Source |
| :--- | :--- | :--- | :--- |
| SQuAD 2.0 | Basic comprehension + "I don't know" | 150K | Wikipedia paragraphs |
| MS MARCO | Real-world search query handling | 1M+ | Real Bing searches |
| Natural Questions | End-to-end Q&A from real Google searches | 300K+ | Google queries + Wikipedia |
| HotpotQA | Multi-document reasoning (hardest) | 113K | Wikipedia (multi-hop) |
| BEIR | Cross-domain retrieval (18 benchmarks) | Varies | Multiple domains |

Our project uses LLM-as-judge (Mistral scores itself). These benchmarks provide
objective, human-verified evaluation — a more rigorous alternative.

### Category 2: Domain-Specific Corpora (for RAG documents or fine-tuning)

Massive professional document collections for specific industries:

| Source | Domain | Size | Use As |
| :--- | :--- | :--- | :--- |
| PubMed / PMC | Medical / biomedical | 35M+ abstracts, 8M+ full-text | RAG documents for medical chatbot |
| arXiv | Scientific research | 2.3M+ papers | RAG documents for research assistant |
| SEC EDGAR | Financial / corporate | 20M+ filings | RAG documents for financial analysis |

### How They Connect to Our Project

- Use SQuAD/MS MARCO to BENCHMARK our chatbot's accuracy (instead of just LLM-as-judge)
- Use PubMed/arXiv/EDGAR as RAG document sources if building a domain-specific chatbot
- Use PubMedQA or generated Q&A from arXiv for domain-specific fine-tuning

Full details with code examples: [TRAINING_DATA_GUIDE.md](TRAINING_DATA_GUIDE.md) (Sections 11-13)

---

## 20. Which datasets run locally vs need Colab?

### Quick Rule

- **Loading/exploring** any dataset -> always works locally (use streaming for big ones)
- **Evaluating** with benchmarks (SQuAD, BEIR) -> always works locally (CPU is fine)
- **Training** -> depends on dataset size and your GPU

### Training Decision Table

| Dataset Size | No GPU | 8GB GPU | Colab Free (T4) | Colab Pro |
| :--- | :--- | :--- | :--- | :--- |
| 100-500 examples | NO | ~10 min | ~5 min | ~2 min |
| 1K-5K examples | NO | ~30-60 min | ~15-30 min | ~10 min |
| 5K-50K examples | NO | 2-8 hrs | ~1-4 hrs | ~30-90 min |
| 50K+ examples | NO | NO | Tight | YES |

### Practical Advice

- Start with 500-1000 examples locally (if you have a GPU) or on Colab Free
- RAG benchmarks (SQuAD, MS MARCO, BEIR) run on CPU - no GPU needed
- Common Crawl is too large for anything except a compute cluster - skip it
- For large domain corpora (PubMed, arXiv), use a small subset (1-5K) for fine-tuning

Full details: [TRAINING_DATA_GUIDE.md](TRAINING_DATA_GUIDE.md) (Section 14)

---

## 21. What kind of model do we get after training? Do we have Fine-Tuned Mistral + RAG?

### Current State: Generic Mistral + RAG

Right now, the chatbot uses the stock Mistral 7B from ollama. It has NOT been fine-tuned.
The fine-tuning scripts are built and ready, but not yet executed.

---
WHAT'S RUNNING NOW:
Generic Mistral 7B + FULL RAG Pipeline
├── RAG, guardrails, memory, prompts, evaluation — all WORKING
├── Fine-tuning scripts — READY but not executed
└── Model — GENERIC (not fine-tuned)
---

### After Fine-Tuning: Fine-Tuned Mistral + RAG

Once you run the fine-tuning script and import the model into ollama, you change
ONE line in `src/chatbot.py`:

```python
# FROM (current):
self.generator = ChatOllama(model="mistral", temperature=0.3)

# TO (after fine-tuning):
self.generator = ChatOllama(model="my-hr-model", temperature=0.3)
```

Everything else stays the same. The fine-tuned model is a drop-in replacement.

### What Changes After Fine-Tuning

| Aspect | Before (Generic) | After (Fine-Tuned) |
| :--- | :--- | :--- |
| Writing style | Generic | Your company's style |
| Domain terms | May misuse | Uses correctly |
| Answer format | Inconsistent | Consistent |
| "I don't know" | Generic | Domain-appropriate |

### What Does NOT Change

Fine-tuning does NOT make the model smarter at reasoning, give it knowledge it never
saw (that's RAG's job), or make it larger/slower. It changes HOW the model answers,
not WHAT it knows.

Full details: [TRAINING_DATA_GUIDE.md](TRAINING_DATA_GUIDE.md) (Sections 15-16)

---

## 22. What are hybrid search and re-ranking?

### Hybrid Search (Vector + BM25 Keyword)

Our chatbot originally used ONLY vector search (semantic similarity). This finds
documents by MEANING but misses EXACT keyword matches.

---
User asks: "What does Policy 4.2 say?"

Vector search: Finds chunks about policies in general (meaning of "policy")
               Might return Policy 3.1 or 5.0 - wrong ones
BM25 keyword:  Finds chunks containing exact text "Policy 4.2"
               Gets the right chunk immediately

Hybrid:        Combines both -> Policy 4.2 chunk ranks #1
---

When vector search wins: "How many vacation days?" -> finds "annual leave" (synonym)
When keyword search wins: "What is FMLA?" -> finds exact acronym match

Hybrid gets the best of both. Neither alone is perfect.

### Re-Ranking (Cross-Encoder)

Initial retrieval is approximate — embeddings compress paragraphs into 768 numbers,
losing nuance. A cross-encoder re-scores results by reading the FULL query + FULL
document TOGETHER, which is much more accurate.

---
WITHOUT re-ranking:
  Query -> vector -> top 3 by cosine similarity -> Done (approximate)

WITH re-ranking:
  Query -> vector + BM25 -> top 10 candidates (wider net)
  -> Cross-encoder reads each (query + document) pair
  -> Re-scores based on actual relevance
  -> Top 3 from re-ranked results -> Done (accurate)
---

### Impact

| Approach | Accuracy Improvement | Latency Added |
| :--- | :--- | :--- |
| Vector only (original) | Baseline | 0ms |
| + Hybrid search | +15-20% | ~10ms |
| + Re-ranking | +10-20% on top | ~100-200ms |
| All combined | +25-35% total | ~200ms |

### Implementation

Both are now implemented in `src/retrieval/hybrid.py` and integrated into the chatbot.
Configurable in `config.yaml`:

```yaml
retrieval:
  use_hybrid: true       # Enable BM25 keyword search
  use_reranker: true     # Enable cross-encoder re-ranking
  vector_weight: 0.6     # 60% weight to vector search
  bm25_weight: 0.4       # 40% weight to keyword search
```

Set either to `false` to disable. The re-ranker downloads an ~80MB model on first use.

---

## 23. What are OpenClaw and NemoClaw?

### The Short Version

| | OpenClaw | NemoClaw |
| :--- | :--- | :--- |
| What | Open-source AI agent framework | Security-hardened wrapper around OpenClaw |
| Who | Community (MIT license) | NVIDIA (Apache 2.0) |
| Can do | Read/write files, browse web, run scripts, send emails | Same — but inside a secure sandbox |
| Target | Developers, power users | Enterprises, regulated environments |
| Security | App-level (you manage it) | Kernel-level sandboxing, policy enforcement |
| OS | Windows, macOS, Linux | Linux (Ubuntu 22.04+) |
| Models | Any (OpenAI, Anthropic, DeepSeek, local) | NVIDIA Nemotron by default |

### What's an AI Agent? (vs Our Chatbot)

Our chatbot is a **reader** — it reads documents and answers questions. It cannot take
any action in the real world. This is safe by design.

An AI agent is a **doer** — it can execute actions: read/write files, browse the web,
run shell commands, send emails, manage calendars, interact with APIs.

---
OUR RAG CHATBOT                           AI AGENT (OpenClaw/NemoClaw)
----------------                          ----------------------------
User: "What's the leave policy?"          User: "Book me 3 days off next week"
Bot: "You get 20 days per year..."        Agent: [Opens calendar API]
                                                 [Checks available dates]
                                                 [Submits leave request]
                                                 [Sends confirmation email]
                                          Agent: "Done. I've booked Dec 15-17 off
                                                 and emailed your manager."
---

The agent doesn't just ANSWER — it ACTS. This is powerful but dangerous.

### OpenClaw - The Flexible Agent

OpenClaw is an open-source autonomous AI agent you run on your own machine.

**What it can do:**
- Read and write files on your computer
- Browse the web and extract information
- Run shell scripts and commands
- Manage email, calendars, and messaging
- Connect to chat apps (WhatsApp, Telegram, Slack, Discord)
- Use any LLM (OpenAI, Anthropic, local models via ollama)
- Persistent memory (remembers across sessions)
- Plugin/skill ecosystem (extend with custom capabilities)

**Architecture:**
---
User (via chat app: Slack, Telegram, etc.)
  |
  ▼
OpenClaw Agent
  ├── LLM Brain (any model — OpenAI, ollama, etc.)
  ├── Memory (persistent, like our Memory Bank)
  ├── Tools/Plugins:
  │   ├── File system access (read/write/delete)
  │   ├── Web browser (search, scrape, navigate)
  │   ├── Shell executor (run commands)
  │   ├── Email client (send/receive)
  │   ├── Calendar manager
  │   └── Custom plugins (you build these)
  └── Skill library (pre-built capabilities)
---

**Strengths:**
- MIT licensed, fully open source
- Runs locally (your data stays on your machine)
- Model-agnostic (works with any LLM)
- Massive plugin ecosystem
- Fast to set up

**Risk:**
If you give it file system access and shell execution, a prompt injection attack
could make it delete files, exfiltrate data, or run malicious commands. Security
is YOUR responsibility.

### NemoClaw - The Secure Agent (NVIDIA)

NemoClaw is NVIDIA's security-hardened version of OpenClaw. Same agent brain,
but locked inside a sandbox with strict policies.

**What it adds over OpenClaw:**
- Kernel-level sandboxing (agent can't escape its container)
- Filesystem isolation (agent only sees what you allow)
- Network restrictions (agent can only access approved URLs)
- Shell restrictions (only approved commands can run)
- Policy enforcement (define what the agent can and cannot do)
- Full audit trails (every action logged)
- Built on NVIDIA Agent Toolkit + OpenShell runtime

**Architecture:**
---
User
  |
  ▼
NemoClaw (Security Layer)
  ├── Policy Engine (what the agent CAN do)
  │   ├── Filesystem policy: /data/read-only, /output/read-write
  │   ├── Network policy: only api.company.com allowed
  │   ├── Shell policy: only "python", "pip" commands allowed
  │   └── Action policy: no email sending, no file deletion
---
**Strengths:**
- Enterprise-grade security
- Policy-based guardrails (not just prompt-level)
- Sandboxed execution (kernel-level isolation)
- Audit trails for compliance
- Safer for always-on agents

**Trade-offs:**
- Linux only (Ubuntu 22.04+)
- Higher resource usage (sandbox overhead)
- More complex setup
- Smaller ecosystem than vanilla OpenClaw
- Still early/evolving

### How They Compare to Our Project

| Aspect | Our RAG Chatbot | OpenClaw | NemoClaw |
| :--- | :--- | :--- | :--- |
| Type | Q&A chatbot | Autonomous agent | Secure autonomous agent |
| Can take actions | NO (read-only) | YES (full system access) | YES (sandboxed) |
| Security risk | LOW | HIGH | MEDIUM (sandboxed) |
| Guardrails | Content filters + PII | App-level only | Kernel-level + policy |
| Memory | Memory Bank (3 layers) | Persistent memory | Persistent + audited |
| Models | Ollama (local) | Any provider | NVIDIA Nemotron |
| Use case | Answer questions from docs | Automate tasks | Automate tasks securely |
| Complexity | Medium | Medium | High |

### When Would You Use Each?

| Scenario | Best Choice | Why |
| :--- | :--- | :--- |
| "Answer HR policy questions" | Our RAG Chatbot | Read-only, safe, grounded in documents |
| "Automate my daily tasks" | OpenClaw | Needs to take actions (files, email, web) |
| "Deploy an agent for the whole company" | NemoClaw | Needs security, audit, policy enforcement |
| "Research assistant that browses the web" | OpenClaw | Needs web access |
| "Compliance bot that processes documents" | Our RAG + NemoClaw patterns | Needs both document Q&A and security |

### Could We Add Agent Capabilities to Our Project?

Yes. LangChain supports agents with tools. We could evolve our chatbot into an agent:

```python
# Conceptual — not implemented
from langchain.agents import create_react_agent

tools = [
    vector_search_tool,      # Our existing RAG retrieval
    calendar_tool,           # Book leave (new)
    email_tool,              # Send notifications (new)
    hr_system_tool,          # Submit requests (new)
]

agent = create_react_agent(llm, tools, prompt)
# Now the chatbot can both ANSWER and ACT
```

But adding agency means adding the Excessive Agency security threat (currently LOW
in our project, would become CRITICAL). You'd need NemoClaw-style sandboxing
and policy enforcement before deploying an agent in production.

### The Evolution Path

---
Level 1: RAG Chatbot (what we built)
  -> Answers questions from documents
  -> Zero agency, zero risk

Level 2: RAG + Simple Tools
  -> Answers questions + can search web, look up data
  -> Low agency, low risk

Level 3: AI Agent (OpenClaw)
  -> Can take actions: files, email, scripts
  -> High agency, high risk, needs careful guardrails

Level 4: Secure AI Agent (NemoClaw)
  -> Same as Level 3 but sandboxed with policies
  -> High agency, managed risk, enterprise-ready
---
Our project is at Level 1. The advanced topics in the lesson plan (Context Gathering,
LangChain Agents) would move it toward Level 2. OpenClaw/NemoClaw are Levels 3-4.

---

## 24. What are the 4 levels of AI systems?

### Level 1: RAG Chatbot (Our Project)

- Can: Read documents, answer questions
- Cannot: Take ANY action in the real world
- Security risk: LOW (worst case = wrong answer)
- Example: "What's the leave policy?" -> answers from documents

### Level 2: RAG + Simple Tools

- Can: Everything Level 1 + call read-only APIs, search web
- Cannot: Modify anything, write files, send emails
- Security risk: MEDIUM (could leak info via API calls)
- Example: "What's the weather?" -> calls weather API, answers

### Level 3: AI Agent (OpenClaw)

- Can: EVERYTHING — files, web, email, shell, APIs, databases
- Cannot: Nothing is off-limits
- Security risk: HIGH (prompt injection = attacker controls your system)
- Example: "Book me 3 days off" -> opens calendar, submits request, emails manager

### Level 4: Secure AI Agent (NemoClaw)

- Can: Same as Level 3, but CONSTRAINED by policies
- Cannot: Anything not explicitly allowed (default deny)
- Security risk: MANAGED (kernel-level sandbox, policy enforcement, audit trails)
- Example: Same tasks as Level 3, but every action checked against policy first

### Quick Comparison

| Aspect | Level 1 | Level 2 | Level 3 | Level 4 |
| :--- | :--- | :--- | :--- | :--- |
| Type | Q&A Bot | Smart Assistant | Agent | Secure Agent |
| File access | NO | NO | YES | Policy-controlled |
| Shell access | NO | NO | YES | Policy-controlled |
| Email access | NO | NO | YES | Policy-controlled |
| Prompt injection impact | Bad answer | Data leak | System compromise | Blocked by policy |
| Best for | Document Q&A | Research | Personal automation | Enterprise automation |

Our project is Level 1. The advanced topics in the lesson plan (Context Gathering,
LangChain Agents) would move it toward Level 2. OpenClaw is Level 3. NemoClaw is Level 4.

Each level adds capability AND risk. The jump from Level 1 to Level 3 is where
security becomes critical — which is exactly why NemoClaw (Level 4) wraps OpenClaw
(Level 3) in a security sandbox.

---

## 25. Can this system handle 1M documents?

### Short Answer: No.

The current system handles ~1,000-5,000 document chunks comfortably. At 50,000 chunks
it slows down. At 100,000+ it breaks. 1M documents is a fundamentally different
architecture problem.

### What Breaks and Why

| Scale | Status | Bottleneck |
| :--- | :--- | :--- |
| 1-5K chunks | Works well | - |
| 10-50K chunks | Slows down | ChromaDB file I/O |
| 100K+ chunks | Breaks | ChromaDB single-process limit |
| 1M documents | Impossible | Need distributed vector DB + GPU embedding cluster |

### What Would Need to Change

| Component | Current | For 1M Documents |
| :--- | :--- | :--- |
| Vector DB | ChromaDB (file-based) | Qdrant or Milvus (distributed, sharded) |
| Embedding | Sequential, one at a time | Batch GPU embedding (Ray or Spark) |
| Ingestion | Single-process Python | Distributed pipeline |
| Search index | Flat (exact) | HNSW or IVF (approximate, 100x faster) |

### The Scaling Path

---
Level 1 (current): 1-5K docs,    1-5 users    -> ChromaDB + Ollama
Level 2:           10-50K docs,  10-50 users  -> pgvector + vLLM
Level 3:           100-500K docs, 100-500 users -> Qdrant + vLLM cluster
Level 4:           1M+ docs,     1000+ users  -> Qdrant cluster + Kubernetes
---

Each level is an infrastructure change, not a code change. The RAG pipeline logic
(retrieve -> generate -> evaluate) stays the same — only the underlying storage
and serving infrastructure changes.

This is a learning project. It teaches the CONCEPTS that apply at every scale.
The scaling is an ops problem, not a code problem.

Full details: [ENGINEERING_REVIEW.md](ENGINEERING_REVIEW.md) -> "Scaling Limits" section

---

## 26. Can the fine-tuning pipeline handle 1M training examples?

### Loading: Yes. Validating: No.

The fine-tuning scripts CAN load and train on 1M examples. But they have NO
testing or validation phase to verify the trained model is actually good.

### What Works

```python
# Loading 1M examples — works via streaming (no memory issues)
from datasets import load_dataset
dataset = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
# Streams data, never loads all 1M into RAM at once
```

### Training Time for 1M Examples

| Hardware | 1M Examples, 1 Epoch | 1M Examples, 3 Epochs |
| :--- | :--- | :--- |
| Colab Free (T4 16GB) | ~15-20 hours (may timeout) | Not feasible (session limit) |
| Colab Pro (L4 24GB) | ~8-12 hours | ~24-36 hours |
| Colab Pro+ (A100 40GB) | ~4-6 hours | ~12-18 hours |
| Local RTX 3090 (24GB) | ~10-15 hours | ~30-45 hours |

### What's Missing (The Real Gap)

| Phase | Status | What It Should Do |
| :--- | :--- | :--- |
| Data loading | WORKS | Stream from Hugging Face |
| Data filtering | IMPLEMENTED | Remove duplicates, empty, toxic, too-short, too-long |
| Train/validation split | IMPLEMENTED | Hold out 10% for validation |
| Training | WORKS | QLoRA fine-tuning |
| Validation during training | IMPLEMENTED | Track validation loss each epoch, detect overfitting |
| Early stopping | IMPLEMENTED | Stop when validation loss stops improving, select best checkpoint |
| Post-training evaluation | IMPLEMENTED | Before/after benchmark comparing base vs fine-tuned |
| Model selection | IMPLEMENTED | Auto-select best checkpoint by validation loss |

All implemented in `scripts/training_validator.py` and integrated into `scripts/finetune.py`.

### What Overfitting Looks Like (and Why Validation Matters)

---
WITHOUT validation (what we have):
  Epoch 1: training loss = 1.5
  Epoch 2: training loss = 0.8
  Epoch 3: training loss = 0.3  <- "Great, loss is going down!"

  But the model MEMORIZED the training data instead of LEARNING patterns.
  On new questions, it gives weird, repetitive, or wrong answers.
  You have NO WAY TO KNOW this happened.

WITH validation (what we need):
  Epoch 1: train_loss = 1.5, val_loss = 1.6  <- Both going down, good
  Epoch 2: train_loss = 0.8, val_loss = 1.0  <- Both going down, good
  Epoch 3: train_loss = 0.3, val_loss = 1.2  <- Train down, val UP = OVERFITTING

  Early stopping: "Validation loss increased. Stopping at epoch 2."
  Model selection: "Using epoch 2 checkpoint (lowest validation loss)."
---

### What a Proper Training Pipeline Looks Like

---
1. LOAD DATA
   dataset = load_dataset("teknium/OpenHermes-2.5", streaming=True)

2. FILTER DATA (missing)
   Remove duplicates, empty examples, toxic content
   Filter by quality score if available

3. SPLIT DATA (missing)
   90% training, 10% validation
   Ensure validation set covers all topics

4. TRAIN
   QLoRA fine-tuning (what we have)
   BUT with eval_dataset and eval_strategy="steps"

5. VALIDATE DURING TRAINING (missing)
   Track validation loss every N steps
   Early stopping if validation loss increases for 3 checks

6. SELECT BEST MODEL (missing)
   Pick checkpoint with lowest validation loss
   Not the last checkpoint

7. POST-TRAINING EVALUATION (missing)
   Run ground truth test set (TruthfulnessScorer)
   Compare: base model accuracy vs fine-tuned accuracy
   If fine-tuned is WORSE -> don't deploy

8. DEPLOY (only if evaluation passes)
   Convert to GGUF -> import to Ollama -> update config.yaml
---

### Practical Advice for 1M Examples

You almost certainly don't need all 1M examples:

| Approach | Examples Used | Quality | Time |
| :--- | :--- | :--- | :--- |
| Use all 1M | 1,000,000 | Diminishing returns after ~50K | 12-36 hours |
| Random sample 50K | 50,000 | Nearly as good as 1M | 2-6 hours |
| Curated 10K | 10,000 | Often BETTER than random 1M | 30-90 min |
| Curated 5K + 500 domain | 5,500 | Best for domain-specific use | 30-60 min |

Quality > Quantity. 5,000 curated examples with 500 domain-specific ones will
outperform 1M random examples for most use cases.

---

## 27. What do we have for observability?

### What We Built (5 Layers)

| Layer | What It Tracks | File |
| :--- | :--- | :--- |
| Audit Logging | Every Q&A: user, question, answer, scores, timing | `src/api/audit.py` |
| Monitoring Metrics | Request counts, latency, cache hit rate, quality scores | `src/api/monitoring.py` |
| RAG Quality Monitor | Groundedness, hallucination rate, health status over time | `src/evaluation/rag_monitor.py` |
| Human Feedback | User satisfaction ratings, LLM-vs-human score correlation | `src/evaluation/feedback.py` |
| Guardrail Stats | Content safety violations by category | `src/guardrails/content_safety.py` |

### What's Missing

| Missing | What It Is | Why It Matters |
| :--- | :--- | :--- |
| Structured logging | Python `logging` with DEBUG/INFO/WARNING/ERROR levels | Can't filter or search logs by severity |
| LangChain tracing | Traces every chain step (prompt -> LLM -> parse) | Can't see WHERE in the pipeline a problem occurred |
| Distributed tracing | OpenTelemetry spans across all components | Can't trace a single request end-to-end |
| Prometheus format | Standard `/metrics` endpoint for Grafana | Can't connect to industry-standard dashboards |

### The Biggest Gap: LangChain Tracing

LangChain has built-in support for tracing tools that show exactly what happened at
each step — what the prompt looked like, what the LLM returned, how long each step
took, what documents were retrieved. We're not using any of them.

| Tracing Tool | What It Does | Cost | Self-Hosted? |
| :--- | :--- | :--- | :--- |
| LangSmith | Official LangChain tracing | Free tier + paid | No (cloud) |
| Langfuse | Open-source LLM observability | Free | YES |
| Phoenix (Arize) | Open-source LLM tracing | Free | YES |
| OpenLLMetry | OpenTelemetry for LLMs | Free | YES |

For our local project, Langfuse would be the best fit — open source, self-hosted,
integrates with LangChain in 2 lines of code.
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
## 28. Why LangChain and not LlamaIndex?
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
### Why We Chose LangChain
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
<!-- Line hidden/folded in original view -->
2. **Production patterns** - The chain pattern (`prompt | llm | parser`) is the same
pattern used in production systems. LlamaIndex abstracts more away — easier but
teaches less about what's happening underneath.

3. **Flexibility** - We built custom guardrails, custom memory, custom retrieval,
custom evaluation. LangChain lets you plug in custom components at every step.

4. **Ecosystem** - LangSmith (tracing), LangGraph (stateful agents), LangServe
(deployment). The ecosystem is larger.

### When LlamaIndex Would Be Better

| Scenario | Better Choice | Why |
| :--- | :--- | :--- |
| Pure RAG (just document Q&A) | LlamaIndex | Built-in optimized pipeline, less code |
| Complex agents with tools | LangChain | Better agent framework |
| Learning AI concepts broadly | LangChain | Covers more ground |
| Quick prototype | LlamaIndex | Less boilerplate |
| Multi-modal RAG (images/tables) | LlamaIndex | Better built-in support |

### Same RAG in LlamaIndex (6 Lines vs Our 350)

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama

documents = SimpleDirectoryReader("data/documents").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(llm=Ollama(model="mistral"))
response = query_engine.query("How many leave days?")
```

6 lines vs our ~350 lines. LlamaIndex abstracts more — simpler but you learn less
about what's happening underneath (chunking, embedding, retrieval, prompt assembly).

### Can You Use Both?

Yes. Some production systems use LlamaIndex for the RAG pipeline and LangChain for
the agent/tool layer. They're not mutually exclusive.

### The Java Equivalent

| Python | Java |
| :--- | :--- |
| LangChain | LangChain4j |
| LlamaIndex | No direct equivalent (use LangChain4j for everything) |

---


## 29. Are we using LangGraph to manage the RAG pipeline?

### No. We use plain Python with manual if/else flow.

Our pipeline in `src/chatbot.py` is a manually coded sequence:

```python
def ask(self, question):
    sanitized = validate_and_sanitize_input(question)    # Step 0
    if not sanitized["valid"]: return error
    input_check = self.guardrails.check_input(question)  # Step 1
    if not input_check["safe"]: return blocked
    retrieval = self.retrieve_with_confidence(query)     # Step 2
    if not retrieval["confident"]: return low_confidence
    answer = self.rag_chain.invoke(...)                  # Step 3
    output_check = self.guardrails.check_output(answer)  # Step 4
    evaluation = evaluate_answer(...)                    # Step 5
    if not evaluation["passed"]: retry...                # Step 6
    self._save_exchange(...)                             # Step 7
    return answer
```

This works but has no formal state machine, no graph visualization, no checkpointing.

### What LangGraph Is

LangGraph is LangChain's framework for building stateful, multi-step AI workflows
as directed graphs. Instead of linear scripts, you define nodes (steps) and edges
(transitions), and the framework manages state, branching, and execution.

### What LangGraph Would Add

| Feature | Our Approach (plain Python) | LangGraph |
| :--- | :--- | :--- |
| State management | Manual dict passing | Automatic typed state |
| Branching | if/else | Conditional edges (declarative) |
| Retry logic | Manual loop | Built-in graph cycles |
| Visualization | Read the code | `graph.draw_mermaid()` generates diagram |
| Checkpointing | None | Save/resume state at any node |
| Human-in-the-loop | Not possible | Built-in interrupt/resume |
| Parallel execution | Not possible | Multiple nodes run simultaneously |
| Debugging | Print statements | Step-by-step state inspection |

### Why We Didn't Use It

1. **Learning** — Seeing raw Python flow (if/else, function calls) is more educational
   than a graph framework abstraction.
2. **Our pipeline is linear** — Mostly sequential with one retry branch. LangGraph
   shines with complex multi-branch, multi-agent workflows.
3. **Minimal dependencies** — LangGraph is a separate package we didn't need.

### When You SHOULD Use LangGraph

| Scenario | Use LangGraph? | Why |
| :--- | :--- | :--- |
| Simple linear RAG (our project) | Optional | Works fine without it |
| RAG with multiple retry strategies | YES | Conditional edges handle branching |
| Multi-agent systems | YES | Each agent is a node |
| Human-in-the-loop approval | YES | Built-in interrupt/resume |
| Complex workflows (10+ steps) | YES | Graph visualization prevents spaghetti |
| Parallel retrieval (3 sources at once) | YES | Built-in parallel execution |

### What Our Pipeline Would Look Like in LangGraph

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(RAGState)
graph.add_node("sanitize", sanitize_node)
graph.add_node("guardrail_input", guardrail_input_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_node("evaluate", evaluate_node)

graph.add_edge("sanitize", "guardrail_input")
graph.add_conditional_edges("guardrail_input",
    lambda s: "blocked" if s["blocked"] else "retrieve")
graph.add_conditional_edges("evaluate",
    lambda s: "retrieve" if not s["passed"] else END)

app = graph.compile()
result = app.invoke({"question": "How many leave days?"})
```

Same logic, but declarative instead of imperative. The graph can be visualized,
checkpointed, and extended with parallel branches.

For our project, LangGraph is a "nice to have" for code organization, not a
"must have" for functionality. Everything it does, we already do manually.
