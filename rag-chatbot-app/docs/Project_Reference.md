# LESSON PLAN - From Zero to RAG Chatbot

## LESSON 0: VOCABULARY (Read This First, No Excuses)

Before you touch ANY code, you need to understand these words. I will use them constantly. If you forget, come back here.

| Term | Plain English |
| :--- | :--- |
| **LLM** | Large Language Model. A program trained on massive text data that can generate human-like text. Think of it as a very smart autocomplete. |
| **Model** | The AI brain. A file (usually 4-14GB) containing everything the AI "learned" during training. |
| **Ollama** | A tool to run AI models on your computer. Like Docker, but for AI models. |
| **Prompt** | The text you send to the AI. Your question or instruction. |
| **Token** | A chunk of text (roughly 3/4 of a word). Models process text as tokens, not words. |
| **Embedding** | Converting text into numbers (a list of numbers called a vector). Similar texts get similar numbers. This is how AI "understands" meaning. |
| **Vector** | A list of numbers representing text. Example: "cat" might become [0.2, 0.8, 0.1, ...]. Similar meanings = similar vectors. |
| **Vector Store / Vector Database** | A database optimized for storing and searching vectors. It finds "similar" documents by comparing vectors. |
| **RAG** | Retrieval Augmented Generation. A technique where you: (1) Find relevant documents, (2) Give them to the AI, (3) AI answers using those documents. |
| **LangChain** | A Python framework that connects LLMs with data sources, tools, and memory. Think of it as the plumbing between your AI and your data. |
| **Hugging Face** | A company/platform that hosts thousands of free AI models. Like Github but for AI models. |
| **Chunking** | Splitting large documents into smaller pieces so the AI can process them. |
| **Context Window** | The maximum amount of text an AI model can "see" at once. Like the model's short-term memory. |
| **Inference** | Running a model to get an answer. When you ask a question and the AI responds. |
| **Guardrails** | Rules that prevent the AI from generating harmful, incorrect, or off-topic content. |
| **Groundedness** | How well the AI's answer is supported by the actual source documents (not making stuff up). |
| **Hallucination** | When the AI confidently says something that is NOT in the source documents. This is the #1 problem in AI. |
| **Persistent Memory** | Conversation memory saved to disk (files/database) that survives app restarts. |
| **Conversation Summary** | An LLM-generated compressed version of older conversation exchanges. Preserves meaning while saving space. |
| **Sliding Window** | A technique that keeps only the N most recent exchanges in full detail, compressing older ones. |
| **Key Facts** | Important pieces of information extracted from conversations by the LLM and stored permanently. |
| **Session** | One conversation thread. A user can have multiple sessions, each with its own memory. |
| **Prompt Template** | A reusable text format with {variables} that get filled in at runtime. |
| **Prompt Assembly** | Combining multiple parts (persona + guardrails + task instructions + context) into one final prompt. |
| **Persona** | The "character" the AI plays, defined in the system message (e.g., "document assistant", "evaluation judge"). |
| **Prompt Registry** | A central dictionary of all named prompt templates, managed by the Prompt Assembler. |
| **Fine-Tuning** | Additional training on specific data to specialize a general model for your domain. |
| **QLORA** | Quantized Low-Rank Adaptation - fine-tune large models on consumer GPUs by combining 4-bit quantization with small trainable adapters. |
| **LoRA** | Low-Rank Adaptation - adds tiny trainable adapter layers (~0.1% of parameters) while keeping the base model frozen. |
| **Quantization** | Compressing model numbers from 16-bit to 4-bit to reduce memory usage. Slight quality loss, massive memory savings. |
| **Adapter** | Small trainable layers added on top of a frozen base model during LoRA/QLoRA fine-tuning. |
| **Epoch** | One complete pass through your entire training dataset. Typical: 1-3 epochs. |
| **Loss** | A number measuring how wrong the model is during training. Should decrease over time. Lower = better. |
| **VRAM** | Video RAM on your GPU. The main bottleneck for fine-tuning. 8GB minimum, 24GB ideal. |
| **SFT** | Supervised Fine-Tuning - training on input/output pairs (question â†’ answer). The most common fine-tuning approach. |

## LESSON 1: INSTALL OLLAMA (Your First AI Model)

### What is Ollama?
Ollama is a tool that downloads and runs AI models locally on your computer.
No internet needed after download. No API keys. No costs. Your data stays on YOUR machine.


### Step 1.1: Install Ollama

**Windows:**
1. Go to https://ollama.com/download
2. Download the Windows installer
3. Run the installer
4. Open a NEW terminal and verify:

```bash
ollama --version
```

You should see a version number. If you don't, restart your terminal.

### Step 1.2: Download Your First Model

We'll start with **Mistral 7B** - a powerful open-source model that runs on most machines.

```bash
ollama pull mistral
```

This downloads ~4GB. Wait for it to finish.


**What just happened?**

You downloaded a 7-billion parameter AI model to your computer.
"Parameters" are the model's learned knowledge - 7 billion tiny numbers that encode everything the model learned from training on internet text.

### Step 1.3: Talk to Your First AI

```bash
ollama run mistral
```

Type a question. Any question. The AI will respond.
Type /bye to exit.

**Congratulations. You just ran an AI model locally. No cloud. No API key. Free.**

### Step 1.4: Download the Embedding Model

For RAG, we need a SECOND model - not for chatting, but for understanding document meaning.

```bash
ollama pull nomic-embed-text
```

This is an embedding model (~274MB). It converts text into vectors (numbers).

**Why two models?**
* **mistral** = generates text answers (the "brain")

* **nomic-embed-text** = converts documents to searchable vectors (the "librarian")

### Step 1.5: Verify Both Models

```bash
ollama list
```

You should see both **mistral** and **nomic-embed-text**.

## LESSON 2: PYTHON ENVIRONMENT SETUP

### Step 2.1: Create the Project

```bash
cd local-rag-chatbot
python -m venv venv
```

### Step 2.2: Activate the Virtual Environment

**Windows (bash/git bash):**

```bash
source venv/Scripts/activate
```


**Windows (PowerShell):**

```powershell
venv\Scripts\Activate.ps1
```

**Windows (CMD):**

```cmd
venv\Scripts ctivate.bat
```

You should see (venv) in your terminal prompt.

### Step 2.3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## LESSON 3: LANGCHAIN BASICS (Connecting to Ollama)

### What is LangChain?
LangChain is a Python framework. It doesn't DO the AI work itself.

It CONNECTS things together: your model + your documents + your logic.

Think of it like this:
- **Ollama** = the engine
- **LangChain** = the car frame, steering wheel, and pedals
- **Your code** = the driver

### Step 3.1: Run the basic chat script

```bash
python notebooks/lessons/01_basic_chat.py
```

This script:
1. Connects to ollama (running on your machine)
2. Sends a prompt to Mistral
3. Prints the response

Open the file and READ every comment. I wrote them for you.

### Step 3.2: Understand the Chain Concept

LangChain uses "chains" - a sequence of steps:

`User Question` → `Prompt Template` → `LLM` → `Output Parser` → `Answer`


Each step transforms the data. This is the core pattern you'll use everywhere.

## LESSON 4: EMBEDDINGS AND VECTOR STORES

### What Are Embeddings?
Imagine you could convert any sentence into a point on a map.
Similar sentences would be CLOSE together on the map.
Different sentences would be FAR apart.

That's what embeddings do, but in 768+ dimensions (not just 2D).

[Image of high-dimensional vector space mapping for text similarity]

### Step 4.1: Run the embedding demo

```bash
python notebooks/lessons/02_embeddings.py
```

This shows you how text becomes numbers, and how similar texts have similar numbers.

### Step 4.2: Understanding ChromaDB

ChromaDB is our vector database. It:
1. Stores document embeddings (the number representations)
2. When you ask a question, converts your question to numbers

3. Finds the documents with the CLOSEST numbers (most similar meaning)

This is the "Retrieval" in RAG.

---

## LESSON 5: RAG - RETRIEVAL AUGMENTED GENERATION

### The Big Picture

[Image of RAG vs standard LLM workflow diagram]

**Without RAG:**
* User: "What is our refund policy?"
* AI: "I don't know your specific refund policy." (or worse, makes one up)

**With RAG:**
* User: "What is our refund policy?"
* System: [searches your documents, finds refund-policy.pdf]
* System: [gives the relevant section to the AI]
* AI: "According to your policy document, refunds are available within 30 days..."

### Step 5.1: Add Your Documents


### Step 5.1: Add Your Documents
Put any .txt, .pdf, .md, or .csv files in the `data/documents/` folder.
Start with a few small files. We'll process them.

### Step 5.2: Run the document ingestion

```bash
python notebooks/lessons/03_ingest_documents.py
```

This script:
1. Reads all files from `data/documents/`
2. Splits them into chunks (because models have limited context windows)
3. Converts each chunk to a vector using `nomic-embed-text`
4. Stores everything in ChromaDB

### Step 5.3: Run the RAG chatbot

```bash
python notebooks/lessons/04_rag_chatbot.py
```

NOW you have a chatbot that answers from YOUR documents.

## LESSON 6: QUALITY EVALUATION (LLM-as-Judge)


### Why Evaluate?
An AI can sound confident while being completely wrong.
We need to CHECK if the answer is actually good.

### Two Metrics We'll Measure:

[Image of LLM-as-a-Judge evaluation framework]

1. **Answer Relevance**: Does the answer actually address the question?
2. **Groundedness**: Is the answer supported by the retrieved documents?

### Step 6.1: Run the evaluation

```bash
python notebooks/lessons/05_evaluation.py
```

This uses Mistral itself to judge the quality of its own answers.
(In the AWS version, they use a separate Mistral 7B Instruct for this.)

## LESSON 7: HUGGING FACE MODELS (Expanding Your Options)

### What is Hugging Face?


Think of it as the "Github of AI models."
Thousands of free models, datasets, and tools.

### Step 7.1: Using Hugging Face models with Ollama

Many Hugging Face models can be converted to Ollama format.
We'll also learn to use Hugging Face's "transformers" library directly.

```bash
python notebooks/lessons/06_huggingface.py
```

### Step 7.2: Trying Different Models

[Image of comparative analysis of open-source LLMs by size and performance]

| Model | Size | Good For |
| :--- | :--- | :--- |
| **mistral (7B)** | ~4GB | General purpose, good quality |
| **llama3.2 (3B)** | ~2GB | Faster, lighter, still decent |
| **phi3 (3.8B)** | ~2.3GB | Microsoft's small but capable model |
| **gemma2 (9B)** | ~5.4GB | Google's open model, very capable |

Try them:

```bash
ollama pull llama3.2
ollama pull phi3
```

## LESSON 8: PUTTING IT ALL TOGETHER - FULL CHATBOT


### Step 8.1: Run the complete chatbot

```bash
python -m src.chatbot
```

This combines EVERYTHING:
* Document ingestion with multiple format support
* Vector search retrieval (Like Bedrock Knowledge Base)
* Context-aware answer generation (like Claude on AWS)
* Quality evaluation with LLM-as-judge (Like Mistral 7B evaluation)
* Query cleaning and normalization
* Source attribution
* Persistent conversation memory via Memory Bank (Lesson 9)
* Content safety guardrails (local version)
* Centralized prompts via Prompt Assembler (Lesson 10)
* Gradio web interface (--web flag)

### What Maps to What (Local vs AWS Reference)

| AWS Reference Feature | Our Local Equivalent |
| :--- | :--- |
| **Amazon Bedrock Knowledge Base** | ChromaDB + nomic-embed-text |
| **Claude 3 Haiku (answer generation)** | Mistral 7B via Ollama |
| **Mistral 7B Instruct (evaluation)** | Mistral 7B via Ollama (same model, different prompt) |

| Aurora PostgreSQL pgvector | ChromaDB (local vector DB) |
| :--- | :--- |
| **TITAN_EMBED_TEXT_V2_1024** | nomic-embed-text (768 dimensions) |
| **S3 Document Storage** | Local `data/documents/` folder |
| **Bedrock Guardrails** | Local keyword + LLM-based content filtering |
| **Lambda processing** | Direct Python function calls |
| **AppSync / Next.js frontend** | Gradio web interface (local) |
| **Ping Identity OIDC** | Not needed (local = you're the only user) |

---

## LESSON 9: MEMORY BANK - Persistent, Intelligent Conversation Memory

### The Problem with Lesson 8's Memory

In Lesson 8, we stored conversations in a Python dictionary:

```python
self.conversations: dict[str, list] = {}
```

This has THREE fatal flaws:
1. **Dies on restart** - Close the app, memory is gone forever
2. **No compression** - Long conversations eat up the entire context window
3. **No learning** - Doesn't extract or remember key facts

### The Memory Bank Solution

The new `memory_bank.py` has THREE Layers of memory:

| Layer | What It Stores | How It Works |
| :--- | :--- | :--- |
| **Buffer** | Last 6 exchanges in full detail | Like your short-term memory |
| **Summary** | LLM-compressed older exchanges | Like your long-term memory |
| **Key Facts** | Important facts extracted by LLM | Like notes you write down |

### How It Works

[Image of three-layered memory bank operation flow]

1. **Exchange 7:** Oldest exchanges get SUMMARIZED by the LLM
2. **Exchange 9:** Summary is saved, buffer keeps only recent ones
3. **LL extracts key facts** from recent conversation
4. **Exchange 10+:** Cycle continues - buffer → summary → facts

Everything is saved to `memory_bank/` as JSON files. Restart the app? Memory is still there.

### Step 9.1: Try the standalone Memory Bank demo

```bash
python -m src.memory.memory_bank
```

Chat for a while, then type:
* **stats** - See how memory is structured
* **context** - See exactly what the LLM receives
* **quit** - Exit, then restart and see your memory is STILL THERE


### Step 9.2: Use it in the full chatbot

```bash
python -m src.chatbot
```

New commands available:
* **memory** - Show memory stats (buffer size, summary, key facts)
* **sessions** - List all saved sessions
* **switch <name>** - Switch to a different session
* **forget** - Clear current session memory

### What Maps to What

| AWS Reference | Our Local Version |
| :--- | :--- |
| **S3-backed session storage** | JSON files in `memory_bank/` folder |
| **DynamoDB session management** | File-based session per user |
| **AppSync session context** | Memory Bank's `get_context()` method |
| **Session TTL / cleanup** | `cleanup_old_sessions()` (30-day max) |

## LESSON 10: PROMPT ASSEMBLER - Centralized Prompt Management


### The Prompt Assembler Logic
Instead of hard-coding prompts, the Prompt Assembler fetches components from a registry and constructs the final prompt at runtime.

[Image of prompt assembler architecture and registry pattern]

**Components:**
1. **Persona:** Defines the "who" (e.g., "You are an expert document assistant").
2. **Task Instructions:** The core command (e.g., "Summarize this document").
3. **Context:** Dynamic data retrieved from the vector store.
4. **Guardrails:** Rules injected to prevent prohibited outputs.

### Benefits of Centralized Management
* **Decoupling:** Change prompts without redeploying code.
* **Versioning:** Track prompt history and roll back instantly.
* **Consistency:** Ensure the same persona/rules are applied across all modules.

### Step 10.1: How to use it
In your code, you don't write strings anymore. You fetch them by name:

```python
from src.prompts.assembler import PromptAssembler

assembler = PromptAssembler()
final_prompt = assembler.assemble(
    template_name="chat_bot",
    variables={"user_query": "What is the refund policy?"}
)
```


### How Assembly Works

`Persona` + `Guardrails` + `Task Instructions` + `Context Block` = `Final Prompt`

Change a persona once → every prompt using it updates automatically.
Add a guardrail once → it's injected into every guarded prompt.

### Step 10.1: See all prompts

```bash
python -m src.generation.prompts
```

This shows every registered prompt and its assembled content.

### Step 10.2: Use in your code

```python
from prompt_assembler import prompts

# Get any prompt by name
rag_prompt = prompts.get("rag_with_history")
eval_prompt = prompts.get("eval_combined")

# Customize behavior globally
prompts.customize_persona("document_assistant", "You are a pirate. Arr!")
# Now ALL RAG prompts talk like a pirate.
```


```python
# Add your own prompt from existing blocks
prompts.add_custom_prompt(
    name="my_prompt",
    persona_key="document_assistant",
    task_key="rag_answer",
    human_template="{question}",
    context_block_key="documents"
)
```

### Available Prompts

| Name | Used By | Purpose |
| :--- | :--- | :--- |
| `basic_chat` | Step 1 | Simple chat |
| `rag_simple` | Step 4 | RAG without history |
| `rag_with_history` | Step 5 | RAG with memory |
| `rag_for_evaluation` | Step 5 | RAG for eval testing |
| `eval_combined` | Step 7 | Relevance + groundedness |
| `eval_relevance` | Step 6 | Relevance only |
| `eval_groundedness` | Step 6 | Groundedness only |
| `refine_query` | Step 6 | Query improvement |
| `memory_summarize_new` | Memory Bank | Summarize conversation |
| `memory_summarize_update` | Memory Bank | Update existing summary |
| `memory_extract_facts` | Memory Bank | Extract key facts |
| `memory_demo_chat` | Memory Bank | Demo chat with history |


## LESSON 11: FINE-TUNING WITH QLORA - Make the Model YOUR Expert

### What Is Fine-Tuning?
Mistral 7B knows a little about everything. Fine-tuning trains it further on YOUR data so it becomes an expert in YOUR domain. Like hiring a generalist and training them for your job.

### The Three Approaches

| Type | What It Does | VRAM Needed | For You? |
| :--- | :--- | :--- | :--- |
| **Full Fine-Tuning** | Updates ALL 7B parameters | 80-160GB | No. Enterprise only. |
| **LoRA** | Adds small trainable adapters, freezes the rest | 12-24GB | Maybe. Need good GPU. |
| **QLORA** | LoRA + 4-bit quantized base model | 6-12GB | YES. This is your path. |

### How QLORA Works
* **Original Model:** 7B params × 16-bit = 14GB VRAM needed
* **Quantize to 4-bit:** 7B params × 4-bit = 3.5GB + fits on your GPU

* **Add LoRA adapters:** ~7M trainable params (0.1% of total)
* **Train ONLY adapters** on your data
* **Save adapters:** ~50-200MB (not the full 4GB model)

### When to Fine-Tune vs When to Use RAG

[Image of comparison between RAG and fine-tuning approaches]

| Situation | RAG | Fine-Tune | Both |
| :--- | :--- | :--- | :--- |
| **Answer from specific documents** | YES | - | - |
| **Change the model's writing style** | - | YES | - |
| **Teach domain terminology** | - | YES | - |
| **Keep answers up-to-date** | YES | - | - |
| **Reduce hallucination on your domain** | - | - | YES |

The best approach: Fine-tune for style/format + RAG for content.

### Step 11.1: Install fine-tuning dependencies

```bash
pip install -r requirements-finetune.txt
```

### Step 11.2: Run the fine-tuning script


```bash
python scripts/finetune.py
```

This will:
1. Check your hardware (GPU/VRAM)
2. Create a sample training dataset
3. Show how to load Hugging Face datasets
4. Run QLoRA fine-tuning (if GPU available)
5. Test the fine-tuned model
6. Show how to convert to Ollama format

### Step 11.3: Use in your RAG chatbot

After converting to Ollama format:

```python
# In src/chatbot.py, change one line:
self.generator = ChatOllama(model="my-finetuned-model", temperature=0.3)
```

Your RAG chatbot now uses YOUR fine-tuned model.

---

## LESSON 12: PRODUCTION FEATURES - Auth, Audit, Monitoring, Caching


### What This Adds

Everything that makes a chatbot enterprise-ready, running locally:

| Feature | What It Does | File |
| :--- | :--- | :--- |
| **JWT Authentication** | Users must log in to ask questions | `src/api/auth.py` |
| **Role-Based Access** | employee, hr_admin, admin - different permissions | `src/api/auth.py` |
| **Audit Logging** | Every Q&A recorded with user, timestamp, scores | `src/api/audit.py` |
| **Monitoring Metrics** | Request counts, latency, quality scores, cache rates | `src/api/monitoring.py` |
| **Response Caching** | Repeated questions served instantly (no GPU) | `src/api/server.py` |
| **Rate Limiting** | 30 requests/minute per user | `src/api/server.py` |
| **Health Checks** | /health endpoint for load balancer probes | `src/api/server.py` |
| **Swagger UI** | Auto-generated API docs at /docs | FastAPI built-in |

### Step 12.1: Install production dependencies

```bash
pip install -r requirements-prod.txt
```

### Step 12.2: Start the API server

```bash
cd local-rag-chatbot
python -m src.api.server
```


Open `http://localhost:8000/docs` - full Swagger UI.

### Step 12.3: Test the flow

```bash
# 1. Login (get a token)
curl -X POST http://localhost:8000/login -H "Content-Type: application/json" -d '{"username": "employee1", "password": "emp123"}'

# 2. Ask a question (use the token from step 1)
curl -X POST http://localhost:8000/ask -H "Authorization: Bearer YOUR_TOKEN_HERE" -H "Content-Type: application/json" -d '{"question": "How many days of annual leave do I get?"}'
```

View metrics (requires `hr_admin` or `admin`):
`http://localhost:8000/metrics`
`-H "Authorization: Bearer ADMIN_TOKEN_HERE"`

### Default Users


| Username | Password | Role | Can Do |
| :--- | :--- | :--- | :--- |
| **admin** | admin123 | admin | Everything |
| **hr_manager** | h123_adm | hr_manager | Ask questions + view metrics/audit stats |
| **employee1** | emp123 | employee | Ask questions only |

### What's Local vs What Changes for Production

| Feature | Local (runs today) | Production (documented) |
| :--- | :--- | :--- |
| **Auth** | JWT + JSON user file | SSO/OIDC (Okta, Azure AD) |
| **Audit** | JSONL file | PostgreSQL or Elasticsearch |
| **Monitoring** | In-memory + /metrics | Prometheus + Grafana |
| **Cache** | In-memory dict | Redis cluster |
| **Concurrency** | Single process | Multi-worker, Java/Spring Boot, or Go |
| **Availability** | Single machine | AWS EC2 Auto Scaling + ALB |
| **Vector DB** | ChromaDB (local files) | PostgreSQL pgvector on RDS |
| **LLM serving** | Ollama (single request) | VLLM on GPU EC2 instances |

Full production architecture: `[PRODUCTION_ARCHITECTURE.md]`

## WHAT'S NEXT?

After completing all 18 lessons, you understand:
* Running local AI models (Ollama)

* Running local AI models (Ollama)
* Connecting models with data (LangChain)
* Semantic search with embeddings and vector stores (ChromaDB)
* Retrieval Augmented Generation (RAG)
* Quality evaluation with LLM-as-judge
* The open-source AI ecosystem (Hugging Face)
* Persistent, intelligent conversation memory (Memory Bank)
* Centralized prompt management (Prompt Assembler)
* Fine-tuning models with QLORA on your own data (Lesson 11)
* Production patterns: auth, audit, monitoring, caching (Lesson 12)

These are the same concepts used in production systems on AWS, Azure, and GCP that cost thousands per month. You did it for free, on your laptop.

### Where to Go From Here

1. Add your real documents to `data/documents/` and test with actual use cases
2. Try bigger models for better quality: `ollama pull gemma2` or `ollama pull llama3.2`
3. Deploy the Gradio web UI for your team: `python -m src.chatbot --web`
4. Fine-tune a model on your specific domain using the QLoRA script
5. Add more document formats or connect to databases
6. Build custom prompts using the Prompt Assembler for your specific workflows
7. Explore the advanced topics below

### Advanced Topics - Beyond the 18 Lessons


These are the next frontiers after completing this project. Each builds on what you've learned.

## A. Context Gathering (Intelligent Multi-Step Retrieval)

Our current chatbot does single-shot retrieval: `question` → `vector search` → `top 3 chunks` → `answer`.

A context gatherer does multi-step exploration: `question` → `figure out what's relevant` → `read multiple sources` → `follow cross-references` → `synthesize` → `answer`.

**Why it matters:**
Single-shot retrieval works for self-contained documents (like HR policies). But when answers require combining information from multiple documents, or when documents reference each other, you need a smarter retrieval strategy.

[Image of single-shot retrieval vs multi-step context gathering workflow]

### How it works:

| Feature | SINGLE-SHOT (what we have) | CONTEXT GATHERING (advanced) |
| :--- | :--- | :--- |
| **Step 1** | Question | Analyze question: "What type of info do I need?" |
| **Step 2** | Vector search (1 query) | Search (maybe multiple queries): "Search leave policy AND benefits" |
| **Step 3** | Top 3 chunks | Synthesize information |


* **Step 3:** Read results, follow references: "This doc mentions Policy 4.2, let me get that too"
* **Step 4:** Combine all context
* **Step 5:** Answer with full picture

**LangChain implementation:** This uses LangChain Agents with tools. The agent decides WHEN to search, WHAT to search for, and WHETHER it has enough context to answer.

```python
# Conceptual example (not in our project yet)
from langchain.agents import create_react_agent

tools = [
    vector_search_tool,      # Search documents
    document_reader_tool,    # Read a specific document in full
    metadata_search_tool,    # Search by document title/date/author
]

agent = create_react_agent(llm, tools, prompt)
# Agent decides which tools to use and in what order
```


#### B. Multimodal RAG (Images, Audio, Video, Charts)

**What it is:**
Our chatbot only handles text. Multimodal RAG handles images, PDFs with charts, spreadsheets, audio transcripts, and video - all searchable, all retrievable.

**Why context gathering is ESSENTIAL for multimodal:**
When you have mixed content types, the system needs to decide HOW to process each type:

* **Question:** "What does the org chart show about the HR department?"
* **Context Gatherer (Router):**
    * Detect: "org chart" → this needs IMAGE understanding.
    * Search image embeddings (CLIP model) → Find the org chart image.
    * Use vision model (LLaVA or GPT-4V) to describe the chart → "The org chart shows HR department with 3 teams: Recruitment (5 people), Benefits (3 people), Compliance (4 people)".
    * Also search text documents for HR department info.

* **Final step:** Combine image description + text context → Answer.

**Key models for multimodal:**

| Modality | Embedding Model | Understanding Model |
| :--- | :--- | :--- |
| **Text** | `nomic-embed-text` | `Mistral` |
| **Images** | `CLIP` or `SigLIP` | `LLaVA`, `Llama 3.2 Vision` |
| **Audio** | `Whisper` (transcribe first) | Then use text pipeline |
| **Tables/CSV** | Text embedding on rows | LLM with structured prompt |
| **PDF w/ Charts** | Extract images + text separately | Vision model + text model |

**Ollama supports multimodal models:**

```bash
ollama pull llava            # Vision + Language model
ollama pull llama3.2-vision  # Meta's multimodal model
```

**How it connects to our project:**
* **ChromaDB:** Can store image embeddings alongside text embeddings.
* **Prompt Assembler:** Would need new templates for image-description prompts.
* **Guardrails:** Would need image content safety checks.


* **Memory Bank:** Would track which modalities were used in each exchange.

#### C. LangChain Agents (AI That Uses Tools)

**What it is:**
Our chatbot is a "brain in a jar" - it can think but cannot act. An agent can USE TOOLS: search the web, run code, query databases, call APIs, send emails.

**Why this is powerful AND dangerous:**
* **Powerful:** "Find the latest leave policy, compare it with last year's, and email the diff to HR."
* **Dangerous:** Every tool is an attack surface (Excessive Agency threat from `SECURITY.md`).

**Available agent frameworks:**
* **LangChain Agents:** ReAct pattern.
* **LangGraph:** Stateful, multi-step workflows.
* **CrewAI:** Multi-agent collaboration.
* **AutoGen:** Microsoft's multi-agent framework.

[Image of AI agent tool-use orchestration and attack surface]

#### D. Knowledge Graphs (Structured Relationships)

**What it is:**
Vector search finds SIMILAR documents. Knowledge graphs find RELATED concepts.

**Example:** "Who reports to the VP of HR?"
* Vector search might find documents mentioning the VP.
* A knowledge graph knows the actual reporting structure because it stores entities (`VP of HR`, `Manager X`) and relationships (`reportsTo`, `manages`).


**When to use:**
* **Organizational hierarchies**
* **Policy dependencies** ("Policy A requires compliance with Policy B")
* **Regulatory cross-references**

[Image of graph-based entity-relationship model for policy dependencies]

#### E. Streaming Responses

**What it is:**
Our chatbot waits for the ENTIRE answer before showing anything. Streaming shows tokens as they're generated — like ChatGPT's typing effect.

**LangChain supports this:**

```python
# Instead of .invoke() (waits for full response)
for chunk in chain.stream({"question": question}):
    print(chunk, end='', flush=True)
```

### Full Project Reference

#### 18 Lesson Scripts (`notebooks/lessons/`)

| File | Why It Exists | What You Learn |
| :--- | :--- | :--- |

| File | Why It Exists | What You Learn |
| :--- | :--- | :--- |
| **01** | `01_basic_chat.py` | Foundation — LLM interaction, LangChain chains, prompt templates, Ollama connection |
| **02** | `02_vector_basics.py` | Search requires converting text to numbers — Text → vectors, cosine similarity |
| **03** | `03_ingest_documents.py` | RAG needs documents indexed — File loading, chunking strategy, ChromaDB storage |
| **05** | `05_evaluation.py` | AI can be confidently wrong — LLM-as-judge, relevance scoring, groundedness |
| **06** | `06_huggingface.py` | Ollama has ~10 models, Hugging Face has 500K+ — Transformers, classification |
| **07** | `07_memory_bank.py` | Chatbots need persistent context — 3-layer memory: buffer + summary + key facts |
| **08** | `08_prompt_engineering.py` | Centralized templates ensure maintainability — Personas, guardrail injection |
| **09** | `09_guardrails.py` | AI content safety — 6-category filtering (SEXUAL, VIOLENCE, etc.) |
| **10** | `10_pii_detection.py` | Protect personal data — Anonymization vs. blocking (PII/PHI) |
| **11** | `11_input_sanitization.py` | Defend against injection — Null bytes, homoglyphs, supply chain validation |
| **12** | `12_hybrid_search.py` | Improving retrieval accuracy — BM25 keyword search + cross-encoders |

| 13 | `13_streaming.py` | Improving UX — Token-by-token output, time-to-first-token measurement |
| 14 | `14_fine_tuning.py` | Domain adaptation — QLoRA concepts, training data format, RAG vs. Fine-tuning |
| 15 | `15_human_feedback.py` | Improving over time — Thumbs up/down collection, data export for fine-tuning |
| 16 | `16_rag_monitoring.py` | Truthfulness & Health — Ground truth testing, handling failure types |
| 17 | `17_api_and_auth.py` | Enterprise readiness — JWT auth, RBAC, audit logging, rate limiting |
| 18 | `18_configuration.py` | Maintainability — Centralized `config.yaml`, zero hardcoded values |

## Source Modules (`src/`)

| File | Why It Exists | What It Does |
| :--- | :--- | :--- |
| `src/chatbot.py` | Orchestration | Ties all modules together: sanitize → guardrails → retrieve → generation |


| File | Why It Exists | What It Does |
| :--- | :--- | :--- |
| `src/chatbot.py` | Orchestration | Ties all modules together: sanitize → guardrails → retrieve → generate → evaluate → save |
| `src/config.py` | Configuration | Loads `config.yaml`, provides typed access: `cfg.models.generator`, `cfg.retrieval.top_k` |
| `src/retrieval/hybrid.py` | Hybrid Search | BM25 + vector + reciprocal rank fusion + cross-encoder re-ranking |
| `src/generation/prompts.py` | Prompt Management | 12 templates assembled from reusable parts (persona + guardrail + task + context) |
| `src/guardrails/content_safety.py` | Content Safety | 6 categories, PII anonymize/block, prompt attack detection, output scanning |
| `src/guardrails/model_governance.py` | Model Integrity | SHA-256 checksums, pickle blocking, supply chain validation, input sanitization |
| `src/evaluation/feedback.py` | Feedback Loop | Collects thumbs up/down, exports positive Q&A as fine-tuning training data |
| `src/evaluation/rag_monitor.py` | Quality Monitoring | Quality monitor, ground truth testing, graceful degradation, semantic cache |
| `src/memory/memory_bank.py` | Persistent Memory | 3-layer memory (buffer + LLM summary + extracted facts), saved as JSON |
| `src/api/server.py` | API Infrastructure | FastAPI with JWT auth, RBAC, audit logging, caching, rate limiting, health checks |
| `src/api/auth.py` | Authentication | JWT tokens, bcrypt password hashing, 3 roles (employee, hr_admin, admin) |
| `src/api/audit.py` | Compliance | Thread-safe JSONL logging: user, question, answer, scores, timing |
| `src/api/monitoring.py` | Monitoring | Request counts, latency, cache hit rate, quality scores, auth events |


### Scripts and Notebooks

| File | Why It Exists | What It Does |
| :--- | :--- | :--- |
| `scripts/finetune.py` | Model adaptation | QLoRA fine-tuning with data filtering, validation, and overfitting detection |
| `scripts/training_validator.py` | Data hygiene | Quality filtering, train/val split, before/after model benchmarking |
| `notebooks/colab_finetune.ipynb` | Hardware accessibility | Same fine-tuning workflow on Google Colab's free T4 GPU |

#### Documentation

| File | Why It Exists | What It Covers |
| :--- | :--- | :--- |
| `docs/COMPLETE_GUIDE.md` | Master reference | 27 sections covering everything from core concepts to production |
| `docs/LESSON_PLAN.md` | Learning path | 18 step-by-step lessons with vocabulary and exercises |
| `docs/ARCHITECTURE.md` | System design | Component map, 12-step request flow, AWS mapping |
| `docs/SECURITY.md` | Security & Governance | 15 AI threats, 8 security layers, enterprise governance |
| `docs/ENGINEERING_REVIEW.md` | Quality assessment | 5 fixes applied, 8.6/10 scorecard, scaling limits |
| `docs/PRODUCTION_ARCHITECTURE.md` | Deployment planning | Multi-employee scenario, hardware costs, migration path |
| `docs/TRAINING_DATA_GUIDE.md` | Data sourcing | 13 sources: HuggingFace, Kaggle, SQUAD, PubMed, arXiv, etc. |
| `docs/HARDWARE_REQUIREMENTS.md` | Resource planning | Local, Colab, and production specs with cost estimates |
| `docs/QA.md` | Troubleshooting | 26 detailed Q&As covering every concept |
