# Teacher Notes — Every Lesson Explained Like You're 10

Written by your chill AI teacher. No jargon without explanation.
No concept without an analogy. No "just trust me" — everything has a WHY.

Read this alongside the lesson scripts. Each section matches a lesson file.

---

## Before We Start: What Are We Even Building?

Imagine you're a new employee at a big company. You have questions:
- "How many vacation days do I get?"
- "Can I work from home?"
- "What's the expense limit?"

You COULD read the entire 200-page employee handbook. Or you could ask someone
who already read it and can find the answer for you instantly.

That "someone" is what we're building. A chatbot that:
1. Reads all your company documents
2. Understands what they say
3. Answers your questions using ONLY those documents
4. Tells you which document the answer came from

It's like having a really smart intern who memorized the entire handbook.

---

## Lesson 01: Basic Chat — Talking to an AI

### The Analogy

Think of an AI model like a really smart parrot. It was trained by reading the
entire internet. It can generate text that SOUNDS human. But it doesn't "know"
things — it predicts what word comes next based on patterns it learned.

When you type "The capital of France is" — the model predicts "Paris" because
in its training data, those words were followed by "Paris" millions of times.

### What Happens in This Lesson

---
You type a question
   ↓
LangChain formats it into a prompt (adds instructions like "be helpful")
   ↓
Sends it to ollama (which runs Mistral on your computer)
   ↓
Mistral generates an answer word by word
   ↓
LangChain extracts the text and shows it to you
---

### The Key Concept: Chains

LangChain's core idea is CHAINS — connecting steps with the pipe `|` operator:

```python
chain = prompt | llm | parser
```

Think of it like a factory assembly line:
- Station 1 (prompt): Formats your question into a proper request
- Station 2 (llm): Generates the answer
- Station 3 (parser): Extracts just the text from the response

Data flows left to right. Each station transforms it.

### Why This Matters

This is the foundation of EVERYTHING we build. Every feature — RAG, evaluation,
memory, guardrails — is built by adding more stations to this assembly line.

### What Could Go Wrong

- Ollama not running → "Connection refused" error. Fix: start ollama.
- Model not pulled → "Model not found." Fix: `ollama pull mistral`
- Slow responses → Normal on CPU. 5-15 seconds. GPU makes it 1-3 seconds.

### Alternatives

Instead of Ollama + Mistral, you could use:
- OpenAI API (GPT-4) — better quality, but costs money and sends data to cloud
- Anthropic API (Claude) — same trade-off
- Hugging Face Transformers — run models directly in Python, more control
- vLLM — production server for serving models to many users

We chose Ollama because it's free, local, and dead simple.

---

## Lesson 02: Embeddings — How AI "Understands" Meaning

### The Analogy

Imagine you could convert any sentence into a GPS coordinate on a map.
Similar sentences would be CLOSE together on the map.
Different sentences would be FAR apart.

"The cat sat on the mat" → coordinate (45.2, 73.8)
"A kitten rested on the rug" → coordinate (45.1, 73.9)  ← CLOSE! Similar meaning.
"Python is a programming language" → coordinate (12.5, 91.3)  ← FAR! Different topic.

That's what embeddings do, except instead of 2 coordinates (latitude, longitude),
they use 768 coordinates (dimensions). More dimensions = more nuance.

### Why This Matters

This is HOW the chatbot finds relevant documents. When you ask "How many vacation
days?" the system converts your question into 768 numbers, then finds which
document chunks have the CLOSEST 768 numbers. "Closest" = most similar meaning.

The magic: "vacation days" and "annual leave" get SIMILAR numbers even though
the words are completely different. The embedding model understands they mean
the same thing.

### The Key Concept: Cosine Similarity

How do you measure if two lists of 768 numbers are "close"?

Imagine two arrows pointing from the center of a circle:
- If they point the same direction → similarity = 1.0 (identical meaning)
- If they point at right angles → similarity = 0.0 (unrelated)
- If they point opposite directions → similarity = -1.0 (opposite meaning)

That's cosine similarity. It measures the ANGLE between two vectors.

### What Could Go Wrong

- Wrong embedding model → poor search results. Different models understand
  different things better.
- Too few dimensions → loses nuance. Too many → slower search.
- Embedding model doesn't know your domain → "FMLA" might not be understood.
  Fix: use a domain-specific embedding model or fine-tune.

### Alternatives

| Model | Dimensions | Speed | Quality |
|-------|------------|-------|---------|
| nomic-embed-text (ours) | 768 | Fast | Good |
| all-MiniLM-L6-v2 | 384 | Fastest | Decent |
| BGE-large-en-v1.5 | 1024 | Slower | Better |
| OpenAI text-embedding-3 | 3072 | API call | Best |

We chose nomic-embed-text because it runs locally via ollama (free, private).

---

## Lesson 03: Document Ingestion — Feeding Documents to the System

### The Analogy

Imagine you're a librarian. Someone gives you 50 books and says "organize these
so people can find answers quickly."

You wouldn't just stack them on a shelf. You'd:
1. Read each book
2. Break it into chapters/sections (CHUNKING)
3. Write a summary card for each section (EMBEDDING)
4. File the cards in a searchable index (VECTOR STORE)

That's exactly what document ingestion does.

### The Three Steps

---
Step 1: LOAD - Read files from the documents/ folder
    PDF → text, DOCX → text, TXT → text, CSV → text

Step 2: CHUNK - Split into smaller pieces
    Why? Models have limited memory (context window).
    A 100-page PDF won't fit. But 100 chunks of 1000 characters each will.

    We use "recursive character splitting":
    - First try to split on paragraph breaks (\n\n)
    - Then on line breaks (\n)
    - Then on spaces
    - Last resort: split mid-word

    Overlap: Each chunk shares 200 characters with the next one.
    Why overlap? So we don't lose meaning at chunk boundaries.
    If a sentence spans two chunks, the overlap catches it.

Step 3: EMBED + STORE - Convert each chunk to numbers and save
    Each chunk → 768 numbers (via nomic-embed-text)
    Numbers stored in ChromaDB (a file-based vector database)
---

### Why Chunk Size Matters

| Chunk Size | Pros | Cons |
|------------|------|------|
| 200 chars | Very precise retrieval | Might miss context |
| 500 chars | Good balance | - |
| 1000 chars (ours)| Good context | Might include irrelevant text |
| 2000 chars | Lots of context | Less precise, uses more tokens |

We use 1000 with 200 overlap — a standard starting point.

### What Could Go Wrong

- Chunks too small → answer is split across chunks, model gets incomplete info
- Chunks too big → irrelevant text dilutes the answer
- No overlap → sentences at chunk boundaries get cut in half
- Wrong file format → loader fails. We support: .txt, .md, .pdf, .csv, .doc, .docx

### Alternatives

Instead of ChromaDB, you could use:
- PostgreSQL + pgvector - production-grade, concurrent access
- Qdrant - purpose-built vector DB, very fast
- Pinecone - cloud-hosted, managed
- FAISS - Facebook's in-memory library, fastest search

We chose ChromaDB because it's file-based (no server needed) and simple.

---

## Lesson 04: RAG Chatbot — The Core Product

### The Analogy

Remember the librarian from Lesson 03? Now someone walks in and asks a question.

The librarian:
1. Hears the question
2. Searches the index cards (vector search)
3. Pulls the 3 most relevant sections
4. Reads them and formulates an answer
5. Tells the person the answer AND which books it came from

That's RAG: Retrieval → Augmented → Generation.

### The Flow

---
User: "How many vacation days do I get?"
   ↓
RETRIEVE: Convert question to vector → search ChromaDB → get top 3 chunks
    Chunk 1: "All full-time employees are entitled to 20 days..."
    Chunk 2: "Leave must be requested 5 business days in advance..."
    Chunk 3: "Unused leave can be carried over, up to 5 days..."
   ↓
AUGMENT: Stuff the chunks into the prompt
    "Based on these documents: [chunk1] [chunk2] [chunk3]
     Answer this question: How many vacation days do I get?"
   ↓
GENERATE: LLM reads the chunks and generates an answer
    "According to the employee handbook, full-time employees
     get 20 days of paid annual leave per year."
---

### Why RAG Instead of Just Asking the LLM?

Without RAG:
---
You: "How many vacation days do I get?"
LLM: "I don't know your company's policy." (or worse, makes one up)
---

The LLM was trained on internet text. It doesn't know YOUR company's policies.
RAG gives it your documents as context, so it answers from YOUR data.

### The Critical Prompt

The system prompt is what makes RAG work:
---
"Answer ONLY from the provided context documents.
 If the context doesn't have the answer, say so.
 Do NOT make up information."
---

Without this instruction, the LLM might ignore the documents and answer from
its general knowledge — which could be wrong for your specific company.

### What Could Go Wrong

- Wrong chunks retrieved → LLM answers from irrelevant text
- No relevant chunks → LLM should say "I don't know" but might hallucinate
- Chunks too short → LLM doesn't have enough context
- LLM ignores the documents → needs stronger prompt instructions

---

## Lesson 05: Evaluation — Is the AI Actually Right?

### The Analogy

You hired that smart intern (the chatbot). They're answering questions all day.
But how do you know they're giving CORRECT answers? You can't check every one.

Solution: Hire a SECOND intern whose only job is to grade the first intern's work.

That's LLM-as-judge. We use the same Mistral model with a DIFFERENT prompt:
- First call: "Answer this question using these documents"
- Second call: "Score this answer on relevance and groundedness"

### Two Metrics

**Relevance** (0.0 - 1.0): Does the answer address the question?
- 1.0: "How many leave days?" → "You get 20 days per year." (perfect)
- 0.5: "How many leave days?" → "Leave policy was updated in 2024." (related but doesn't answer)
- 0.0: "How many leave days?" → "The API rate limit is 100/min." (completely off-topic)

**Groundedness** (0.0 - 1.0): Is the answer supported by the documents?
- 1.0: Every claim in the answer is in the documents
- 0.5: Some claims are in the documents, some are made up
- 0.0: The answer is entirely fabricated (HALLUCINATION)

### Why Both Metrics?

An answer can be relevant but not grounded:
- "You get 30 days of leave" - relevant (about leave) but WRONG (docs say 20)

An answer can be grounded but not relevant:
- "The expense limit is $50" - true (in the docs) but doesn't answer "how many leave days?"

You need BOTH to be high for a good answer.

### The Retry Loop

If the average score is below 0.6, the system automatically:
1. Refines the query (makes it more specific)
2. Retrieves documents again
3. Generates a new answer
4. Re-scores

This is "iterative refinement" — the system self-corrects.

### What Could Go Wrong

- Same model judges itself → might not catch its own mistakes
- LLM judge is inconsistent → scores vary on the same answer
- Threshold too high → rejects good answers
- Threshold too low → accepts bad answers

### Alternatives

- Use a DIFFERENT model for judging (Llama judges Mistral's answers)
- Use human evaluation (most accurate, but slow and expensive)
- Use benchmark datasets (SQuAD, MS MARCO) with known correct answers
- Use multiple judges and average their scores

---

## Lesson 06: Hugging Face — The AI Model Supermarket

### The Analogy

Ollama is like a convenience store — 10-20 models, easy to use, grab and go.
Hugging Face is like Costco — 500,000+ models, datasets, tools. Overwhelming
but has everything you could ever need.

### Three Things Hugging Face Gives Us

1. **Alternative embedding models** — More options than Ollama's ~5 embedding models
2. **Classification models** — Pre-trained models that categorize text (sentiment,
   topic, toxicity) without any training from you
3. **The model zoo** — If you need a model for ANY task, it's probably on Hugging Face

### Zero-Shot Classification (The Cool Part)

Normal classification: Train a model on 10,000 labeled examples → it learns categories.
Zero-shot: Give the model categories it's NEVER seen → it classifies anyway.

```python
classifier("Employees get 20 days of annual leave",
           categories=["HR Policy", "Technical Documentation", "Financial"])
# Result: "HR Policy" (confidence: 0.95)
```

The model was never trained on "HR Policy" as a category. It just UNDERSTANDS
what the words mean and matches them. This is useful for content routing.

### Why This Matters for Our Project

- Content safety: Classify queries as safe/unsafe
- Topic routing: Is this an HR question or a technical question?
- Sentiment analysis: Is the user frustrated?
- Alternative embeddings: Try different models for better search

---

## Lesson 07: Memory Bank — The Chatbot Remembers

### The Analogy

Imagine talking to someone with amnesia. Every sentence, they forget everything
you said before. You'd have to repeat yourself constantly.

That's a chatbot without memory. "How many leave days?" → "20 days."
"Can I carry them over?" → "Carry what over? I don't know what you're talking about."

Memory fixes this. The chatbot remembers the conversation.

### Three Layers (Like Human Memory)

**Buffer** = Short-term memory
- Last 6 exchanges in full detail
- Like remembering what someone said 5 minutes ago

**Summary** = Long-term memory
- Older exchanges compressed by the LLM into a paragraph
- Like remembering "we talked about leave policy yesterday" (not every word)

**Key Facts** = Notes you wrote down
- Important info extracted: "User prefers concise answers", "Company has 20 leave days"
- Persist even when summaries get compressed further

### Why Three Layers?

Models have limited context windows (how much text they can "see" at once).
If you stuff 100 full exchanges into the prompt, it won't fit.

Solution: Keep recent ones in full, compress old ones, extract key facts.
This way the model always has context without overflowing.

### Persistence

Everything saves to `memory_bank/{session_id}.json`. Close the app, reopen it,
your conversation is still there. Like a chat app that syncs.

---

## Lesson 08: Prompt Engineering — The Art of Asking

### The Analogy

Imagine you're training a new employee. You don't just say "answer questions."
You say:
- "You are a helpful HR assistant" (PERSONA - who you are)
- "Only answer from the employee handbook" (TASK - what to do)
- "Never make up information" (GUARDRAILS - what NOT to do)
- "Here are the relevant handbook sections" (CONTEXT - what to use)
- "Here's what we discussed earlier" (HISTORY - conversation memory)

That's prompt engineering. The quality of the AI's answer depends heavily on
HOW you ask, not just WHAT you ask.

### Why Centralize Prompts?

Before: 8 prompts scattered across 5 files. Want to change "be concise" to
"be detailed"? Edit 5 files and hope you don't miss one.

After: 12 prompts in ONE file. Change "be concise" once → applies everywhere.

### The Assembly Pattern

---
Final Prompt = Persona + Guardrails + Task Instructions + Context + History
---

Each part is a reusable building block. Mix and match for different use cases.

---

## Lesson 09: Guardrails — The Safety Net

### The Analogy

A guardrail on a highway doesn't stop you from driving. It stops you from
driving off a cliff. Same idea — guardrails don't stop the AI from answering,
they stop it from answering DANGEROUSLY.

### Six Categories (Matching AWS Bedrock)

| Category | What It Catches | Example |
|----------|-----------------|---------|
| SEXUAL | Explicit content | "Write an erotic story" |
| VIOLENCE | Threats, harm | "How to make a weapon" |
| HATE | Discrimination | "Why are [group] inferior" |
| INSULTS | Personal attacks | "You're an idiot" |
| MISCONDUCT | Illegal activity | "How to hack a system" |
| PROMPT_ATTACK | Jailbreaking | "Ignore all previous rules" |

### Input AND Output

Most people only filter the INPUT (user's question). We filter BOTH:
- Input: Block harmful questions before they reach the LLM
- Output: Block harmful answers before they reach the user

Why output filtering? Because of INDIRECT prompt injection — malicious
instructions hidden inside documents that the LLM retrieves and follows.

---

## Lesson 10: PII Detection — Protecting Personal Data

### The Analogy

Imagine someone walks up to the HR chatbot and says: "My SSN is 123-45-6789,
can you check my benefits?" The chatbot should NOT store, process, or repeat
that SSN. It should replace it with [SSN] before doing anything.

### Two Modes

**Anonymize**: Replace with placeholder, continue processing
- "Email me at john@company.com" → "Email me at [EMAIL]"
- The LLM never sees the real email

**Block**: Reject the entire query
- "My credit card is 4111-1111-1111-1111" → BLOCKED
- Credit cards are too sensitive to even anonymize

### Why This Matters

- Legal: GDPR, CCPA require PII protection
- Security: PII in LLM context could leak via prompt injection
- Audit: PII in logs creates compliance risk
- Memory: PII in conversation memory persists on disk

---

## Lesson 11: Input Sanitization — Defending Against Sneaky Attacks

### The Analogy

You have a bouncer at a club (guardrails). They check IDs and turn away
troublemakers. But what if someone wears a disguise? The bouncer doesn't
recognize them.

Input sanitization is like an X-ray machine at the door — it sees through
disguises. It catches attacks that LOOK innocent but aren't.

### Three Sneaky Attacks

**Null Byte Injection**: Hidden `\x00` characters that can truncate strings
in some systems. The text looks normal but has invisible poison.

**Homoglyph Attack**: Using characters that LOOK identical but are different.
Cyrillic 'a' (U+0430) looks exactly like Latin 'a' (U+0061). So "hack"
bypasses a regex filter for "hack" because the 'a' is Cyrillic.
Our defense: Unicode NFKC normalization converts all look-alikes to standard form.

**Pickle Poisoning**: Python's pickle format can contain executable code.
Loading a malicious .pkl file = running the attacker's code on your machine.
Our defense: Detect pickle by extension AND by magic bytes (catches renamed files).

### Supply Chain Validation

Not all models on the internet are safe. Someone could upload a model to
Hugging Face that contains a backdoor. We maintain an approved sources list —
only models from verified publishers (mistralai/, meta-llama/, google/) are allowed.

---

## Lesson 12: Hybrid Search — Finding the Right Documents

### The Analogy

Imagine searching a library two ways:
1. **By topic** (vector search): "Find books about employee time off"
   → Finds books about vacation, leave, PTO, sabbatical (understands MEANING)
   → But misses "Policy 4.2" because that's a code, not a meaning

2. **By exact words** (keyword search): "Find books containing 'Policy 4.2'"
   → Finds the exact page with "Policy 4.2" on it
   → But misses "annual leave" when you search "vacation" (different words)

Hybrid search does BOTH and combines the results. Best of both worlds.

### Re-Ranking: The Second Opinion

After finding 10 candidate documents, a re-ranker reads each one alongside
your question and says "actually, #7 is more relevant than #1."

It's like asking a subject matter expert to review the librarian's picks.
The librarian (initial search) is fast but approximate. The expert (re-ranker)
is slower but much more accurate.

### Impact: +25-35% Better Answers

That's not a small improvement. One in four questions that got a wrong answer
before now gets a right answer. For an HR chatbot, that's the difference between
"useful" and "frustrating."

---

## Lesson 13: Streaming — The ChatGPT Typing Effect

### The Analogy

Imagine ordering food at a restaurant:
- **Without streaming**: You wait 20 minutes, then the entire meal arrives at once.
  For the first 20 minutes, you're staring at an empty table wondering if they forgot.
- **With streaming**: Appetizer comes in 2 minutes, then bread, then salad, then main.
  You're eating the whole time. Same total wait, but MUCH better experience.

### Why It Matters

Without streaming, users see nothing for 5-15 seconds. They think the app crashed.
With streaming, the first word appears in ~1 second, then more words flow in.
Same total time, but the user SEES progress.

### How It Works

```python
# Without streaming - wait for everything
response = chain.invoke({"question": "What is RAG?"})
print(response)  # Nothing for 10 seconds, then EVERYTHING at once

# With streaming - word by word
for chunk in chain.stream({"question": "What is RAG?"}):
    print(chunk, end="", flush=True)  # Words appear as they're generated
```

The LLM generates one token at a time internally. Streaming just shows each
token as it's produced instead of waiting to collect them all.

---

## Lesson 14: Fine-Tuning — Making the Model YOUR Expert

### The Analogy

You hire a smart generalist (Mistral). They're good at everything but expert
at nothing. Fine-tuning is sending them to a specialized training program.

After training on 500 HR Q&A pairs, they:
- Use your company's terminology ("PTO" instead of "vacation")
- Answer in your company's tone (formal, concise, with policy references)
- Know common follow-up patterns ("Can I carry over?" after "How many days?")

The model's GENERAL knowledge stays. You ADD domain expertise on top.

### QLoRA: The Budget-Friendly Approach

Full fine-tuning = retraining ALL 7 billion parameters. Needs $10,000+ in GPU time.
QLoRA = freeze the model, add tiny adapters (0.1% of parameters), train only those.
Needs a $0 Google Colab account.

It's like teaching someone a new skill without erasing everything they already know.
You're adding a thin layer of expertise, not rebuilding from scratch.

### The Key Insight: Fine-Tune for HOW, RAG for WHAT

- Fine-tuning teaches the model HOW to answer (style, format, terminology)
- RAG provides WHAT to answer with (your actual documents)
- Together = the best possible chatbot for your domain

---

## Lesson 15: Human Feedback — The Improvement Loop

### The Analogy

A restaurant with no reviews never improves. They don't know which dishes are
great and which are terrible. Customer feedback is how they get better.

Same with our chatbot. Without feedback, we're flying blind. The LLM-as-judge
(Lesson 05) is like the chef tasting their own food — useful but biased.
Human feedback is like actual customer reviews — the real truth.

### The Feedback → Fine-Tuning Loop

---
1. Users ask questions → chatbot answers
2. Users rate answers (👍/👎)
3. Collect 500+ thumbs-up Q&A pairs
4. Export as training data
5. Fine-tune the model on these real-world pairs
6. Deploy improved model
7. Collect more feedback → repeat
---

This is simplified RLHF (Reinforcement Learning from Human Feedback).
It's how ChatGPT, Claude, and every major AI product improves over time.

---

## Lesson 16: RAG Monitoring — Is the System Actually Truthful?

### The Analogy

A doctor doesn't just treat patients — they track outcomes. "Did the treatment
work? Are patients getting better or worse over time?"

RAG monitoring is the same. We track: Are answers getting more accurate?
Is the hallucination rate going up? Are users satisfied?

### How to Measure Truthfulness

You can't ask the AI "are you telling the truth?" — it'll say yes.
Instead, measure through multiple signals:

1. **Groundedness score** > 0.7 → answers are supported by documents
2. **Hallucination rate** < 5% → rarely makes things up
3. **Retrieval confidence** > 0.5 → finding relevant documents
4. **Ground truth testing** → compare against KNOWN correct answers
5. **Human feedback correlation** → do users agree with LLM scores?

### Graceful Degradation

When the system CAN'T answer, it should explain WHY — not crash, not hallucinate,
not give a generic "I don't know."

"I found some documents but none are closely related to your question.
Try rephrasing, or contact HR directly. (Confidence: 0.25, threshold: 0.40)"

That's 100x more helpful than "Error 500" or a made-up answer.

---

## Lesson 17: API & Auth — Making It Real

### The Analogy

The chatbot in terminal mode is like a prototype in your garage. It works, but
only you can use it. The API is like opening a store — now anyone can walk in,
but you need:
- A door lock (authentication)
- A security camera (audit logging)
- A cash register (monitoring)
- A "no more than 5 items" sign (rate limiting)

### Three Roles

| Role | Can Do | Can't Do |
|------|--------|----------|
| employee | Ask questions | See metrics, logs, or admin functions |
| hr_admin | Ask questions + view metrics/stats | See full audit logs, re-ingest |
| admin | Everything | Nothing restricted |

### Why Audit Logging?

Compliance. If the chatbot tells an employee "you have unlimited sick days"
(wrong), the company needs to prove what was said, when, and to whom.
Every Q&A is logged with user, timestamp, question, answer, and quality scores.

---

## Lesson 18: Configuration — One File to Rule Them All

### The Analogy

Imagine a car where changing the radio station requires opening the hood and
rewiring the electronics. That's what hardcoded settings are like.

config.yaml is the dashboard — all controls in one place. Change the model,
adjust the retrieval settings, tune the guardrails — all without touching code.

### What You Can Change

```yaml
models:
  generator: "mistral"        # Swap to "my-finetuned-model"
retrieval:
  top_k: 3                    # Retrieve more or fewer documents
  confidence_threshold: 0.4   # How strict the "I don't know" threshold is
guardrails:
  violence: "HIGH"            # Change to "MEDIUM" or "NONE"
```

Edit the file, restart the chatbot, done. No Python editing required.

### Why This Matters

- Different environments need different settings (dev vs prod)
- Non-developers can tune the system (HR admin adjusts guardrail levels)
- A/B testing: try different settings and compare results
- Rollback: revert config.yaml to undo a bad change

---

## The Big Picture: How It All Connects

---

Lesson 01-02: FOUNDATION
    "How do LLMs and embeddings work?"

Lesson 03-04: CORE PRODUCT
    "Load documents, search them, generate answers"

Lesson 05-06: QUALITY + ECOSYSTEM
    "Is the answer good? What other tools exist?"

Lesson 07-08: INTELLIGENCE
    "Remember conversations, manage prompts centrally"

Lesson 09-11: SECURITY
    "Filter harmful content, protect PII, block attacks"

Lesson 12-13: PERFORMANCE
    "Better search, faster responses"

Lesson 14-15: IMPROVEMENT
    "Fine-tune the model, collect feedback"

Lesson 16-18: PRODUCTION
    "Monitor truthfulness, add auth, centralize config"

---

Each layer builds on the previous ones. Skip a layer and the system has a gap.
Complete all 18 and you understand the entire system — from how an LLM generates
text to how a production API serves thousands of users with audit trails.

You didn't just build a chatbot. You built a complete AI application platform.
