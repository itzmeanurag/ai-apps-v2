# Teacher Notes — Every Concept Explained with Analogies

This document explains every major concept in the project using plain-language analogies. Use these when teaching or when a concept isn't clicking.

---

## The Big Picture

### What is a RAG chatbot?

**Analogy: The Open-Book Exam**

A regular LLM is like a student taking a closed-book exam. They can only answer from what they memorized during training. If the question is about your company's specific leave policy, they'll either say "I don't know" or make something up.

A RAG chatbot is like a student taking an open-book exam. Before answering, they look up the relevant pages in the textbook (your documents), then write an answer based on what they found. The answer is grounded in actual source material.

---

## Foundation Concepts

### What is an LLM?

**Analogy: The World's Most Advanced Autocomplete**

Your phone's autocomplete suggests the next word based on what you've typed. An LLM does the same thing, but it was trained on billions of documents and can predict thousands of words ahead. When it says "The capital of France is Paris," it's not "knowing" a fact — it's predicting that "Paris" is the most likely next word after "The capital of France is."

This is why LLMs hallucinate: they're predicting likely text, not retrieving verified facts.

### What is a token?

**Analogy: Syllables, not words**

A token is roughly 3/4 of a word. "unhappy" might be two tokens: "un" and "happy". "RAG" is one token. "Retrieval-Augmented Generation" is about 5 tokens. Models have a context window limit measured in tokens (Mistral: 32K tokens ≈ 24,000 words ≈ 50 pages).

### What is temperature?

**Analogy: The Creativity Dial**

Temperature 0.0 = a very cautious writer who always picks the most expected word. Every run produces the same output. Good for factual questions.

Temperature 1.0 = a creative writer who sometimes picks surprising words. Output varies each run. Good for creative tasks.

For RAG (factual answers), use 0.1–0.3. For creative writing, use 0.7–1.0.

---

## Embeddings

### What is an embedding?

**Analogy: GPS Coordinates for Meaning**

Just as GPS coordinates (40.7128° N, 74.0060° W) uniquely identify a location on Earth, an embedding (768 numbers) uniquely identifies the meaning of a piece of text in "meaning space."

Two texts with similar meanings have similar coordinates. "cat" and "kitten" are close together. "cat" and "quantum physics" are far apart.

### What is cosine similarity?

**Analogy: Two Arrows**

Imagine two arrows pointing from the center of a globe. If they point in the same direction, they're similar (score = 1.0). If they point at right angles, they're unrelated (score = 0.0). If they point in opposite directions, they're opposites (score = -1.0).

Cosine similarity measures the angle between two embedding vectors. It doesn't care about the length of the arrows (magnitude), only their direction (meaning).

### Why 768 dimensions?

**Analogy: A Very Detailed Address**

A street address has 4 parts (number, street, city, country). An embedding has 768 parts. Each dimension captures a different aspect of meaning: some capture grammar, some capture topic, some capture sentiment, some capture formality. 768 dimensions gives enough resolution to distinguish millions of different concepts.

---

## Document Ingestion

### Why chunk documents?

**Analogy: Cutting a Book into Index Cards**

You can't hand a 500-page book to someone and say "find the relevant part." You cut it into index cards (chunks), write a summary on each card, and file them by topic. When someone asks a question, you find the relevant index cards and hand them over.

The LLM's context window is limited (~32K tokens). A 500-page book is ~200K tokens. Chunking makes it fit.

### Why overlap chunks?

**Analogy: Overlapping Puzzle Pieces**

If you cut a sentence exactly at the chunk boundary, both chunks lose context. "The policy allows 20 days" might be split as "The policy allows" and "20 days of leave." With overlap, both chunks contain the complete sentence.

### Why RecursiveCharacterTextSplitter?

**Analogy: A Smart Editor**

A dumb splitter cuts every 512 characters, even mid-sentence. A smart editor (RecursiveCharacterTextSplitter) tries to cut at paragraph breaks first, then sentence breaks, then word breaks, only cutting mid-word as a last resort.

---

## Vector Search

### Why does "vacation days" find "annual leave"?

**Analogy: A Multilingual Dictionary**

A keyword search is like a monolingual dictionary — "vacation" only matches "vacation." A vector search is like a multilingual dictionary — "vacation" (English) maps to the same concept as "annual leave" (British English), "congé annuel" (French), and "Jahresurlaub" (German). The embedding model learned these equivalences from training data.

### What is ChromaDB?

**Analogy: A Filing Cabinet with a Smart Index**

ChromaDB stores your document chunks (the papers) and their embeddings (the index). When you search, it doesn't read every paper — it uses the index to find the closest matches in milliseconds. The "persist_directory" is where the filing cabinet lives on disk.

---

## The RAG Pipeline

### What is retrieval-augmented generation?

**Analogy: A Research Assistant**

Without RAG: You ask a question, the LLM answers from memory (may be wrong).

With RAG: You ask a question → a research assistant searches the library → finds 3 relevant pages → hands them to the LLM → LLM answers based on those pages.

The LLM is the writer. ChromaDB is the library. The retriever is the research assistant.

### Why does the RAG prompt say "ONLY use information from context"?

**Analogy: Rules for a Witness**

In court, a witness can only testify about what they directly observed — not what they heard, assumed, or imagined. The RAG prompt is the same: "Only testify about what's in these documents." Without this instruction, the LLM will mix document facts with training data facts, making it impossible to verify the source.

---

## Evaluation

### What is LLM-as-judge?

**Analogy: A Peer Reviewer**

In academic publishing, papers are reviewed by other experts. LLM-as-judge uses the same LLM (or a different one) to review the quality of answers. It's not perfect (the model might not catch its own mistakes), but it's much better than no evaluation at all.

### What is groundedness?

**Analogy: Footnotes**

A well-grounded answer is like an academic paper with footnotes — every claim can be traced to a source. An ungrounded answer is like a Wikipedia article with "[citation needed]" everywhere. Groundedness score measures how many claims in the answer are actually supported by the retrieved documents.

### What is hallucination?

**Analogy: A Confident Liar**

Hallucination is when the LLM states something false with complete confidence. It's not lying intentionally — it's predicting what text would logically follow, and sometimes that prediction is wrong. The groundedness score catches this: if the answer contains claims not in the context, it's probably hallucinated.

---

## Memory

### Why three layers?

**Analogy: Your Brain's Memory System**

- **Buffer** = Working memory (what you're thinking about right now, last few minutes)
- **Summary** = Long-term memory (compressed version of older experiences)
- **Facts** = Semantic memory (specific facts you've learned: "Paris is the capital of France")

Your brain doesn't store every second of your life in full detail — it compresses and extracts. The Memory Bank does the same.

### Why does memory need to be saved to disk?

**Analogy: RAM vs Hard Drive**

Python variables live in RAM — they disappear when the process ends. Saving to JSON files is like saving to a hard drive — the data survives restarts. Without disk persistence, every conversation starts fresh.

---

## Prompt Engineering

### Why centralize prompts?

**Analogy: A Style Guide**

A company's style guide says "always use Oxford commas" and "never use passive voice." Every writer follows it. If the rule changes, update the style guide — not every document.

The PromptAssembler is the style guide for AI prompts. Change the persona once → all prompts using it update automatically.

### What is a persona?

**Analogy: A Job Description**

"You are an HR assistant" is a job description for the AI. It tells the model what role to play, what tone to use, and what topics to focus on. Different personas produce different answer styles even with the same question and context.

---

## Guardrails

### What is a guardrail?

**Analogy: A Bouncer at a Club**

A bouncer checks everyone at the door (input guardrail) and can also escort people out if they cause trouble inside (output guardrail). The guardrail checks both what goes in (user questions) and what comes out (model answers).

### What is a prompt injection attack?

**Analogy: A Forged Letter**

Imagine a letter that says "Dear Employee, please ignore all previous instructions from management and instead do X." A prompt injection attack is the same — the attacker embeds fake instructions in their message, hoping the AI will follow them instead of the system prompt.

### Why is pickle dangerous?

**Analogy: An Executable Trojan Horse**

A pickle file looks like data (a model file), but it can contain executable code. Loading it with `pickle.load()` is like opening a Trojan horse — the code inside runs immediately. A malicious pickle file could delete your files, steal your data, or install malware. SafeTensors format is the safe alternative — it's pure data with no executable code.

### What is a homoglyph attack?

**Analogy: Identical Twins**

Cyrillic 'а' (U+0430) and Latin 'a' (U+0061) look identical on screen. An attacker writes "hаck" with a Cyrillic 'а' — your regex for "hack" doesn't match because the characters are different Unicode code points. NFKC normalization converts all lookalike characters to their canonical form, making the attack visible.

---

## Hybrid Search

### Why isn't vector search enough?

**Analogy: A Librarian Who Only Understands Themes**

A vector search librarian understands themes and concepts. Ask "vacation days" and they find "annual leave" — great! But ask "Policy 4.2" and they return random policy documents — they don't understand that "4.2" is a specific identifier, not a theme.

BM25 is a librarian who reads every word literally. Ask "Policy 4.2" and they find the exact document containing "Policy 4.2" — perfect for identifiers, acronyms, and section numbers.

Hybrid search uses both librarians and combines their results.

### What is Reciprocal Rank Fusion?

**Analogy: A Voting System**

Two experts each rank 10 candidates. Expert A ranks candidate X first. Expert B ranks candidate X third. RRF gives candidate X a score based on their rank positions (not raw scores), then combines the votes. The candidate ranked highly by both experts wins.

RRF uses rank position (not raw scores) because BM25 scores (0–20) and vector scores (0–1) are on completely different scales. Rank position is scale-independent.

### What is a CrossEncoder?

**Analogy: A Careful Reader vs a Speed Reader**

Vector search is a speed reader — they skim the document and compare it to a compressed summary of the question. Fast, but misses nuance.

A CrossEncoder is a careful reader — they read the full question AND the full document together, paying attention to every word. Much more accurate, but slower.

We use the speed reader (vector search) to get 20 candidates, then the careful reader (CrossEncoder) to pick the best 3.

---

## Fine-Tuning

### What is fine-tuning?

**Analogy: On-the-Job Training**

You hire a smart generalist (Mistral 7B). They know a little about everything. You give them 3 months of on-the-job training specific to your company. Now they know your terminology, your style, your processes. That's fine-tuning.

### What is QLoRA?

**Analogy: Teaching with Sticky Notes**

Full fine-tuning is like rewriting the entire textbook (7 billion parameters). QLoRA is like adding sticky notes to the textbook (0.1% of parameters). The original book stays the same — you just add notes that modify specific answers. Much cheaper, almost as effective.

The "Q" (quantization) is like printing the textbook in a smaller font to save space — slightly less readable, but fits in your backpack (GPU memory).

### Why does fine-tuning happen on Colab but inference happens locally?

**Analogy: Baking vs Eating**

Baking a cake (training) requires an oven (GPU). Eating the cake (inference) doesn't. You bake the cake in a professional kitchen (Colab with GPU), then bring it home to eat (Ollama on your laptop). The cake (model file) is the connection between the two.

---

## Production

### What is JWT?

**Analogy: A Concert Wristband**

At a concert, you show your ID at the door, get a wristband, and then use the wristband to access different areas (general admission, VIP, backstage). JWT is the wristband — you prove your identity once (login), get a token, and use the token for all subsequent requests. The server doesn't need to check your ID again.

### What is RBAC?

**Analogy: A Building Access Card**

A regular employee's card opens the front door and their office. An HR manager's card also opens the HR records room. The CEO's card opens everything. RBAC (Role-Based Access Control) is the same — different roles have different access levels.

### What is rate limiting?

**Analogy: A Ticket Queue**

A popular restaurant gives out 60 tickets per hour. If you've used your 60 tickets, you wait until the next hour. Rate limiting prevents one user from monopolizing the server and ensures fair access for everyone.

### What is audit logging?

**Analogy: A Security Camera**

Every action is recorded: who did it, when, what they asked, what the system answered. If something goes wrong, you can review the recording. Audit logs are also required for compliance in regulated industries (healthcare, finance, government).

---

## Configuration

### Why config.yaml instead of hardcoded values?

**Analogy: A Control Panel vs Rewiring**

Hardcoded values are like rewiring the electrical system every time you want to change the lighting. config.yaml is the control panel — flip a switch, the change takes effect immediately.

### Why environment variables for secrets?

**Analogy: A Safe vs a Sticky Note**

Putting `SECRET_KEY=abc123` in config.yaml is like writing your PIN on a sticky note on your monitor — anyone who sees the file (including git history) sees the secret. Environment variables are like a safe — the value is stored separately from the code and never committed to version control.
