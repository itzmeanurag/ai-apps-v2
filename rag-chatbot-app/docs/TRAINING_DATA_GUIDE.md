# Training Data Guide — 13 Sources with Code Examples

How to collect, prepare, and validate training data for fine-tuning the RAG chatbot.

---

## Overview

Fine-tuning needs high-quality question-answer pairs in JSONL format:

```jsonl
{"instruction": "How many leave days?", "context": "...", "response": "20 days per year.", "quality": "good"}
{"instruction": "Can I work from home?", "context": "...", "response": "Yes, 3 days/week.", "quality": "good"}
```

**Minimum:** 100 examples for style changes, 1,000+ for domain knowledge.

---

## Source 1: Human Feedback from Production

The best training data comes from real users rating real answers.

```python
# Export positively-rated Q&A pairs from the feedback store
from evaluation.feedback import FeedbackStore

store = FeedbackStore("./data/feedback.jsonl")

# Export only 4-5 star ratings
n = store.export_training_data(
    output_path="./data/training_from_feedback.jsonl",
    only_positive=True,
    min_rating=4,
)
print(f"Exported {n} high-quality examples")
```

**Why it's the best:** Real questions from real users, rated by real humans.
**Minimum needed:** 200+ positive ratings before fine-tuning is worthwhile.

---

## Source 2: Your Own Documents (Synthetic Q&A)

Generate Q&A pairs from your existing documents using the LLM.

```python
import json
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="mistral", temperature=0.7)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Generate 5 question-answer pairs from the document below. "
               "Return JSON array: [{\"question\": \"...\", \"answer\": \"...\"}]"),
    ("human", "Document:\n{document}"),
])
chain = qa_prompt | llm | StrOutputParser()

# Load your documents
documents_dir = Path("./data/documents")
examples = []

for doc_file in documents_dir.glob("*.txt"):
    content = doc_file.read_text(encoding="utf-8")
    # Process in chunks to avoid context overflow
    chunks = [content[i:i+2000] for i in range(0, len(content), 1500)]

    for chunk in chunks:
        try:
            result = chain.invoke({"document": chunk})
            start = result.find("[")
            end = result.rfind("]") + 1
            if start >= 0:
                pairs = json.loads(result[start:end])
                for pair in pairs:
                    examples.append({
                        "instruction": pair["question"],
                        "context": chunk,
                        "response": pair["answer"],
                        "quality": "synthetic",
                    })
        except Exception as e:
            print(f"Error: {e}")

# Save
with open("./data/training_synthetic.jsonl", "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"Generated {len(examples)} synthetic examples")
```

**Quality note:** Synthetic data is good for style but may contain errors. Always review before training.

---

## Source 3: Hugging Face Datasets — General Instruction Following

```python
from datasets import load_dataset
import json

# Alpaca: 52K general instruction-following examples
dataset = load_dataset("tatsu-lab/alpaca", split="train[:2000]")

examples = []
for item in dataset:
    if item["input"]:  # has context
        examples.append({
            "instruction": item["instruction"],
            "context": item["input"],
            "response": item["output"],
            "quality": "good",
        })
    else:
        examples.append({
            "instruction": item["instruction"],
            "context": "",
            "response": item["output"],
            "quality": "good",
        })

with open("./data/training_alpaca.jsonl", "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"Loaded {len(examples)} Alpaca examples")
```

**Best for:** Teaching the model to follow instructions in a specific format.

---

## Source 4: Hugging Face Datasets — High-Quality Conversations

```python
from datasets import load_dataset
import json

# OpenHermes: 1M+ high-quality conversations
dataset = load_dataset("teknium/OpenHermes-2.5", split="train[:5000]")

examples = []
for item in dataset:
    conversations = item.get("conversations", [])
    if len(conversations) >= 2:
        # Take first user-assistant pair
        user_msg = next((c["value"] for c in conversations if c["from"] == "human"), None)
        asst_msg = next((c["value"] for c in conversations if c["from"] == "gpt"), None)
        if user_msg and asst_msg:
            examples.append({
                "instruction": user_msg,
                "context": "",
                "response": asst_msg,
                "quality": "good",
            })

with open("./data/training_openhermes.jsonl", "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"Loaded {len(examples)} OpenHermes examples")
```

---

## Source 5: SQuAD — Reading Comprehension Q&A

```python
from datasets import load_dataset
import json

# SQuAD: 100K+ reading comprehension Q&A pairs
dataset = load_dataset("rajpurkar/squad", split="train[:3000]")

examples = []
for item in dataset:
    if item["answers"]["text"]:
        examples.append({
            "instruction": item["question"],
            "context": item["context"],
            "response": item["answers"]["text"][0],
            "quality": "good",
        })

with open("./data/training_squad.jsonl", "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"Loaded {len(examples)} SQuAD examples")
```

**Best for:** Teaching the model to answer questions from context (exactly what RAG needs).

---

## Source 6: MS MARCO — Real Search Queries

```python
from datasets import load_dataset
import json

# MS MARCO: Real Bing search queries with answers
dataset = load_dataset("ms_marco", "v2.1", split="train[:2000]")

examples = []
for item in dataset:
    if item["answers"] and item["answers"][0] != "No Answer Present.":
        # Use the first passage as context
        passages = item.get("passages", {})
        passage_texts = passages.get("passage_text", [])
        if passage_texts:
            examples.append({
                "instruction": item["query"],
                "context": passage_texts[0][:1000],
                "response": item["answers"][0],
                "quality": "good",
            })

with open("./data/training_msmarco.jsonl", "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"Loaded {len(examples)} MS MARCO examples")
```

**Best for:** Teaching the model to handle real-world search-style questions.

---

## Source 7: HotpotQA — Multi-Hop Reasoning

```python
from datasets import load_dataset
import json

# HotpotQA: Questions requiring reasoning across multiple documents
dataset = load_dataset("hotpot_qa", "distractor", split="train[:1000]")

examples = []
for item in dataset:
    # Combine supporting facts as context
    context_parts = []
    for title, sentences in zip(
        item["context"]["title"],
        item["context"]["sentences"]
    ):
        context_parts.append(f"{title}: {' '.join(sentences)}")
    context = "\n\n".join(context_parts[:3])  # top 3 passages

    examples.append({
        "instruction": item["question"],
        "context": context[:2000],
        "response": item["answer"],
        "quality": "good",
    })

with open("./data/training_hotpotqa.jsonl", "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"Loaded {len(examples)} HotpotQA examples")
```

**Best for:** Teaching multi-step reasoning across multiple documents.

---

## Source 8: Domain-Specific — HR Policies

```python
import json

# Manually curated HR Q&A pairs
# These are the highest-quality examples for an HR chatbot
hr_examples = [
    {
        "instruction": "How many days of annual leave do full-time employees receive?",
        "context": "Full-time employees are entitled to 20 days of paid annual leave per calendar year.",
        "response": "Full-time employees receive 20 days of paid annual leave per calendar year. "
                   "Part-time employees working 20-39 hours per week receive leave at 50% of the full-time rate.",
        "quality": "good",
    },
    {
        "instruction": "What is the remote work policy?",
        "context": "Remote work is permitted up to 3 days per week with manager approval.",
        "response": "Employees may work remotely up to 3 days per week. This requires manager approval "
                   "and must be documented in the HR system. Employees must be available during core hours "
                   "(10 AM - 3 PM local time) and maintain a reliable internet connection.",
        "quality": "good",
    },
    {
        "instruction": "What expenses require a receipt?",
        "context": "Business expenses over $50 require a receipt and manager sign-off.",
        "response": "Any business expense over $50 requires both a receipt and manager approval. "
                   "Expenses under $50 can be submitted without a receipt, though keeping receipts "
                   "is always recommended.",
        "quality": "good",
    },
]

with open("./data/training_hr_domain.jsonl", "w") as f:
    for ex in hr_examples:
        f.write(json.dumps(ex) + "\n")

print(f"Created {len(hr_examples)} domain-specific HR examples")
```

---

## Source 9: Wikipedia (General Knowledge)

```python
from datasets import load_dataset
import json

# Wikipedia: Use for general knowledge fine-tuning
# Load a small subset to avoid downloading the full dataset
dataset = load_dataset("wikipedia", "20220301.en", split="train[:500]", trust_remote_code=True)

examples = []
for item in dataset:
    # Create simple Q&A from Wikipedia articles
    title = item["title"]
    text = item["text"][:1000]  # first 1000 chars

    examples.append({
        "instruction": f"What is {title}?",
        "context": text,
        "response": text[:300],  # first paragraph as answer
        "quality": "synthetic",
    })

with open("./data/training_wikipedia.jsonl", "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"Loaded {len(examples)} Wikipedia examples")
```

---

## Source 10: PubMed (Medical Domain)

```python
from datasets import load_dataset
import json

# PubMed QA: Medical question answering
dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train[:500]", trust_remote_code=True)

examples = []
for item in dataset:
    context = " ".join(item["context"]["contexts"][:3])
    examples.append({
        "instruction": item["question"],
        "context": context[:2000],
        "response": item["long_answer"],
        "quality": "good",
    })

with open("./data/training_pubmed.jsonl", "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"Loaded {len(examples)} PubMed examples")
```

**Best for:** Medical or healthcare domain chatbots.

---

## Source 11: arXiv (Technical/Research Domain)

```python
from datasets import load_dataset
import json

# arXiv: Research paper abstracts
dataset = load_dataset("arxiv_dataset", split="train[:500]", trust_remote_code=True)

examples = []
for item in dataset:
    if item.get("abstract") and item.get("title"):
        examples.append({
            "instruction": f"Summarize this research paper: {item['title']}",
            "context": item["abstract"],
            "response": item["abstract"][:500],
            "quality": "synthetic",
        })

with open("./data/training_arxiv.jsonl", "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"Loaded {len(examples)} arXiv examples")
```

---

## Source 12: SEC EDGAR (Financial/Legal Domain)

```python
import json
import urllib.request

# SEC EDGAR: Public company filings
# Example: Download Apple's 10-K filing
url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=AAPL&type=10-K&dateb=&owner=include&count=1&search_text="

# For actual use, download filings and create Q&A pairs
# This is a template showing the data format
sec_examples = [
    {
        "instruction": "What were Apple's total revenues in fiscal 2023?",
        "context": "Apple Inc. total net sales for fiscal 2023 were $383.3 billion, "
                  "compared to $394.3 billion in fiscal 2022.",
        "response": "Apple's total net sales for fiscal 2023 were $383.3 billion, "
                   "a decrease from $394.3 billion in fiscal 2022.",
        "quality": "good",
    },
]

with open("./data/training_sec.jsonl", "w") as f:
    for ex in sec_examples:
        f.write(json.dumps(ex) + "\n")

print(f"Created {len(sec_examples)} SEC EDGAR examples")
```

---

## Source 13: BEIR Benchmark (Retrieval Evaluation)

```python
from datasets import load_dataset
import json

# BEIR: Benchmark for Information Retrieval
# Contains multiple domain-specific datasets
dataset = load_dataset("BeIR/nfcorpus", "queries", split="test[:200]", trust_remote_code=True)

examples = []
for item in dataset:
    examples.append({
        "instruction": item["text"],
        "context": "",  # BEIR provides separate corpus
        "response": "Please refer to the relevant medical literature.",
        "quality": "synthetic",
    })

with open("./data/training_beir.jsonl", "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"Loaded {len(examples)} BEIR examples")
```

---

## Combining Multiple Sources

```python
import json
from pathlib import Path

def combine_training_files(
    input_files: list[str],
    output_file: str,
    max_per_source: int = 1000,
    shuffle: bool = True,
) -> int:
    import random

    all_examples = []
    for filepath in input_files:
        path = Path(filepath)
        if not path.exists():
            print(f"Skipping {filepath} (not found)")
            continue
        examples = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        # Cap per source to avoid imbalance
        if len(examples) > max_per_source:
            examples = random.sample(examples, max_per_source)
        all_examples.extend(examples)
        print(f"  {filepath}: {len(examples)} examples")

    if shuffle:
        random.shuffle(all_examples)

    with open(output_file, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nCombined: {len(all_examples)} total examples → {output_file}")
    return len(all_examples)

# Example usage
combine_training_files(
    input_files=[
        "./data/training_from_feedback.jsonl",
        "./data/training_synthetic.jsonl",
        "./data/training_squad.jsonl",
        "./data/training_hr_domain.jsonl",
    ],
    output_file="./data/training_combined.jsonl",
    max_per_source=500,
)
```

---

## Validating Training Data

Always validate before fine-tuning:

```bash
python scripts/training_validator.py \
  --data_path ./data/training_combined.jsonl \
  --min_examples 100 \
  --check_pii \
  --verbose
```

The validator checks:
- Required fields (`instruction`, `response`)
- Text length constraints
- Duplicate detection
- Quality distribution
- PII in training data

---

## Data Quality Guidelines

| Criterion | Requirement |
|-----------|-------------|
| Accuracy | Every answer must be factually correct |
| Completeness | Answers should be complete, not truncated |
| Format | Consistent style across all examples |
| Diversity | Cover all topics the model will encounter |
| Balance | Roughly equal coverage of each topic area |
| No PII | Remove all personal information |
| No duplicates | Each question should be unique |

---

## Dataset Size Guidelines

| Task | Minimum | Recommended | Diminishing Returns |
|------|---------|-------------|---------------------|
| Style/format change | 50–100 | 200–500 | 1,000+ |
| Domain Q&A | 200–500 | 1,000–5,000 | 10,000+ |
| Instruction following | 1,000 | 5,000–20,000 | 50,000+ |
| General capability | 10,000+ | 50,000–200,000 | 500,000+ |
