# Hardware Requirements

Local development, Google Colab, and production specs for the RAG chatbot.

---

## Local Development

### Minimum (runs, but slowly)

| Component | Minimum |
|-----------|---------|
| RAM | 8 GB |
| CPU | 4-core (any modern x86-64) |
| Storage | 10 GB free |
| GPU | Not required |
| OS | Windows 10+, macOS 12+, Ubuntu 20.04+ |

**What works at minimum:**
- Ollama with `mistral` (CPU inference, ~30–60s per response)
- ChromaDB ingestion and search
- All 18 lesson scripts
- Gradio UI

**What doesn't work at minimum:**
- Fine-tuning (needs GPU)
- Fast responses (CPU inference is slow)

---

### Recommended (comfortable development)

| Component | Recommended |
|-----------|-------------|
| RAM | 16 GB |
| CPU | 8-core (Intel i7/i9, AMD Ryzen 7/9) |
| Storage | 20 GB free (SSD preferred) |
| GPU | NVIDIA RTX 3060 (12 GB VRAM) or better |
| OS | Any supported |

**What works at recommended:**
- Everything at minimum, plus:
- Ollama with GPU acceleration (~3–8s per response)
- QLoRA fine-tuning on 7B models
- Comfortable interactive development

---

### Optimal (fast development + fine-tuning)

| Component | Optimal |
|-----------|---------|
| RAM | 32 GB |
| CPU | 12-core+ |
| Storage | 50 GB SSD |
| GPU | NVIDIA RTX 4090 (24 GB VRAM) |
| OS | Linux preferred for GPU drivers |

**What works at optimal:**
- Everything, fast
- Fine-tuning 7B models in ~20 minutes
- Fine-tuning 13B models with QLoRA
- Running multiple models simultaneously

---

### Model Size Reference

| Model | VRAM for Inference | VRAM for QLoRA Fine-Tuning |
|-------|-------------------|---------------------------|
| Mistral 7B (Q4) | 4–5 GB | 6–10 GB |
| Llama 3.1 8B (Q4) | 5–6 GB | 8–12 GB |
| Gemma 2 9B (Q4) | 6–7 GB | 10–14 GB |
| Llama 3.1 13B (Q4) | 8–10 GB | 14–18 GB |
| Llama 3.1 70B (Q4) | 40–45 GB | Not feasible locally |

**Q4 = 4-bit quantized** (what Ollama uses by default). Reduces VRAM by ~4x vs full precision.

---

### Checking Your Hardware

```bash
# Check RAM
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().total / 1e9:.1f} GB')"

# Check GPU (NVIDIA)
nvidia-smi

# Check GPU (Python)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Check Ollama GPU usage
ollama run mistral "hello"
# Watch nvidia-smi in another terminal
```

---

## Google Colab

Google Colab provides free GPU access for fine-tuning. No local GPU needed.

### Colab Tiers

| Tier | GPU | VRAM | RAM | Cost | Session Limit |
|------|-----|------|-----|------|---------------|
| Free | T4 | 16 GB | 12 GB | Free | ~12 hrs/day |
| Colab Pro | L4 | 24 GB | 52 GB | ~$10/month | ~24 hrs/day |
| Colab Pro+ | A100 | 40 GB | 83 GB | ~$50/month | Priority access |

### What Fits on Each Tier

| Model | Free T4 (16 GB) | Pro L4 (24 GB) | Pro+ A100 (40 GB) |
|-------|-----------------|----------------|-------------------|
| Mistral 7B QLoRA | ✅ Easy | ✅ Easy | ✅ Easy |
| Llama 3.1 8B QLoRA | ✅ Easy | ✅ Easy | ✅ Easy |
| Gemma 2 9B QLoRA | ⚠️ Tight (rank=8) | ✅ Easy | ✅ Easy |
| Llama 3.1 13B QLoRA | ❌ OOM | ✅ Easy | ✅ Easy |
| Llama 3.1 70B QLoRA | ❌ OOM | ❌ OOM | ⚠️ Tight |

### Expected Training Times on Colab Free (T4)

| Examples | Epochs | Mistral 7B | Llama 3.1 8B |
|----------|--------|------------|--------------|
| 500 | 3 | ~15 min | ~20 min |
| 1,000 | 3 | ~30 min | ~40 min |
| 5,000 | 3 | ~2 hrs | ~2.5 hrs |
| 50,000 | 3 | ~15 hrs | ~18 hrs |

### Colab Setup

```python
# Cell 1: Check GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Cell 2: Install dependencies
!pip install -q peft transformers bitsandbytes accelerate datasets trl

# Cell 3: Upload training data
from google.colab import files
uploaded = files.upload()  # upload your training.jsonl

# Cell 4: Run fine-tuning
# (see notebooks/colab_finetune.ipynb for full notebook)
```

### Saving the Model from Colab

```python
# Option A: Download directly
model.save_pretrained("/content/finetuned-model")
!zip -r /content/finetuned-model.zip /content/finetuned-model
from google.colab import files
files.download("/content/finetuned-model.zip")

# Option B: Save to Google Drive (recommended for large models)
from google.colab import drive
drive.mount("/content/drive")
model.save_pretrained("/content/drive/MyDrive/finetuned-model")
```

---

## Production Deployment

### Small Production (up to 50 concurrent users)

| Component | Specification | Monthly Cost (AWS) |
|-----------|--------------|-------------------|
| LLM Server | 1x g4dn.xlarge (T4 16GB) | ~$380 |
| App Server | 1x t3.medium (2 vCPU, 4 GB) | ~$30 |
| Vector DB | RDS PostgreSQL t3.medium | ~$50 |
| Cache | ElastiCache Redis t3.micro | ~$15 |
| Load Balancer | ALB | ~$20 |
| Storage | 100 GB EBS | ~$10 |
| **Total** | | **~$505/month** |

**Handles:** ~500 questions/day, 20 concurrent users, 99.5% uptime

---

### Medium Production (50K employees, 5,000 questions/month)

| Component | Specification | Monthly Cost (AWS) |
|-----------|--------------|-------------------|
| LLM Servers | 2x g4dn.2xlarge (T4 16GB) | ~$1,100 |
| App Servers | 2x t3.large (2 vCPU, 8 GB) | ~$120 |
| Vector DB | RDS PostgreSQL r6g.large | ~$200 |
| Cache | ElastiCache Redis r6g.large | ~$150 |
| Load Balancer | ALB | ~$30 |
| Storage | 500 GB EBS | ~$50 |
| Monitoring | CloudWatch + Grafana | ~$50 |
| **Total** | | **~$1,700/month** |

**Handles:** 5,000 questions/month (~170/day), 50 concurrent users, 99.9% uptime

---

### Large Production (enterprise, high availability)

| Component | Specification | Monthly Cost (AWS) |
|-----------|--------------|-------------------|
| LLM Servers | 4x g4dn.4xlarge (T4 16GB) | ~$4,400 |
| App Servers | 4x t3.xlarge (4 vCPU, 16 GB) | ~$480 |
| Vector DB | RDS PostgreSQL r6g.2xlarge (Multi-AZ) | ~$800 |
| Cache | ElastiCache Redis r6g.xlarge (cluster) | ~$600 |
| Load Balancer | ALB + WAF | ~$100 |
| Storage | 2 TB EBS | ~$200 |
| Monitoring | Full observability stack | ~$200 |
| **Total** | | **~$6,780/month** |

**Handles:** 50,000+ questions/month, 200 concurrent users, 99.99% uptime

---

### On-Premises Alternative

For organizations that cannot use cloud:

| Component | Hardware | One-Time Cost |
|-----------|----------|--------------|
| LLM Server | 2x NVIDIA A10G (24 GB VRAM each) | ~$20,000 |
| App Server | 2x Dell PowerEdge R750 | ~$15,000 |
| Storage | NAS with 10 TB | ~$5,000 |
| Networking | 10 GbE switches | ~$3,000 |
| UPS + Rack | APC Smart-UPS | ~$4,000 |
| **Total** | | **~$47,000** |

**Break-even vs cloud:** ~7 months at medium production scale.

---

### LLM Serving: Ollama vs vLLM

| Feature | Ollama | vLLM |
|---------|--------|------|
| Concurrent requests | 1 (queued) | 50+ (batched) |
| Throughput | ~2 req/min | ~30–100 req/min |
| Setup complexity | Very easy | Moderate |
| API format | Custom | OpenAI-compatible |
| GPU utilization | Low | High |
| Best for | Development, 1–5 users | Production, 10+ users |

**Switching from Ollama to vLLM** requires changing 2 lines in `src/chatbot.py`:

```python
# Before (Ollama)
from langchain_ollama import ChatOllama
llm = ChatOllama(model="mistral", temperature=0.3)

# After (vLLM — OpenAI-compatible API)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    base_url="http://vllm-server:8000/v1",
    model="mistral",
    api_key="not-needed",
    temperature=0.3,
)
```

Everything else (RAG pipeline, memory, guardrails) stays the same.

---

### Vector Database: ChromaDB vs PostgreSQL + pgvector

| Feature | ChromaDB | PostgreSQL + pgvector |
|---------|----------|-----------------------|
| Setup | Zero (file-based) | Needs database server |
| Concurrent reads | Limited | Full ACID, unlimited |
| Concurrent writes | Limited | Full ACID, unlimited |
| Backup | Copy folder | Standard DB backup |
| Scale | Up to ~1M vectors | Billions of vectors |
| Best for | Development, single user | Production |

**Switching to pgvector** requires changing the vector store initialization in `src/chatbot.py`:

```python
# Before (ChromaDB)
from langchain_chroma import Chroma
vector_store = Chroma(persist_directory="./data/chroma_db", ...)

# After (pgvector)
from langchain_postgres import PGVector
vector_store = PGVector(
    connection="postgresql://user:pass@host:5432/ragdb",
    collection_name="rag_documents",
    embeddings=embeddings,
)
```
