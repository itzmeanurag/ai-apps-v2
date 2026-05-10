# rag-chatbot-app

A production-grade RAG (Retrieval-Augmented Generation) chatbot built with:

- **LLM**: Mistral 7B via [Ollama](https://ollama.ai) (local, private)
- **Embeddings**: nomic-embed-text via Ollama
- **Vector Store**: ChromaDB (persistent)
- **Retrieval**: Hybrid BM25 + vector search + RRF + CrossEncoder re-ranking
- **API**: FastAPI with JWT auth, RBAC, rate limiting, audit logging
- **UI**: Gradio web interface
- **Memory**: 3-layer persistent memory (buffer + summary + facts)
- **Guardrails**: Content safety, PII detection/anonymization, model governance
- **Evaluation**: Quality monitoring, semantic caching, human feedback

---

## Quick Start

### 1. Install Ollama and pull models

```bash
# Install Ollama: https://ollama.ai
ollama pull mistral
ollama pull nomic-embed-text
```

### 2. Set up Python environment

```bash
python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements-prod.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env – set SECRET_KEY at minimum
```

### 4. Ingest documents

```bash
python -c "
from src.chatbot import RAGChatbot
bot = RAGChatbot()
bot.ingest('./data/documents')
print('Done!')
"
```

### 5. Start the API

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Start the Gradio UI

```bash
python app.py
# Open http://localhost:7860
```

---

## API Usage

### Authenticate

```bash
curl -X POST http://localhost:8000/auth/token \
  -d "username=admin&password=admin123"
```

### Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the PTO policy?"}'
```

### Check health

```bash
curl http://localhost:8000/health
```

---

## Project Structure

```
rag-chatbot-app/
├── src/
│   ├── chatbot.py          # Main RAG orchestrator
│   └── config.py           # YAML config loader (dot-access)
├── api/
│   ├── server.py           # FastAPI app (JWT, RBAC, rate limiting)
│   ├── auth.py             # JWT + BCrypt + user store
│   ├── audit.py            # Thread-safe JSONL audit logger
│   └── monitoring.py       # Metrics tracker
├── guardrails/
│   ├── content_safety.py   # 6-category filter + PII detection
│   └── model_governance.py # Checksums + pickle detection + sanitization
├── retrieval/
│   └── hybrid.py           # BM25 + vector + RRF + CrossEncoder
├── generation/
│   └── prompts.py          # PromptAssembler with 12 templates
├── memory/
│   └── memory_bank.py      # 3-layer persistent memory
├── evaluation/
│   ├── feedback.py         # Human feedback + training data export
│   └── rag_monitor.py      # Quality monitor + semantic cache
├── scripts/
│   ├── finetune.py         # QLoRA fine-tuning skeleton
│   └── training_validator.py
├── data/
│   └── documents/          # Sample documents
├── app.py                  # Gradio UI
├── mcp_client.py           # MCP server client
├── config.yaml             # All configuration
├── .env.example            # Environment variable template
├── requirements.txt        # Base dependencies
├── requirements-prod.txt   # + FastAPI/uvicorn/jose/passlib
├── requirements-finetune.txt # + transformers/peft/bitsandbytes
└── pyproject.toml
```

---

## Roles & Permissions

| Endpoint | employee | hr_admin | admin |
|----------|----------|----------|-------|
| POST /ask | ✅ | ✅ | ✅ |
| POST /feedback | ✅ | ✅ | ✅ |
| GET /metrics | ❌ | ✅ | ✅ |
| GET /audit | ❌ | ❌ | ✅ |
| POST /ingest | ❌ | ❌ | ✅ |

Default credentials (change in production!):
- `admin` / `admin123` → admin role
- `hr_user` / `hr_password` → hr_admin role
- `employee1` / `emp_password` → employee role

---

## Fine-tuning

Export feedback as training data:

```bash
python -c "
from evaluation.feedback import FeedbackStore
store = FeedbackStore('./data/feedback.jsonl')
n = store.export_training_data('./data/training.jsonl', only_positive=True)
print(f'Exported {n} examples')
"
```

Validate training data:

```bash
python scripts/training_validator.py --data_path ./data/training.jsonl --verbose
```

Run QLoRA fine-tuning:

```bash
pip install -r requirements-finetune.txt
python scripts/finetune.py \
  --model_name mistralai/Mistral-7B-v0.1 \
  --data_path ./data/training.jsonl \
  --output_dir ./models/finetuned \
  --epochs 3
```

---

## MCP Integration

To connect to a Java MCP server:

```python
from mcp_client import McpClient, McpAugmentedGenerator, McpConfig
from src.chatbot import RAGChatbot

config = McpConfig(server_url="http://localhost:8080", api_key="your-key")
client = McpClient(config)
bot = RAGChatbot()
augmented = McpAugmentedGenerator(bot, client)

result = augmented.ask("What is the vacation policy?")
print(result["answer"])
```

---

## License

MIT
