# Production Architecture — 50K Employee Deployment Guide

How to scale the RAG chatbot from a local development tool to a production system serving 50,000 employees.

---

## Gap Analysis: Local vs Production

| Area | Local (Current) | Production (Required) |
|------|----------------|----------------------|
| LLM serving | Ollama (1 request at a time) | vLLM (50+ concurrent) |
| Vector DB | ChromaDB (file-based) | PostgreSQL + pgvector |
| Auth | Local JSON user store | Corporate SSO/OIDC |
| Cache | In-memory (lost on restart) | Redis cluster |
| Audit logs | JSONL file | PostgreSQL or Elasticsearch |
| Metrics | In-memory | Prometheus + Grafana |
| Deployment | Single process | Multi-worker + load balancer |
| Availability | Single machine | Multi-AZ, auto-scaling |

---

## Production Architecture Diagram

```
                        ┌─────────────────────────────────────────┐
                        │           CORPORATE NETWORK              │
                        │                                         │
  Employees             │   ┌──────────┐    ┌──────────────────┐  │
  (50,000)  ──HTTPS──▶  │   │   WAF    │──▶ │  Load Balancer   │  │
                        │   │(AWS WAF) │    │  (ALB / nginx)   │  │
                        │   └──────────┘    └────────┬─────────┘  │
                        │                            │            │
                        │              ┌─────────────┼──────────┐ │
                        │              ▼             ▼          ▼ │
                        │   ┌──────────────┐  ┌──────────────┐   │
                        │   │  App Server  │  │  App Server  │   │
                        │   │  (FastAPI)   │  │  (FastAPI)   │   │
                        │   │  Worker 1    │  │  Worker 2    │   │
                        │   └──────┬───────┘  └──────┬───────┘   │
                        │          │                  │           │
                        │    ┌─────┴──────────────────┘           │
                        │    │                                     │
                        │    ▼                                     │
                        │  ┌─────────────────────────────────┐    │
                        │  │         SHARED SERVICES          │    │
                        │  │                                  │    │
                        │  │  ┌──────────┐  ┌─────────────┐  │    │
                        │  │  │  Redis   │  │  PostgreSQL  │  │    │
                        │  │  │  Cache   │  │  + pgvector  │  │    │
                        │  │  └──────────┘  └─────────────┘  │    │
                        │  │                                  │    │
                        │  │  ┌──────────────────────────┐   │    │
                        │  │  │     vLLM GPU Servers      │   │    │
                        │  │  │  (Mistral 7B, 2x T4 GPU)  │   │    │
                        │  │  └──────────────────────────┘   │    │
                        │  │                                  │    │
                        │  │  ┌──────────┐  ┌─────────────┐  │    │
                        │  │  │Corporate │  │  Prometheus  │  │    │
                        │  │  │  SSO     │  │  + Grafana   │  │    │
                        │  │  └──────────┘  └─────────────┘  │    │
                        │  └─────────────────────────────────┘    │
                        └─────────────────────────────────────────┘
```

---

## Component Changes for Production

### 1. LLM Serving: Ollama → vLLM

**Why:** Ollama processes one request at a time. vLLM batches multiple requests together, achieving 30–100x higher throughput.

```bash
# Install vLLM
pip install vllm

# Start vLLM server (on GPU machine)
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --host 0.0.0.0 \
  --port 8080 \
  --max-model-len 8192 \
  --tensor-parallel-size 1  # increase for multi-GPU
```

**Code change in `src/chatbot.py`** (2 lines):
```python
# Before
from langchain_ollama import ChatOllama
llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

# After
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    base_url="http://vllm-server:8080/v1",
    model="mistralai/Mistral-7B-Instruct-v0.3",
    api_key="not-needed",
    temperature=cfg.models.temperature,
)
```

---

### 2. Vector Database: ChromaDB → PostgreSQL + pgvector

**Why:** ChromaDB is file-based and doesn't support concurrent writes. PostgreSQL with pgvector supports full ACID transactions, concurrent access, and standard backup procedures.

```sql
-- PostgreSQL setup
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(768),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX ON document_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
```

**Code change in `src/chatbot.py`**:
```python
# Before
from langchain_chroma import Chroma
vector_store = Chroma(persist_directory="./data/chroma_db", ...)

# After
from langchain_postgres import PGVector
vector_store = PGVector(
    connection=os.getenv("DATABASE_URL"),
    collection_name="rag_documents",
    embeddings=embeddings,
)
```

---

### 3. Authentication: Local JSON → Corporate SSO

**Why:** 50,000 employees already have corporate accounts. They shouldn't need separate credentials for the chatbot.

```python
# api/auth.py — add OIDC support
from authlib.integrations.starlette_client import OAuth

oauth = OAuth()
oauth.register(
    name="corporate",
    client_id=os.getenv("OIDC_CLIENT_ID"),
    client_secret=os.getenv("OIDC_CLIENT_SECRET"),
    server_metadata_url=os.getenv("OIDC_DISCOVERY_URL"),
    client_kwargs={"scope": "openid email profile"},
)

# New endpoint in api/server.py
@app.get("/auth/sso")
async def sso_login(request: Request):
    redirect_uri = request.url_for("sso_callback")
    return await oauth.corporate.authorize_redirect(request, redirect_uri)

@app.get("/auth/sso/callback")
async def sso_callback(request: Request):
    token = await oauth.corporate.authorize_access_token(request)
    user_info = token.get("userinfo")
    # Map corporate role to chatbot role
    role = map_corporate_role(user_info.get("groups", []))
    jwt_token = create_access_token(user_info["sub"], user_info["email"], role)
    return {"access_token": jwt_token, "role": role}
```

---

### 4. Cache: In-Memory → Redis

**Why:** In-memory cache is lost on restart and not shared between multiple app server instances.

```python
# Install: pip install redis
import redis
import json

class RedisSemanticCache:
    def __init__(self, redis_url: str, ttl: int = 3600):
        self.client = redis.from_url(redis_url)
        self.ttl = ttl

    def get(self, key: str) -> str | None:
        value = self.client.get(f"rag:cache:{key}")
        return value.decode() if value else None

    def set(self, key: str, value: str) -> None:
        self.client.setex(f"rag:cache:{key}", self.ttl, value)
```

---

### 5. Audit Logs: JSONL File → PostgreSQL

**Why:** JSONL files can't be queried efficiently. PostgreSQL enables complex audit queries (by user, by date range, by question topic).

```sql
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    event VARCHAR(50) NOT NULL,
    user_id VARCHAR(100),
    role VARCHAR(50),
    ip_address INET,
    question TEXT,
    answer_preview TEXT,
    quality_score FLOAT,
    cached BOOLEAN,
    latency_ms INTEGER,
    status VARCHAR(20),
    metadata JSONB
);

-- Query examples
SELECT user_id, COUNT(*) as questions, AVG(quality_score) as avg_quality
FROM audit_log
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY user_id
ORDER BY questions DESC;
```

---

### 6. Metrics: In-Memory → Prometheus + Grafana

**Why:** In-memory metrics are lost on restart and can't be visualized or alerted on.

```python
# Install: pip install prometheus-client
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
REQUEST_COUNT = Counter("rag_requests_total", "Total requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("rag_request_duration_seconds", "Request latency", ["endpoint"])
QUALITY_SCORE = Gauge("rag_quality_score", "Average quality score")
CACHE_HIT_RATE = Gauge("rag_cache_hit_rate", "Cache hit rate")

# Start metrics server (separate port)
start_http_server(9090)

# Record metrics
REQUEST_COUNT.labels(endpoint="/ask", status="200").inc()
REQUEST_LATENCY.labels(endpoint="/ask").observe(4.2)
QUALITY_SCORE.set(0.74)
```

**Grafana dashboard panels:**
- Request rate (req/min)
- Error rate (%)
- P95/P99 latency
- Quality score trend
- Cache hit rate
- Active users

---

## Deployment: Docker + Kubernetes

### Docker Compose (staging)

```yaml
# docker-compose.prod.yml
version: "3.9"
services:
  app:
    build: .
    ports: ["8000:8000"]
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/ragdb
      - REDIS_URL=redis://redis:6379
      - VLLM_URL=http://vllm:8080
      - SECRET_KEY=${SECRET_KEY}
    depends_on: [postgres, redis, vllm]
    deploy:
      replicas: 2

  vllm:
    image: vllm/vllm-openai:latest
    command: ["--model", "mistralai/Mistral-7B-Instruct-v0.3"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: ragdb
      POSTGRES_USER: user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes: ["pgdata:/var/lib/postgresql/data"]

  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru

volumes:
  pgdata:
```

### Kubernetes (production)

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-chatbot
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-chatbot
  template:
    spec:
      containers:
      - name: app
        image: your-registry/rag-chatbot:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        env:
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: secret-key
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-chatbot-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-chatbot
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## Capacity Planning for 50K Employees

### Usage Assumptions

- 5,000 questions/month = ~170/day = ~7/hour (business hours)
- Peak: 3x average = ~21 questions/hour = ~0.35/minute
- Average response time: 8 seconds
- Concurrent users at peak: ~5

### Infrastructure Sizing

| Component | Sizing | Reasoning |
|-----------|--------|-----------|
| App servers | 2x t3.large | 2 workers each, handles 20 concurrent HTTP connections |
| vLLM server | 1x g4dn.xlarge (T4 16GB) | Handles 5 concurrent LLM requests with batching |
| PostgreSQL | r6g.large (2 vCPU, 16 GB) | Vector search + audit logs |
| Redis | r6g.medium (2 vCPU, 8 GB) | Session cache + rate limiting |

### Scaling Triggers

| Metric | Scale Up When | Scale Down When |
|--------|--------------|-----------------|
| CPU utilization | > 70% for 5 min | < 30% for 15 min |
| Request queue depth | > 10 requests | < 2 requests |
| P95 latency | > 30 seconds | < 10 seconds |
| Error rate | > 5% | < 1% |

---

## Security Hardening for Production

```bash
# 1. Rotate SECRET_KEY (use AWS Secrets Manager)
aws secretsmanager create-secret \
  --name rag-chatbot/secret-key \
  --secret-string "$(openssl rand -hex 32)"

# 2. Enable WAF rules
aws wafv2 create-web-acl \
  --name rag-chatbot-waf \
  --rules file://waf-rules.json

# 3. Enable VPC (no public internet access to DB/cache)
# All services in private subnets
# Only ALB in public subnet

# 4. Enable encryption at rest
# RDS: encrypted storage
# Redis: encryption at rest + in transit
# S3 (for model files): SSE-S3

# 5. Enable CloudTrail for all API calls
aws cloudtrail create-trail \
  --name rag-chatbot-trail \
  --s3-bucket-name your-audit-bucket
```

---

## Monitoring and Alerting

### Key Alerts

```yaml
# alertmanager rules
groups:
- name: rag-chatbot
  rules:
  - alert: HighErrorRate
    expr: rate(rag_requests_total{status=~"5.."}[5m]) > 0.05
    for: 2m
    annotations:
      summary: "Error rate above 5%"

  - alert: LowQualityScore
    expr: rag_quality_score < 0.5
    for: 10m
    annotations:
      summary: "Average quality score below 0.5"

  - alert: HighLatency
    expr: histogram_quantile(0.99, rag_request_duration_seconds) > 60
    for: 5m
    annotations:
      summary: "P99 latency above 60 seconds"

  - alert: LLMServerDown
    expr: up{job="vllm"} == 0
    for: 1m
    annotations:
      summary: "vLLM server is down"
```

---

## Rollout Strategy

### Phase 1: Pilot (Month 1)
- Deploy to 100 employees in one department
- Collect feedback, monitor quality
- Fix issues before wider rollout

### Phase 2: Department Rollout (Month 2–3)
- Expand to 5,000 employees (10%)
- Enable human feedback collection
- Begin fine-tuning on collected data

### Phase 3: Full Rollout (Month 4–6)
- All 50,000 employees
- Fine-tuned model deployed
- Full monitoring and alerting active

### Phase 4: Optimization (Ongoing)
- Monthly fine-tuning cycles
- A/B testing of model versions
- Continuous quality improvement
