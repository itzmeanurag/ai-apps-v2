"""
LESSON 18: Configuration Management — config.yaml + DotDict
============================================================
CONCEPT: All settings in one place; no hardcoded values in code

WHAT THIS DOES:
  Demonstrates the config system from src/config.py:
    - Load config.yaml with dot-access (cfg.models.generator)
    - Environment variable overrides (.env file)
    - Runtime config inspection
    - How to change settings without editing Python code

WHY THIS MATTERS:
  Without centralized config, settings are scattered across 10+ files.
  Want to switch models? Edit 5 files. Change chunk size? Hunt through code.
  With config.yaml, change one value → restart → done.
  Environment variables override config for secrets (SECRET_KEY, API keys).

RUN (from project root):
  python notebooks/lessons/18_configuration.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the config singleton — same object used everywhere in the project
from src.config import cfg, load_config, DotDict


# ── Demo 1: Dot-access config ─────────────────────────────────────────────────

def demo_dot_access() -> None:
    """Show how dot-access works."""
    print("\n--- DEMO 1: Dot-Access Configuration ---")
    print("  cfg.models.generator  instead of  config['models']['generator']\n")

    print(f"  cfg.models.generator        = {cfg.models.generator}")
    print(f"  cfg.models.embedder         = {cfg.models.embedder}")
    print(f"  cfg.models.temperature      = {cfg.models.temperature}")
    print(f"  cfg.models.max_tokens       = {cfg.models.max_tokens}")
    print(f"  cfg.models.ollama_base_url  = {cfg.models.ollama_base_url}")

    print(f"\n  cfg.retrieval.top_k         = {cfg.retrieval.top_k}")
    print(f"  cfg.retrieval.rerank_top_k  = {cfg.retrieval.rerank_top_k}")
    print(f"  cfg.retrieval.rrf_k         = {cfg.retrieval.rrf_k}")

    print(f"\n  cfg.ingestion.chunk_size    = {cfg.ingestion.chunk_size}")
    print(f"  cfg.ingestion.chunk_overlap = {cfg.ingestion.chunk_overlap}")
    print(f"  cfg.ingestion.persist_directory = {cfg.ingestion.persist_directory}")

    print(f"\n  cfg.api.port                = {cfg.api.port}")
    print(f"  cfg.api.rate_limit_per_minute = {cfg.api.rate_limit_per_minute}")

    print(f"\n  cfg.guardrails.enable_content_safety = {cfg.guardrails.enable_content_safety}")
    print(f"  cfg.guardrails.enable_pii_detection  = {cfg.guardrails.enable_pii_detection}")

    print(f"\n  cfg.memory.buffer_size      = {cfg.memory.buffer_size}")
    print(f"  cfg.memory.summary_threshold = {cfg.memory.summary_threshold}")
    print(f"  cfg.memory.session_dir      = {cfg.memory.session_dir}")


# ── Demo 2: All config sections ───────────────────────────────────────────────

def demo_all_sections() -> None:
    """Show all configuration sections."""
    print("\n--- DEMO 2: All Configuration Sections ---")

    sections = [
        ("models",     "LLM and embedding model settings"),
        ("retrieval",  "Search and retrieval parameters"),
        ("ingestion",  "Document loading and chunking"),
        ("evaluation", "Quality thresholds and caching"),
        ("memory",     "Conversation memory settings"),
        ("api",        "FastAPI server settings"),
        ("auth",       "JWT and user management"),
        ("guardrails", "Content safety settings"),
        ("logging",    "Audit and metrics logging"),
        ("mcp",        "MCP server integration"),
    ]

    for section, description in sections:
        section_cfg = getattr(cfg, section, None)
        if section_cfg:
            values = section_cfg.to_dict()
            print(f"\n  [{section}]  — {description}")
            for k, v in values.items():
                print(f"    {k}: {v}")


# ── Demo 3: Environment variable overrides ────────────────────────────────────

def demo_env_overrides() -> None:
    """Show how environment variables override config.yaml."""
    print("\n--- DEMO 3: Environment Variable Overrides ---")
    print("""
  config.yaml sets defaults.
  .env file (or shell environment) overrides specific values.
  This keeps secrets out of config.yaml (which goes in git).

  OVERRIDE MAPPING:
    OLLAMA_BASE_URL  → cfg.models.ollama_base_url
    CHROMA_PERSIST_DIR → cfg.ingestion.persist_directory
    LOG_LEVEL        → cfg.logging.level
    MCP_SERVER_URL   → cfg.mcp.server_url

  SECRETS (never in config.yaml):
    SECRET_KEY       → JWT signing key (set in .env)
    ADMIN_PASSWORD   → Default admin password (set in .env)
    MCP_API_KEY      → MCP server API key (set in .env)

  EXAMPLE .env file:
    SECRET_KEY=your-super-secret-key-min-32-chars
    OLLAMA_BASE_URL=http://gpu-server:11434
    LOG_LEVEL=WARNING

  HOW TO USE:
    cp .env.example .env
    # Edit .env with your values
    # Python-dotenv loads it automatically on import
""")


# ── Demo 4: Common configuration changes ─────────────────────────────────────

def demo_common_changes() -> None:
    """Show the most common config changes users make."""
    print("\n--- DEMO 4: Common Configuration Changes ---")
    print(f"""
  CHANGE 1: Switch to a fine-tuned model
    # In config.yaml:
    models:
      generator: "my-finetuned-model"  # was "mistral"

  CHANGE 2: Retrieve more documents per question
    # In config.yaml:
    retrieval:
      top_k: 7        # was {cfg.retrieval.top_k}
      rerank_top_k: 5 # was {cfg.retrieval.rerank_top_k}

  CHANGE 3: Larger chunks for longer documents
    # In config.yaml:
    ingestion:
      chunk_size: 1024    # was {cfg.ingestion.chunk_size}
      chunk_overlap: 128  # was {cfg.ingestion.chunk_overlap}
    # Then re-ingest: python notebooks/lessons/03_ingest_documents.py

  CHANGE 4: Stricter quality threshold
    # In config.yaml:
    evaluation:
      quality_threshold: 0.75  # was {cfg.evaluation.quality_threshold}

  CHANGE 5: Longer conversation memory
    # In config.yaml:
    memory:
      buffer_size: 20       # was {cfg.memory.buffer_size}
      summary_threshold: 15 # was {cfg.memory.summary_threshold}

  CHANGE 6: Disable guardrails (development only!)
    # In config.yaml:
    guardrails:
      enable_content_safety: false  # was true
      enable_pii_detection: false   # was true

  CHANGE 7: Point to a different Ollama server
    # In .env:
    OLLAMA_BASE_URL=http://192.168.1.100:11434

  After any change: restart the chatbot or API server.
  No Python code changes needed.
""")


# ── Demo 5: Loading a custom config ──────────────────────────────────────────

def demo_custom_config() -> None:
    """Show how to load a custom config file."""
    print("\n--- DEMO 5: Loading a Custom Config ---")

    import yaml
    import tempfile

    # Create a minimal custom config
    custom_config = {
        "models": {
            "generator": "llama3.1",
            "embedder": "nomic-embed-text",
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "ollama_base_url": "http://localhost:11434",
            "temperature": 0.5,
            "max_tokens": 512,
        },
        "retrieval": {"top_k": 3, "rerank_top_k": 2, "bm25_weight": 0.4,
                      "vector_weight": 0.6, "rrf_k": 60, "similarity_threshold": 0.3},
        "ingestion": {"chunk_size": 256, "chunk_overlap": 32,
                      "persist_directory": "./data/chroma_db",
                      "collection_name": "custom_collection",
                      "supported_extensions": [".txt"]},
        "evaluation": {"enable_auto_eval": False, "truthfulness_threshold": 0.6,
                       "quality_threshold": 0.5, "semantic_cache_threshold": 0.92},
        "memory": {"buffer_size": 5, "summary_threshold": 4, "facts_max": 20,
                   "session_dir": "./data/sessions", "enable_summarization": False},
        "api": {"host": "0.0.0.0", "port": 8001, "workers": 1,
                "rate_limit_per_minute": 30, "cache_ttl_seconds": 60,
                "max_request_size_kb": 32, "cors_origins": []},
        "auth": {"token_expire_minutes": 30, "algorithm": "HS256",
                 "users_file": "./data/users.json", "roles": ["employee", "admin"]},
        "guardrails": {"enable_content_safety": True, "enable_pii_detection": True,
                       "enable_pii_anonymization": True, "enable_llm_classification": False,
                       "enable_output_scanning": True, "enable_model_governance": True,
                       "checksum_file": "./data/model_checksums.json"},
        "logging": {"level": "DEBUG", "audit_log_file": "./data/audit.jsonl",
                    "metrics_file": "./data/metrics.json", "format": "json"},
        "mcp": {"enabled": False, "server_url": "http://localhost:8080",
                "timeout_seconds": 5, "retry_attempts": 2},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(custom_config, f)
        temp_path = f.name

    custom_cfg = load_config(temp_path)
    print(f"  Custom config loaded from temp file.")
    print(f"  custom_cfg.models.generator = {custom_cfg.models.generator}")
    print(f"  custom_cfg.api.port         = {custom_cfg.api.port}")
    print(f"  custom_cfg.ingestion.chunk_size = {custom_cfg.ingestion.chunk_size}")

    Path(temp_path).unlink()
    print(f"\n  Usage: from src.config import load_config, cfg")
    print(f"         cfg = load_config('my_custom_config.yaml')")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("LESSON 18: CONFIGURATION MANAGEMENT")
    print("=" * 60)
    print(f"""
Config system (src/config.py + config.yaml):
  from src.config import cfg
  model = cfg.models.generator      # "mistral"
  chunk = cfg.ingestion.chunk_size  # 512

DotDict: recursive dot-access wrapper around YAML dict.
Environment variables override config.yaml values.
""")

    demo_dot_access()
    demo_all_sections()
    demo_env_overrides()
    demo_common_changes()
    demo_custom_config()

    print("\n" + "=" * 60)
    print("CONGRATULATIONS — You've completed all 18 lessons!")
    print("=" * 60)
    print(f"""
You now understand:
  01. LangChain chain pattern (prompt | llm | parser)
  02. Embeddings and cosine similarity
  03. Document ingestion pipeline
  04. RAG retrieval + generation
  05. LLM-as-judge evaluation
  06. Hugging Face ecosystem
  07. 3-layer persistent memory
  08. Centralized prompt management
  09. Content safety guardrails
  10. PII detection and anonymization
  11. Input sanitization and model governance
  12. Hybrid search (BM25 + vector + RRF + CrossEncoder)
  13. Token-by-token streaming
  14. QLoRA fine-tuning concepts
  15. Human feedback collection
  16. RAG quality monitoring
  17. Production API with JWT + RBAC
  18. Centralized configuration management

Next steps:
  - Run the full chatbot: python app.py
  - Start the API: uvicorn api.server:app --port {cfg.api.port}
  - Fine-tune: python scripts/finetune.py
  - Read: docs/COMPLETE_GUIDE.md
""")


if __name__ == "__main__":
    main()
