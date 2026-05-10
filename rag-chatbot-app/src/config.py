"""
src/config.py
Config class loading config.yaml with dot-access (cfg.models.generator).

Sections: models, retrieval, ingestion, evaluation, memory, api, auth, guardrails
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


class Config:
    """
    Recursive dot-access wrapper around a YAML config dict.

    Usage:
        from src.config import cfg
        model = cfg.models.generator        # "mistral"
        chunk = cfg.ingestion.chunk_size    # 512
    """

    def __init__(self, data: dict) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        result: dict = {}
        for key, value in self.__dict__.items():
            result[key] = value.to_dict() if isinstance(value, Config) else value
        return result

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config({self.to_dict()})"


# Keep DotDict as an alias for backward compatibility
DotDict = Config


def _apply_env_overrides(data: dict) -> dict:
    """Override config values from environment variables."""
    overrides = {
        "OLLAMA_BASE_URL":    ("models",    "ollama_base_url"),
        "CHROMA_PERSIST_DIR": ("ingestion", "persist_directory"),
        "LOG_LEVEL":          ("logging",   "level"),
        "MCP_SERVER_URL":     ("mcp",       "server_url"),
    }
    for env_var, (section, key) in overrides.items():
        value = os.getenv(env_var)
        if value is not None and section in data:
            data[section][key] = value
    return data


def load_config(path: Path | str = _CONFIG_PATH) -> Config:
    """Load YAML config and return a dot-accessible Config object."""
    with open(path, "r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh)
    raw = _apply_env_overrides(raw)
    return Config(raw)


# Module-level singleton
cfg: Config = load_config()
