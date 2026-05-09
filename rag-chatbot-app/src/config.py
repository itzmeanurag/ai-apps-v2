"""
config.py – YAML config loader with dot-access (cfg.models.generator)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


class DotDict:
    """Recursive dot-access wrapper around a plain dict."""

    def __init__(self, data: dict) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, DotDict(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        result: dict = {}
        for key, value in self.__dict__.items():
            result[key] = value.to_dict() if isinstance(value, DotDict) else value
        return result

    def __repr__(self) -> str:  # pragma: no cover
        return f"DotDict({self.to_dict()})"


def _apply_env_overrides(data: dict) -> dict:
    """Override config values from environment variables."""
    overrides = {
        "OLLAMA_BASE_URL": ("models", "ollama_base_url"),
        "CHROMA_PERSIST_DIR": ("ingestion", "persist_directory"),
        "LOG_LEVEL": ("logging", "level"),
        "MCP_SERVER_URL": ("mcp", "server_url"),
    }
    for env_var, path in overrides.items():
        value = os.getenv(env_var)
        if value is not None:
            section, key = path
            if section in data:
                data[section][key] = value
    return data


def load_config(path: Path | str = _CONFIG_PATH) -> DotDict:
    """Load YAML config and return a dot-accessible config object."""
    with open(path, "r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh)
    raw = _apply_env_overrides(raw)
    return DotDict(raw)


# Module-level singleton – import as `from src.config import cfg`
cfg: DotDict = load_config()
