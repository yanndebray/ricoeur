"""Configuration management for ricoeur."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import tomli_w


DEFAULT_HOME = Path.home() / ".ricoeur"

DEFAULT_CONFIG = {
    "general": {
        "home": str(DEFAULT_HOME),
        "default_language": "en",
    },
    "embeddings": {
        "model": "st:paraphrase-multilingual-mpnet-base-v2",
        "batch_size": 64,
        "device": "auto",
    },
    "topics": {
        "min_cluster_size": 15,
        "n_topics": "auto",
        "label_model": "",
    },
    "summarize": {
        "enabled": False,
        "model": "ollama:llama3.2",
        "max_input_tokens": 2000,
    },
    "mcp": {
        "transport": "stdio",
        "port": 3100,
    },
    "serve": {
        "port": 8080,
        "host": "127.0.0.1",
        "dashboard": True,
    },
}


def get_home() -> Path:
    """Return the ricoeur home directory."""
    env = os.environ.get("RICOEUR_HOME")
    if env:
        return Path(env)
    return DEFAULT_HOME


def config_path() -> Path:
    return get_home() / "config.toml"


def load_config() -> dict[str, Any]:
    """Load config from disk, falling back to defaults."""
    cfg = _deep_copy(DEFAULT_CONFIG)
    path = config_path()
    if path.exists():
        with open(path, "rb") as f:
            user_cfg = tomllib.load(f)
        _deep_merge(cfg, user_cfg)
    return cfg


def save_config(cfg: dict[str, Any]) -> None:
    """Write config to disk."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(cfg, f)


def set_config_value(key: str, value: str) -> dict[str, Any]:
    """Set a dotted key (e.g. 'embeddings.model') and save."""
    cfg = load_config()
    parts = key.split(".")
    if len(parts) == 1:
        # Try to find which section contains the key
        for section in cfg:
            if isinstance(cfg[section], dict) and parts[0] in cfg[section]:
                cfg[section][parts[0]] = _coerce(value)
                save_config(cfg)
                return cfg
        raise KeyError(f"Unknown config key: {key}")
    elif len(parts) == 2:
        section, k = parts
        if section not in cfg:
            raise KeyError(f"Unknown config section: {section}")
        cfg[section][k] = _coerce(value)
        save_config(cfg)
        return cfg
    else:
        raise KeyError(f"Config key too deep: {key}")


def _coerce(value: str) -> Any:
    """Coerce a string value to the appropriate type."""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _deep_copy(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        out[k] = _deep_copy(v) if isinstance(v, dict) else v
    return out


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
