"""Configuration helpers for loading environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_ENV_CACHE: dict[str, str] | None = None


def load_env(env_file: str | os.PathLike[str] = ".env") -> dict[str, str]:
    """Load key-value pairs from a dotenv-style file.

    The loader is intentionally simple to avoid adding a dependency on
    `python-dotenv`. It reads the file once per process and caches the
    result. Lines starting with ``#`` or blank lines are ignored, and values
    can be quoted with single or double quotes.
    """

    global _ENV_CACHE

    if _ENV_CACHE is not None:
        return _ENV_CACHE.copy()

    path = Path(env_file)
    if not path.exists():
        _ENV_CACHE = {}
        return {}

    env: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        env[key] = value

    # Update process environment for compatibility with existing code.
    for key, value in env.items():
        os.environ.setdefault(key, value)

    _ENV_CACHE = env.copy()
    return env.copy()


def _split_rpc_urls(raw: str) -> list[str]:
    parts = [segment.strip() for segment in raw.split(",")]
    return [segment for segment in parts if segment]


def get_rpc_urls(env: Optional[dict[str, str]] = None) -> list[str]:
    """Return all configured Sui RPC URLs, preserving declared order."""

    env = env or load_env()
    candidates: list[str] = []

    primary = env.get("SUI_RPC_URL")
    if primary:
        candidates.append(primary)

    alternate = env.get("SUI_FULLNODE_URL")
    if alternate:
        candidates.append(alternate)

    raw_list = env.get("SUI_RPC_URLS")
    if raw_list:
        candidates.extend(_split_rpc_urls(raw_list))

    # Preserve order while removing duplicates.
    deduped = list(dict.fromkeys(candidates))
    if not deduped:
        raise RuntimeError(
            "Unable to determine Sui RPC endpoint. Please set SUI_RPC_URL in .env"
        )
    return deduped


def get_rpc_url(env: Optional[dict[str, str]] = None) -> str:
    """Return the first configured Sui RPC URL (backwards compatible helper)."""

    return get_rpc_urls(env)[0]
