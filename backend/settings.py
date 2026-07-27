"""Environment-driven settings for the agent memory demo."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# A blank OPENAI_BASE_URL= line in .env would otherwise be picked up by the
# openai client itself and break every request with a protocol-less URL.
if not os.environ.get("OPENAI_BASE_URL", "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _float(name: str, default: float) -> float:
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL"))
    openai_chat_model: str = field(default_factory=lambda: _env("OPENAI_CHAT_MODEL", "gpt-4o"))
    openai_embedding_model: str = field(
        default_factory=lambda: _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )

    redis_url: str = field(default_factory=lambda: _env("REDIS_URL"))

    memory_api_base_url: str = field(default_factory=lambda: _env("MEMORY_API_BASE_URL"))
    memory_store_id: str = field(default_factory=lambda: _env("MEMORY_STORE_ID"))
    memory_api_key: str = field(default_factory=lambda: _env("MEMORY_API_KEY"))
    memory_namespace: str = field(default_factory=lambda: _env("MEMORY_NAMESPACE", "agent-memory-demo"))
    memory_actor_id: str = field(default_factory=lambda: _env("MEMORY_ACTOR_ID", "memory-demo-agent"))

    demo_user_id: str = field(default_factory=lambda: _env("DEMO_USER_ID", "demo-user"))

    memory_similarity_threshold: float = field(
        default_factory=lambda: _float("MEMORY_SIMILARITY_THRESHOLD", 0.7)
    )
    memory_limit: int = field(default_factory=lambda: _int("MEMORY_LIMIT", 6))
    primitive_recent_window: int = field(default_factory=lambda: _int("PRIMITIVE_RECENT_WINDOW", 8))
    primitive_distance_threshold: float = field(
        default_factory=lambda: _float("PRIMITIVE_DISTANCE_THRESHOLD", 0.35)
    )

    backend_port: int = field(default_factory=lambda: _int("BACKEND_PORT", 8060))

    def openai_configured(self) -> bool:
        return bool(self.openai_api_key)

    def redis_configured(self) -> bool:
        return bool(self.redis_url)

    def agent_memory_configured(self) -> bool:
        return bool(self.memory_api_base_url and self.memory_store_id and self.memory_api_key)
