"""Primitive memory mode: RedisVL message history, assembled by hand.

This is the "starting point" side of the demo — exactly what the deck's
Part 02 shows. One session store, recency window + optional semantic recall
within the session. No summarization, no durable-fact extraction, no
cross-session recall: a new session starts from zero.
"""

from __future__ import annotations

from typing import Any

import redis

from backend.settings import Settings

# Distinct index names: the semantic and plain variants have incompatible
# schemas, and redisvl refuses to reuse an index whose schema doesn't match.
SEMANTIC_HISTORY_NAME = "memory_demo_primitive_sem"
PLAIN_HISTORY_NAME = "memory_demo_primitive_raw"


class PrimitiveMemoryService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._history: Any = None
        self._semantic = True

    def is_configured(self) -> bool:
        return self.settings.redis_configured() and self.settings.openai_configured()

    def configuration_errors(self) -> list[str]:
        errors = []
        if not self.settings.redis_configured():
            errors.append("REDIS_URL is not set — primitive mode stores turns in Redis via RedisVL.")
        if not self.settings.openai_configured():
            errors.append("OPENAI_API_KEY is not set.")
        return errors

    def _get_history(self) -> Any:
        """One SemanticMessageHistory index; sessions are separated by session_tag."""
        if self._history is not None:
            return self._history
        client = redis.Redis.from_url(self.settings.redis_url, decode_responses=False)
        try:
            from redisvl.extensions.message_history import SemanticMessageHistory
            from redisvl.utils.vectorize import OpenAITextVectorizer

            api_config: dict[str, Any] = {"api_key": self.settings.openai_api_key}
            if self.settings.openai_base_url:
                api_config["base_url"] = self.settings.openai_base_url
            vectorizer = OpenAITextVectorizer(
                model=self.settings.openai_embedding_model,
                api_config=api_config,
            )
            self._history = SemanticMessageHistory(
                name=SEMANTIC_HISTORY_NAME,
                redis_client=client,
                vectorizer=vectorizer,
                distance_threshold=self.settings.primitive_distance_threshold,
            )
            self._semantic = True
        except Exception:
            # Fall back to plain recency-window history if the vectorizer
            # can't initialize (e.g. embeddings endpoint unreachable).
            from redisvl.extensions.message_history import MessageHistory

            self._history = MessageHistory(name=PLAIN_HISTORY_NAME, redis_client=client)
            self._semantic = False
        return self._history

    @property
    def semantic_enabled(self) -> bool:
        return self._semantic

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        self._get_history().add_message({"role": role, "content": content}, session_tag=session_id)

    def get_recent(self, session_id: str, top_k: int | None = None) -> list[dict[str, str]]:
        top_k = top_k or self.settings.primitive_recent_window
        return self._get_history().get_recent(top_k=top_k, session_tag=session_id)

    def get_relevant(self, session_id: str, prompt: str, top_k: int = 5) -> list[dict[str, str]]:
        history = self._get_history()
        if not self._semantic:
            return []
        return history.get_relevant(
            prompt=prompt,
            top_k=top_k,
            session_tag=session_id,
            distance_threshold=self.settings.primitive_distance_threshold,
        )

    def dashboard(self, session_id: str) -> dict[str, Any]:
        if not self.is_configured():
            return {"enabled": False, "messages": [], "errors": self.configuration_errors()}
        try:
            messages = self.get_recent(session_id, top_k=50)
            return {"enabled": True, "session_id": session_id, "messages": messages, "errors": []}
        except Exception as exc:
            return {"enabled": True, "session_id": session_id, "messages": [], "errors": [str(exc)]}

    def clear(self, session_id: str | None = None) -> None:
        # redisvl's clear() wipes the whole history index (all session tags),
        # which is exactly what the demo's reset button wants.
        del session_id
        self._get_history().clear()
