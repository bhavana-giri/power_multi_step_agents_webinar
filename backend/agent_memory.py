"""Real-time context engine mode: Redis Agent Memory (Redis Cloud).

Working memory (session events + server-side rolling summary) plus long-term
memory (background extraction, hybrid retrieval by meaning + metadata +
recency). API shape mirrors redis/redis-iris-demos' memory service.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx

from backend.settings import Settings

_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


def _sanitize_id(value: str | None, *, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", (value or "").strip()).strip("-")
    return cleaned or fallback


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _extract_items(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    for key in ("items", "memories"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


class AgentMemoryService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.owner_id = _sanitize_id(settings.demo_user_id, fallback="demo-user")
        self.actor_id = _sanitize_id(settings.memory_actor_id, fallback="memory-demo-agent")
        self._client: httpx.AsyncClient | None = None

    def is_configured(self) -> bool:
        return self.settings.agent_memory_configured() and self.settings.openai_configured()

    def configuration_errors(self) -> list[str]:
        errors = []
        if not self.settings.agent_memory_configured():
            errors.append(
                "Agent Memory is not configured — set MEMORY_API_BASE_URL, MEMORY_STORE_ID "
                "and MEMORY_API_KEY (Redis Cloud console → Agent Memory)."
            )
        if not self.settings.openai_configured():
            errors.append("OPENAI_API_KEY is not set.")
        return errors

    # ── plumbing ─────────────────────────────────────────

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=_TIMEOUT)
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    def _url(self, path: str) -> str:
        base = self.settings.memory_api_base_url.rstrip("/")
        return f"{base}/v1/stores/{self.settings.memory_store_id}{path}"

    def _headers(self) -> dict[str, str]:
        api_key = self.settings.memory_api_key
        if not api_key.lower().startswith(("bearer ", "basic ")):
            api_key = f"Bearer {api_key}"
        return {
            "Authorization": api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _raise_for_error(response: httpx.Response, *, allow_424: bool = False) -> None:
        if response.status_code < 400 or (allow_424 and response.status_code == 424):
            return
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        raise RuntimeError(f"Memory API {response.status_code}: {detail}")

    # ── working (session) memory ─────────────────────────

    async def add_event(self, *, session_id: str, role: str, text: str) -> dict[str, Any]:
        # User events are attributed to the owner (as in the Redis Cloud
        # quickstart) so background extraction files memories under the right
        # ownerId; assistant events belong to the agent actor.
        payload = {
            "actorId": self.owner_id if role.upper() == "USER" else self.actor_id,
            "role": role.upper(),
            "content": [{"text": text}],
            "createdAt": _utc_now_iso(),
            "sessionId": session_id,
        }
        response = await self._http().post(
            self._url("/session-memory/events"), headers=self._headers(), json=payload
        )
        self._raise_for_error(response)
        return response.json() if response.content else {}

    async def get_session(self, session_id: str) -> dict[str, Any]:
        response = await self._http().get(
            self._url(f"/session-memory/{session_id}"), headers=self._headers()
        )
        if response.status_code == 404:
            return {}
        self._raise_for_error(response)
        return response.json() if response.content else {}

    # ── long-term memory ─────────────────────────────────

    def _search_filter(self) -> dict[str, Any]:
        filt: dict[str, Any] = {"ownerId": {"eq": self.owner_id}}
        if self.settings.memory_namespace:
            filt["namespace"] = {"eq": self.settings.memory_namespace}
        return filt

    async def search_long_term(
        self,
        *,
        text: str,
        limit: int | None = None,
        similarity_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "text": text,
            "similarityThreshold": similarity_threshold
            if similarity_threshold is not None
            else self.settings.memory_similarity_threshold,
            "filterOp": "all",
            "limit": limit or self.settings.memory_limit,
            # Filter by owner only: memories promoted by background extraction
            # don't carry a custom namespace, so a namespace filter would hide
            # them. Set MEMORY_NAMESPACE to additionally scope direct writes.
            "filter": self._search_filter(),
        }
        response = await self._http().post(
            self._url("/long-term-memory/search"), headers=self._headers(), json=payload
        )
        self._raise_for_error(response, allow_424=True)
        body = response.json() if response.content else {}
        return _extract_items(body)

    async def create_long_term(
        self, *, text: str, topics: list[str] | None = None, session_id: str | None = None
    ) -> dict[str, Any]:
        memory: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "text": text,
            "memoryType": "semantic",
            "ownerId": self.owner_id,
            "sessionId": session_id,
            "topics": topics or [],
        }
        if self.settings.memory_namespace:
            memory["namespace"] = self.settings.memory_namespace
        payload = {"memories": [memory]}
        response = await self._http().post(
            self._url("/long-term-memory"), headers=self._headers(), json=payload
        )
        self._raise_for_error(response)
        return response.json() if response.content else {"ok": True}

    async def delete_all_long_term(self) -> int:
        """Make the demo repeatable: wipe this owner's memories in the namespace."""
        items = await self.search_long_term(text="", limit=100, similarity_threshold=0.0)
        memory_ids = [item.get("id") for item in items if item.get("id")]
        if not memory_ids:
            return 0
        response = await self._http().request(
            "DELETE",
            self._url("/long-term-memory"),
            headers=self._headers(),
            json={"memoryIds": memory_ids},
        )
        self._raise_for_error(response)
        return len(memory_ids)

    # ── dashboard ────────────────────────────────────────

    async def dashboard(self, session_id: str | None) -> dict[str, Any]:
        if not self.is_configured():
            return {
                "enabled": False,
                "working": {},
                "long_term": [],
                "errors": self.configuration_errors(),
            }
        errors: list[str] = []
        working: dict[str, Any] = {}
        long_term: list[dict[str, Any]] = []
        if session_id:
            try:
                working = await self.get_session(session_id)
            except Exception as exc:
                errors.append(f"working memory unavailable: {exc}")
        try:
            long_term = await self.search_long_term(text="", limit=25, similarity_threshold=0.0)
        except Exception as exc:
            errors.append(f"long-term memory unavailable: {exc}")
        return {
            "enabled": True,
            "session_id": session_id,
            "owner_id": self.owner_id,
            "working": working,
            "long_term": long_term,
            "errors": errors,
        }


def event_text(event: dict[str, Any]) -> str:
    content = event.get("content")
    if isinstance(content, list):
        return " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content or "")
