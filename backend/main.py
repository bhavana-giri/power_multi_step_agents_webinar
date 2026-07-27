"""FastAPI app: Primitive Memory (RedisVL) vs Real-time Context Engine (Agent Memory).

The client sends only the newest user message. Each mode reconstructs the
conversation context from its own memory layer, so the difference between the
two approaches is visible rather than papered over by a client-side transcript.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel

from backend.agent_memory import AgentMemoryService, event_text
from backend.catalog import catalog_prompt_block
from backend.primitive_memory import PrimitiveMemoryService
from backend.settings import Settings

settings = Settings()
app = FastAPI(title="Redis Agent Memory Demo")

primitive = PrimitiveMemoryService(settings)
agent_memory = AgentMemoryService(settings)

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

SYSTEM_PROMPT = (
    "You are the Redis Motors shopping assistant, helping one customer choose a car.\n"
    f"{catalog_prompt_block()}\n\n"
    "Rules:\n"
    "- Recommend only vehicles from the inventory above.\n"
    "- Be concise: 2-4 sentences, or a short list when comparing models.\n"
    "- Honor any user preferences, budgets, or decisions provided in the "
    "memory context sections below. Mention that you remembered them when relevant.\n"
    "- If the user asks what you remember and no memory context is provided, say "
    "plainly that you have no record of previous conversations. Never invent memories."
)


# ── demo script: prefilled queries matching the deck's demo slides ──

STARTER_GROUPS = [
    {
        "label": "Session state",
        "eyebrow": "Live demo 07",
        "hint": "Working memory holds recent turns — run these in order.",
        "chips": [
            {"title": "Hybrid SUVs", "prompt": "Show me hybrid SUVs."},
            {"title": "Under $35k", "prompt": "Keep it under $35k."},
            {"title": "Best mileage of those?", "prompt": "Which of those has the best mileage?"},
        ],
    },
    {
        "label": "Durable facts",
        "eyebrow": "Live demo 08",
        "hint": "A confirmed preference gets promoted to long-term memory.",
        "chips": [
            {
                "title": "Remember my budget",
                "prompt": "Remember this: I only want hybrids, and my budget is $35k max.",
            },
            {
                "title": "Top pick decided",
                "prompt": "I've decided — the RAV4 Hybrid is my top pick. Remember that.",
            },
        ],
    },
    {
        "label": "Relevant history",
        "eyebrow": "Live demo 09",
        "hint": "A hybrid query pulls the right memories back into the prompt.",
        "chips": [
            {
                "title": "My preferences?",
                "prompt": "What do you remember about my car preferences?",
            },
            {
                "title": "Cars I considered",
                "prompt": "Which cars did I consider, and what was my budget?",
            },
        ],
    },
    {
        "label": "Session two: recall",
        "eyebrow": "In practice",
        "hint": "Click “New session” in the top bar first, then ask.",
        "chips": [
            {
                "title": "I'm back — recommend",
                "prompt": "I'm back! Based on what you know about me, which car should I buy today?",
            },
        ],
    },
]

MODES = {
    "primitive": {
        "id": "primitive",
        "label": "Primitive Memory",
        "sublabel": "RedisVL message history",
        "description": (
            "Raw turns stored per session with RedisVL MessageHistory. Recency window "
            "+ in-session semantic recall. No summaries, no durable facts, no cross-session memory."
        ),
    },
    "context_engine": {
        "id": "context_engine",
        "label": "Real-time Context Engine",
        "sublabel": "Redis Agent Memory",
        "description": "",
    },
}


class ChatRequest(BaseModel):
    message: str
    mode: str = "context_engine"
    session_id: str


class ResetRequest(BaseModel):
    session_id: str | None = None


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _openai_client() -> AsyncOpenAI:
    kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return AsyncOpenAI(**kwargs)


async def _stream_llm(
    chat_messages: list[dict[str, str]],
) -> AsyncIterator[str]:
    client = _openai_client()
    stream = await client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=chat_messages,
        stream=True,
        temperature=0.4,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta


def _timer() -> float:
    return time.perf_counter()


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


# ── primitive mode turn ──────────────────────────────────


async def _primitive_turn(req: ChatRequest) -> AsyncIterator[str]:
    errors = primitive.configuration_errors()
    if errors:
        yield _sse({"type": "error", "message": " ".join(errors)})
        yield _sse({"type": "done"})
        return

    yield _sse({"type": "status", "text": "Reading session history (RedisVL)…"})

    loop = asyncio.get_running_loop()
    start = _timer()
    try:
        recent = await loop.run_in_executor(None, lambda: primitive.get_recent(req.session_id))
    except Exception as exc:
        yield _sse({"type": "error", "message": f"RedisVL message history failed: {exc}"})
        yield _sse({"type": "done"})
        return
    yield _sse(
        {
            "type": "event",
            "name": "message_history.get_recent",
            "kind": "memory",
            "durationMs": _ms(start),
            "payload": {"session": req.session_id, "returned": len(recent), "messages": recent},
        }
    )

    relevant: list[dict[str, str]] = []
    if primitive.semantic_enabled:
        start = _timer()
        try:
            relevant = await loop.run_in_executor(
                None, lambda: primitive.get_relevant(req.session_id, req.message)
            )
            yield _sse(
                {
                    "type": "event",
                    "name": "message_history.get_relevant",
                    "kind": "memory",
                    "durationMs": _ms(start),
                    "payload": {
                        "prompt": req.message,
                        "returned": len(relevant),
                        "messages": relevant,
                    },
                }
            )
        except Exception as exc:
            yield _sse(
                {
                    "type": "event",
                    "name": "message_history.get_relevant",
                    "kind": "memory",
                    "durationMs": _ms(start),
                    "payload": {"error": str(exc)},
                }
            )

    recent_contents = {m.get("content") for m in recent}
    older_relevant = [m for m in relevant if m.get("content") not in recent_contents]

    system = SYSTEM_PROMPT
    system += (
        "\n\nMemory context: only the CURRENT session's message history below. "
        "There is no long-term memory in this mode."
    )
    if older_relevant:
        block = "\n".join(f"- {m.get('role')}: {m.get('content')}" for m in older_relevant)
        system += f"\n\nSemantically relevant earlier turns (this session only):\n{block}"

    chat_messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for m in recent:
        role = m.get("role", "user")
        chat_messages.append(
            {"role": role if role in ("user", "assistant") else "user", "content": m.get("content", "")}
        )
    chat_messages.append({"role": "user", "content": req.message})

    yield _sse({"type": "status", "text": "Generating answer…"})
    answer = ""
    start = _timer()
    try:
        async for delta in _stream_llm(chat_messages):
            answer += delta
            yield _sse({"type": "delta", "text": delta})
    except Exception as exc:
        yield _sse({"type": "error", "message": f"OpenAI request failed: {exc}"})
        yield _sse({"type": "done"})
        return
    yield _sse(
        {
            "type": "event",
            "name": "openai.chat",
            "kind": "llm",
            "durationMs": _ms(start),
            "payload": {"model": settings.openai_chat_model, "context_messages": len(chat_messages)},
        }
    )

    start = _timer()
    try:
        await loop.run_in_executor(None, lambda: primitive.add_turn(req.session_id, "user", req.message))
        await loop.run_in_executor(None, lambda: primitive.add_turn(req.session_id, "assistant", answer))
        yield _sse(
            {
                "type": "event",
                "name": "message_history.add_message",
                "kind": "memory",
                "durationMs": _ms(start),
                "payload": {"stored": 2, "note": "Raw turns only — nothing is summarized or extracted."},
            }
        )
    except Exception as exc:
        yield _sse(
            {
                "type": "event",
                "name": "message_history.add_message",
                "kind": "memory",
                "payload": {"error": str(exc)},
            }
        )
    yield _sse({"type": "done"})


# ── context engine mode turn ─────────────────────────────


async def _context_engine_turn(req: ChatRequest) -> AsyncIterator[str]:
    errors = agent_memory.configuration_errors()
    if errors:
        yield _sse({"type": "error", "message": " ".join(errors)})
        yield _sse({"type": "done"})
        return

    yield _sse({"type": "status", "text": "Fetching working memory…"})
    start = _timer()
    session_payload: dict[str, Any] = {}
    try:
        session_payload = await agent_memory.get_session(req.session_id)
    except Exception as exc:
        yield _sse(
            {
                "type": "event",
                "name": "working_memory.get",
                "kind": "memory",
                "payload": {"error": str(exc)},
            }
        )
    events = session_payload.get("events", []) if isinstance(session_payload, dict) else []
    summary = session_payload.get("summary") if isinstance(session_payload, dict) else None
    yield _sse(
        {
            "type": "event",
            "name": "working_memory.get",
            "kind": "memory",
            "durationMs": _ms(start),
            "payload": {
                "session": req.session_id,
                "event_count": len(events),
                "summary": summary,
                "recent_events": [
                    {"role": e.get("role"), "text": event_text(e)} for e in events[-6:]
                ],
            },
        }
    )

    yield _sse({"type": "status", "text": "Hybrid search over long-term memory…"})
    start = _timer()
    memories: list[dict[str, Any]] = []
    try:
        memories = await agent_memory.search_long_term(text=req.message)
        yield _sse(
            {
                "type": "event",
                "name": "long_term_memory.search",
                "kind": "memory",
                "durationMs": _ms(start),
                "payload": {
                    "query": req.message,
                    "filters": agent_memory._search_filter(),
                    "returned": len(memories),
                    "memories": memories,
                },
            }
        )
    except Exception as exc:
        yield _sse(
            {
                "type": "event",
                "name": "long_term_memory.search",
                "kind": "memory",
                "durationMs": _ms(start),
                "payload": {"error": str(exc)},
            }
        )

    system = SYSTEM_PROMPT
    if summary:
        system += f"\n\nWorking memory summary (this session):\n{summary}"
    if memories:
        lines = []
        for m in memories:
            text = m.get("text", "")
            topics = m.get("topics") or []
            suffix = f" [topics: {', '.join(map(str, topics))}]" if topics else ""
            lines.append(f"- {text}{suffix}")
        system += "\n\nLong-term memories about this user (from Redis Agent Memory):\n" + "\n".join(lines)

    chat_messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for e in events[-8:]:
        role = str(e.get("role", "USER")).lower()
        chat_messages.append(
            {
                "role": "assistant" if role == "assistant" else "user",
                "content": event_text(e),
            }
        )
    chat_messages.append({"role": "user", "content": req.message})

    yield _sse({"type": "status", "text": "Generating answer…"})
    answer = ""
    start = _timer()
    try:
        async for delta in _stream_llm(chat_messages):
            answer += delta
            yield _sse({"type": "delta", "text": delta})
    except Exception as exc:
        yield _sse({"type": "error", "message": f"OpenAI request failed: {exc}"})
        yield _sse({"type": "done"})
        return
    yield _sse(
        {
            "type": "event",
            "name": "openai.chat",
            "kind": "llm",
            "durationMs": _ms(start),
            "payload": {"model": settings.openai_chat_model, "context_messages": len(chat_messages)},
        }
    )

    start = _timer()
    try:
        await agent_memory.add_event(session_id=req.session_id, role="USER", text=req.message)
        await agent_memory.add_event(session_id=req.session_id, role="ASSISTANT", text=answer)
        yield _sse(
            {
                "type": "event",
                "name": "working_memory.append",
                "kind": "memory",
                "durationMs": _ms(start),
                "payload": {
                    "stored": 2,
                    "note": "Server summarizes the session and promotes durable facts in the background.",
                },
            }
        )
    except Exception as exc:
        yield _sse(
            {
                "type": "event",
                "name": "working_memory.append",
                "kind": "memory",
                "payload": {"error": str(exc)},
            }
        )
    yield _sse({"type": "done"})


# ── routes ───────────────────────────────────────────────


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "openai_configured": settings.openai_configured(),
            "redis_configured": settings.redis_configured(),
            "agent_memory_configured": settings.agent_memory_configured(),
        }
    )


@app.get("/api/config")
async def config() -> JSONResponse:
    return JSONResponse(
        {
            "app_name": "Redis Motors",
            "subtitle": "A shopping assistant that remembers",
            "hero_title": "Power multi-step agents with real-time context.",
            "placeholder_text": "Ask about hybrid SUVs, budgets, or what I remember about you…",
            "modes": list(MODES.values()),
            "starter_groups": STARTER_GROUPS,
            "user_id": settings.demo_user_id,
        }
    )


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    if req.mode == "primitive":
        generator = _primitive_turn(req)
    else:
        generator = _context_engine_turn(req)
    return StreamingResponse(generator, media_type="text/event-stream")


@app.get("/api/memory/dashboard")
async def memory_dashboard(mode: str = "context_engine", session_id: str | None = None) -> JSONResponse:
    if mode == "primitive":
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, lambda: primitive.dashboard(session_id or ""))
        return JSONResponse({"mode": mode, **data})
    data = await agent_memory.dashboard(session_id)
    return JSONResponse({"mode": mode, **data})


@app.post("/api/memory/reset")
async def memory_reset(req: ResetRequest) -> JSONResponse:
    """Wipe demo state so the walkthrough can be re-run cleanly."""
    result: dict[str, Any] = {"primitive_cleared": False, "long_term_deleted": 0, "errors": []}
    if primitive.is_configured():
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: primitive.clear(None))
            result["primitive_cleared"] = True
        except Exception as exc:
            result["errors"].append(f"primitive: {exc}")
    if agent_memory.is_configured():
        try:
            result["long_term_deleted"] = await agent_memory.delete_all_long_term()
        except Exception as exc:
            result["errors"].append(f"agent memory: {exc}")
    return JSONResponse(result)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")
