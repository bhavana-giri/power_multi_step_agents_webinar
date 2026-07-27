# Redis Agent Memory Demo — Primitive Memory vs Real-time Context Engine

Companion demo for the **"Power multi-step AI agents with real-time context"** deck.
Layout follows [redis/redis-iris-demos](https://github.com/redis/redis-iris-demos)'
"Simple RAG vs Real-time Context" comparison, but the axis here is **memory**:

| | Primitive Memory | Real-time Context Engine | Memory + Context Retriever |
|---|---|---|---|
| Backed by | RedisVL `MessageHistory` / `SemanticMessageHistory` | Redis Agent Memory (Redis Cloud) | Agent Memory + live data tools over Redis |
| Scope | One session at a time | Working + long-term, cross-session | Same, plus live operational data |
| State | Raw turns you manage | Rolling summary, bounded automatically | Rolling summary, bounded automatically |
| Durable facts | None | Background extraction + dedup | Background extraction + dedup |
| Retrieval | Recency window + in-session semantic recall | Hybrid: meaning + metadata + recency | Hybrid retrieval + tool calls (inventory, orders) |

The app is a car-shopping assistant ("Redis Motors") with a fixed inventory, so the
only variable in the demo is what the agent remembers.

## Setup

```bash
cd agent-memory-demo
./start.sh          # first run creates .env from .env.example, then exits
# fill in .env (see below), then:
./start.sh          # http://localhost:8060
```

Fill in `.env`:

| Variable | Where to get it |
|---|---|
| `OPENAI_API_KEY` (and optional `OPENAI_BASE_URL`) | OpenAI / your gateway |
| `REDIS_URL` | Redis Cloud database — used by the Primitive Memory mode |
| `MEMORY_API_BASE_URL`, `MEMORY_STORE_ID`, `MEMORY_API_KEY` | Redis Cloud console → Agent Memory — used by the Context Engine mode |

Either mode works alone; the other shows a friendly configuration error.

## Demo script (matches the deck's live-demo slides)

The landing page has prefilled query chips grouped per slide. Suggested flow —
run everything once in **Primitive Memory**, then repeat in **Real-time Context
Engine** (or alternate per act):

1. **Session state** (slide 07) — click the chips in order:
   *"Show me hybrid SUVs."* → *"Keep it under $35k."* → *"Which of those has the best
   mileage?"* Both modes follow the pronoun, because both hold session state. Open the
   **Redis Memory** panel: primitive shows raw turns only; the engine shows working
   memory with a rolling summary.
2. **Durable facts** (slide 08) — *"Remember this: I only want hybrids, and my budget
   is $35k max."* and *"I've decided — the RAV4 Hybrid is my top pick. Remember that."*
   In the panel, watch the engine's **Long-term memory** section fill as the server
   extracts facts in the background (hit Refresh after a few seconds). Primitive mode:
   the long-term section stays empty — that's where the primitives stop.
3. **Relevant history** (slide 09) — *"What do you remember about my car preferences?"*
   Open **Activity** to show the hybrid `long_term_memory.search` call (query + owner
   filter + returned memories) that got fused into the prompt.
4. **Session two** (Part 04 of the deck) — click **⟳ New session**, then *"I'm back!
   Based on what you know about me, which car should I buy today?"*
   Primitive Memory: blank slate, generic answer. Context Engine: recalls the budget,
   the hybrid preference, and the RAV4 decision.

5. **Beyond memory** (Part 05 of the deck) — the *Context Retriever + Memory* chips.
   First ask *"Is the Toyota RAV4 Hybrid in stock today?"* in **Real-time Context
   Engine** mode: the agent answers it can't check live data — memory recalls; it
   can't look things up (slide 30). Switch to **Memory + Context Retriever** mode and
   ask again: the agent calls `context_retriever.get_vehicle` (green *tool* events in
   the Activity panel) against live inventory stored in Redis and answers with current
   stock and discount. *"What's the status of my order 4471?"* matches the slide-30
   example. Finish with *"Given what you know about me, which car should I buy — and
   can I drive it home today?"* — long-term memories (top pick, budget) plus live
   stock fuse into one answer: three sources, one prompt (slide 34).

**Reset demo memory** (panel footer) wipes stored turns and long-term memories so the
walkthrough can be re-run cleanly.

## How it works

- The browser sends **only the newest user message** — each mode must reconstruct
  context from its own memory layer, so the difference is real, not cosmetic.
- `backend/primitive_memory.py` — RedisVL `SemanticMessageHistory` (one index,
  per-session tags; falls back to plain `MessageHistory` if the vectorizer is
  unavailable).
- `backend/agent_memory.py` — Agent Memory Cloud REST API (session-memory events,
  long-term search with `ownerId`/`namespace` filters), same endpoints as
  redis-iris-demos.
- `backend/live_data.py` — Context Retriever-style tools for the Iris mode:
  declared data models (inventory, orders) seeded into and queried from Redis,
  exposed to the agent as typed function-calling tools. Swap the handlers for
  the managed Context Retriever MCP tools to use the real service.
- `backend/main.py` — FastAPI, SSE streaming chat, memory dashboard, reset endpoint;
  serves the static frontend.
