/* Redis Agent Memory Demo — Primitive Memory (RedisVL) vs Real-time Context Engine (Agent Memory) */

const state = {
  config: null,
  mode: localStorage.getItem("amd-mode") || "context_engine",
  sessionNum: 1,
  sessionId: crypto.randomUUID(),
  messages: [], // {id, role, content, status, events: [], error}
  isLoading: false,
  panelOpen: false,
  panelTab: "memory",
  dashboard: null,
};

const $ = (id) => document.getElementById(id);
const shell = $("app");

// ── boot ────────────────────────────────────────────────

async function boot() {
  try {
    const res = await fetch("/api/config");
    state.config = await res.json();
  } catch {
    setTimeout(boot, 1500);
    return;
  }
  renderModeToggle();
  renderStarters();
}
boot();

// ── mode toggle + starters ──────────────────────────────

function renderModeToggle() {
  const wrap = $("mode-toggle");
  wrap.innerHTML = "";
  for (const m of state.config.modes) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className =
      "mode-toggle-option" +
      (m.id !== "primitive" ? " engine" : "") +
      (state.mode === m.id ? " active" : "");
    btn.innerHTML = `<span>${m.label}</span><small>${m.sublabel}</small>`;
    btn.onclick = () => setMode(m.id);
    wrap.appendChild(btn);
  }
  const current = state.config.modes.find((m) => m.id === state.mode);
  $("mode-description").textContent = current ? current.description : "";
  $("panel-btn-label").textContent =
    state.mode === "primitive" ? "RedisVL Memory" : "Redis Agent Memory";
}

function setMode(mode) {
  if (state.mode === mode) return;
  state.mode = mode;
  localStorage.setItem("amd-mode", mode);
  state.messages = [];
  state.sessionNum = 1;
  state.sessionId = crypto.randomUUID();
  state.dashboard = null;
  updateSessionBadge();
  renderModeToggle();
  renderConversation();
  if (state.panelOpen) loadDashboard();
}

function renderStarters() {
  const strip = $("starter-strip");
  strip.innerHTML = "";
  for (const group of state.config.starter_groups) {
    const g = document.createElement("div");
    g.className = "starter-group";
    const head = document.createElement("div");
    head.className = "starter-group-head";
    head.innerHTML =
      `<span class="starter-group-eyebrow">${group.eyebrow}</span>` +
      `<span class="starter-group-label">${group.label}</span>` +
      `<span class="starter-group-hint">${group.hint || ""}</span>`;
    g.appendChild(head);
    const chips = document.createElement("div");
    chips.className = "starter-chips";
    for (const chip of group.chips) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "starter-chip";
      b.textContent = chip.title;
      b.title = chip.prompt;
      b.onclick = () => {
        $("hero-input").value = chip.prompt;
        $("hero-input").focus();
      };
      chips.appendChild(b);
    }
    g.appendChild(chips);
    strip.appendChild(g);
  }
}

// ── session controls ────────────────────────────────────

function updateSessionBadge() {
  $("session-badge").textContent = `Session ${state.sessionNum}`;
}

$("new-session-btn").onclick = () => {
  state.sessionNum += 1;
  state.sessionId = crypto.randomUUID();
  state.messages = [];
  updateSessionBadge();
  renderConversation();
  toast(
    state.mode === "primitive"
      ? `Session ${state.sessionNum} started — RedisVL history is per-session, so the assistant starts blank.`
      : `Session ${state.sessionNum} started — same user, so long-term memories carry over.`
  );
  if (state.panelOpen) loadDashboard();
};

$("brand").onclick = () => {
  state.messages = [];
  renderConversation();
};

// ── chat ────────────────────────────────────────────────

$("hero-composer").onsubmit = (e) => {
  e.preventDefault();
  submit($("hero-input").value);
  $("hero-input").value = "";
};
$("footer-composer").onsubmit = (e) => {
  e.preventDefault();
  submit($("footer-input").value);
  $("footer-input").value = "";
};

async function submit(text) {
  const trimmed = (text || "").trim();
  if (!trimmed || state.isLoading) return;

  state.messages.push({ id: `u-${Date.now()}`, role: "user", content: trimmed, events: [] });
  const assistant = {
    id: `a-${Date.now()}`,
    role: "assistant",
    content: "",
    status: "Contacting agent…",
    events: [],
    error: null,
    userText: trimmed,
  };
  state.messages.push(assistant);
  state.isLoading = true;
  renderConversation();

  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: trimmed, mode: state.mode, session_id: state.sessionId }),
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() ?? "";
      for (const part of parts) {
        if (!part.startsWith("data: ")) continue;
        handleEvent(assistant, JSON.parse(part.slice(6)));
      }
    }
  } catch {
    assistant.error = "Connection error. Is the backend running?";
  }
  assistant.status = null;
  state.isLoading = false;
  renderConversation();
  if (state.panelOpen) {
    if (state.panelTab === "memory") loadDashboard();
    else renderPanel();
  }
}

function handleEvent(assistant, ev) {
  switch (ev.type) {
    case "status":
      assistant.status = ev.text;
      break;
    case "event":
      assistant.events.push(ev);
      break;
    case "delta":
      assistant.status = null;
      assistant.content += ev.text;
      break;
    case "error":
      assistant.error = ev.message;
      break;
  }
  renderConversation();
}

// ── conversation rendering ──────────────────────────────

function renderConversation() {
  const hasMessages = state.messages.length > 0;
  $("empty-state").hidden = hasMessages;
  $("conversation").hidden = !hasMessages;
  shell.classList.toggle("shell--landing", !hasMessages);

  const list = $("message-list");
  list.innerHTML = "";
  for (const msg of state.messages) {
    const div = document.createElement("div");
    div.className = `msg msg-${msg.role}` + (msg.error ? " msg-error" : "");

    if (msg.status) {
      const status = document.createElement("div");
      status.className = "msg-status";
      status.innerHTML = `<span class="spinner"></span><span>${msg.status}</span>`;
      div.appendChild(status);
    }

    const content = msg.error || msg.content;
    if (content) {
      const bubble = document.createElement("div");
      bubble.className = "msg-bubble";
      bubble.textContent = content;
      div.appendChild(bubble);
    }

    if (msg.role === "assistant" && msg.events.length > 0 && !msg.status) {
      const link = document.createElement("button");
      link.type = "button";
      link.className = "msg-activity-link";
      const memOps = msg.events.filter((e) => e.kind === "memory").length;
      link.textContent = `${msg.events.length} operations (${memOps} memory) — view activity`;
      link.onclick = () => openPanel("activity");
      div.appendChild(link);
    }
    list.appendChild(div);
  }
  list.scrollTop = list.scrollHeight;
}

// ── panel ───────────────────────────────────────────────

$("panel-btn").onclick = () => {
  if (state.panelOpen && state.panelTab === "memory") {
    closePanel();
    return;
  }
  openPanel("memory");
};
$("panel-close").onclick = closePanel;
$("refresh-memory-btn").onclick = () => loadDashboard();

document.querySelectorAll(".panel-tab").forEach((tab) => {
  tab.onclick = () => {
    state.panelTab = tab.dataset.tab;
    document.querySelectorAll(".panel-tab").forEach((t) => t.classList.toggle("active", t === tab));
    if (state.panelTab === "memory") loadDashboard();
    else renderPanel();
  };
});

function openPanel(tab) {
  state.panelOpen = true;
  state.panelTab = tab;
  shell.classList.add("panel-open");
  $("panel-btn").classList.add("active");
  document.querySelectorAll(".panel-tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.tab === tab)
  );
  if (tab === "memory") loadDashboard();
  else renderPanel();
}

function closePanel() {
  state.panelOpen = false;
  shell.classList.remove("panel-open");
  $("panel-btn").classList.remove("active");
}

async function loadDashboard() {
  const body = $("panel-body");
  body.innerHTML = `<div class="panel-empty">Loading memory…</div>`;
  try {
    const res = await fetch(
      `/api/memory/dashboard?mode=${state.mode}&session_id=${encodeURIComponent(state.sessionId)}`
    );
    state.dashboard = await res.json();
  } catch {
    state.dashboard = { enabled: false, errors: ["Unable to load memory dashboard."] };
  }
  renderPanel();
}

function renderPanel() {
  const body = $("panel-body");
  body.innerHTML = "";
  if (state.panelTab === "activity") return renderActivity(body);
  renderMemoryDashboard(body);
}

function section(title, count) {
  const s = document.createElement("div");
  s.className = "panel-section";
  const h = document.createElement("p");
  h.className = "panel-section-title";
  h.innerHTML = `${title}${count !== undefined ? ` <span class="count">${count}</span>` : ""}`;
  s.appendChild(h);
  return s;
}

function empty(text) {
  const d = document.createElement("div");
  d.className = "panel-empty";
  d.textContent = text;
  return d;
}

function renderMemoryDashboard(body) {
  const d = state.dashboard;
  if (!d) {
    body.appendChild(empty("No memory data loaded yet."));
    return;
  }
  for (const err of d.errors || []) {
    const note = document.createElement("div");
    note.className = "panel-note";
    note.textContent = err;
    body.appendChild(note);
  }

  if (state.mode === "primitive") {
    const s1 = section(`Session transcript · RedisVL MessageHistory`, (d.messages || []).length);
    if ((d.messages || []).length === 0) {
      s1.appendChild(empty("No stored turns for this session."));
    } else {
      for (const m of d.messages) {
        const line = document.createElement("div");
        line.className = "turn-line";
        line.innerHTML = `<span class="turn-role">${escapeHtml(m.role || "?")}</span><span>${escapeHtml(
          m.content || ""
        )}</span>`;
        s1.appendChild(line);
      }
    }
    body.appendChild(s1);

    const s2 = section("Working summary");
    s2.appendChild(empty("None — primitives don't summarize."));
    body.appendChild(s2);

    const s3 = section("Long-term memory");
    s3.appendChild(
      empty("None — raw turns only, scoped to one session. This is where the primitives stop.")
    );
    body.appendChild(s3);
    return;
  }

  // context engine dashboard
  const working = d.working || {};
  const events = working.events || [];
  const summary = working.summary;

  const s1 = section("Working memory · this session", events.length);
  if (summary) {
    const sum = document.createElement("div");
    sum.className = "mem-summary";
    sum.textContent = summary;
    s1.appendChild(sum);
  }
  if (events.length === 0) {
    s1.appendChild(empty("No events yet in this session."));
  } else {
    for (const e of events.slice(-12)) {
      const text = (e.content || []).map((c) => c.text).join(" ");
      const line = document.createElement("div");
      line.className = "turn-line";
      line.innerHTML = `<span class="turn-role">${escapeHtml(e.role || "?")}</span><span>${escapeHtml(
        text
      )}</span>`;
      s1.appendChild(line);
    }
  }
  body.appendChild(s1);

  const longTerm = d.long_term || [];
  const s2 = section("Long-term memory · all sessions", longTerm.length);
  if (longTerm.length === 0) {
    s2.appendChild(
      empty("No durable facts yet. Confirm a preference and the server will extract it in the background.")
    );
  } else {
    for (const m of longTerm) {
      const card = document.createElement("div");
      card.className = "mem-card";
      const meta = [];
      if (m.memoryType) meta.push(m.memoryType);
      for (const t of m.topics || []) meta.push(t);
      if (m.createdAt) meta.push(new Date(m.createdAt).toLocaleString());
      card.innerHTML =
        `<div class="mem-card-text">${escapeHtml(m.text || "")}</div>` +
        `<div class="mem-card-meta">${meta.map((x) => `<span class="mem-tag">${escapeHtml(String(x))}</span>`).join("")}</div>`;
      s2.appendChild(card);
    }
  }
  body.appendChild(s2);
}

function renderActivity(body) {
  const turns = state.messages.filter((m) => m.role === "assistant" && m.events.length > 0);
  if (turns.length === 0) {
    body.appendChild(empty("No activity yet — send a message first."));
    return;
  }
  for (const turn of turns) {
    const wrap = document.createElement("div");
    wrap.className = "activity-turn";
    const title = document.createElement("div");
    title.className = "activity-turn-title";
    title.textContent = `“${turn.userText}”`;
    wrap.appendChild(title);
    for (const ev of turn.events) {
      const row = document.createElement("div");
      row.className = "event-row";
      const head = document.createElement("button");
      head.type = "button";
      head.className = "event-head";
      head.innerHTML =
        `<span class="event-kind ${ev.kind}">${ev.kind}</span>` +
        `<span class="event-name">${escapeHtml(ev.name)}</span>` +
        (ev.durationMs !== undefined ? `<span class="event-ms">${ev.durationMs} ms</span>` : "");
      head.onclick = () => row.classList.toggle("open");
      row.appendChild(head);
      const pre = document.createElement("pre");
      pre.className = "event-payload";
      pre.textContent = JSON.stringify(ev.payload ?? {}, null, 2);
      row.appendChild(pre);
      wrap.appendChild(row);
    }
    body.appendChild(wrap);
  }
}

// ── reset ───────────────────────────────────────────────

$("reset-btn").onclick = async () => {
  $("reset-note").textContent = "Resetting…";
  try {
    const res = await fetch("/api/memory/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId }),
    });
    const data = await res.json();
    const bits = [];
    if (data.primitive_cleared) bits.push("RedisVL history cleared");
    if (data.agent_memory_cleared)
      bits.push(`Agent Memory: ${data.long_term_deleted} long-term memories deleted`);
    $("reset-note").textContent = bits.join(" · ") || "Nothing to reset.";
    if ((data.errors || []).length) $("reset-note").textContent += ` (${data.errors.join("; ")})`;
  } catch {
    $("reset-note").textContent = "Reset failed.";
  }
  loadDashboard();
};

// ── misc ────────────────────────────────────────────────

let toastTimer = null;
function toast(text) {
  let el = document.querySelector(".toast");
  if (!el) {
    el = document.createElement("div");
    el.className = "toast";
    document.body.appendChild(el);
  }
  el.textContent = text;
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 3200);
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value;
  return div.innerHTML;
}
