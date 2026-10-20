// local-rag-agent-lab の画面。外部ライブラリは使わない (社内ネットワークでも動くように)。
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));

// ── 最小の Markdown (見出し・表・箇条書き・コード・太字・引用番号) ─────────────
function md(src) {
  const lines = String(src || "").replace(/\r\n/g, "\n").split("\n");
  const out = [];
  let i = 0;
  const inline = (t) => esc(t)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\[(\d{1,2})\]/g, '<sup class="cite">[$1]</sup>');
  while (i < lines.length) {
    const ln = lines[i];
    if (/^```/.test(ln)) {
      const buf = [];
      i++;
      while (i < lines.length && !/^```/.test(lines[i])) buf.push(lines[i++]);
      i++;
      out.push("<pre><code>" + esc(buf.join("\n")) + "</code></pre>");
      continue;
    }
    if (/^\s*\|.*\|\s*$/.test(ln) && i + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {
      const row = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const head = row(ln);
      i += 2;
      const body = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) body.push(row(lines[i++]));
      out.push("<table><thead><tr>" + head.map((c) => "<th>" + inline(c) + "</th>").join("") + "</tr></thead><tbody>" +
        body.map((r) => "<tr>" + r.map((c) => "<td>" + inline(c) + "</td>").join("") + "</tr>").join("") + "</tbody></table>");
      continue;
    }
    const h = ln.match(/^(#{1,4})\s+(.*)/);
    if (h) { out.push(`<h${h[1].length + 2}>${inline(h[2])}</h${h[1].length + 2}>`); i++; continue; }
    if (/^\s*([-*・]|\d+[.)])\s+/.test(ln)) {
      const ordered = /^\s*\d+[.)]/.test(ln);
      const items = [];
      while (i < lines.length && /^\s*([-*・]|\d+[.)])\s+/.test(lines[i])) {
        items.push("<li>" + inline(lines[i].replace(/^\s*([-*・]|\d+[.)])\s+/, "")) + "</li>");
        i++;
      }
      out.push((ordered ? "<ol>" : "<ul>") + items.join("") + (ordered ? "</ol>" : "</ul>"));
      continue;
    }
    if (!ln.trim()) { i++; continue; }
    const para = [];
    while (i < lines.length && lines[i].trim() && !/^(```|#{1,4}\s|\s*\||\s*([-*・]|\d+[.)])\s+)/.test(lines[i])) para.push(lines[i++]);
    if (para.length) out.push("<p>" + para.map(inline).join("<br>") + "</p>");
    else { out.push("<p>" + inline(ln) + "</p>"); i++; }
  }
  return out.join("\n");
}

// ── NDJSON を読む ─────────────────────────────────────────────────────────
async function streamNDJSON(url, body, onEvent) {
  const res = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  if (!res.ok || !res.body) { onEvent({type: "error", message: `HTTP ${res.status}: ${await res.text()}`}); return; }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const {value, done} = await reader.read();
    if (done) break;
    buf += dec.decode(value, {stream: true});
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (line) { try { onEvent(JSON.parse(line)); } catch (e) { /* 壊れた行は捨てる */ } }
    }
  }
}

function addMsg(log, cls, html) {
  const el = document.createElement("div");
  el.className = "msg " + cls;
  el.innerHTML = html;
  log.appendChild(el);
  el.scrollIntoView({block: "end"});
  return el;
}

function sourcesHTML(hits) {
  if (!hits || !hits.length) return "";
  return '<details class="sources"><summary>参照した文書 (' + hits.length + ")</summary><ol>" +
    hits.map((h) => `<li><span class="path">[${h.n}] ${esc(h.path)}</span> #${h.chunk_index}
      <span class="hint">score ${Number(h.score).toFixed(3)}</span>
      <details><summary>本文</summary><div class="text">${esc(h.text)}</div></details></li>`).join("") +
    "</ol></details>";
}

function note(el, kind, text) {
  const n = document.createElement("div");
  n.className = "note " + kind;
  n.textContent = text;
  el.appendChild(n);
}

// ── RAG チャット ───────────────────────────────────────────────────────────
const chatHistory = [];
$("#chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("#chat-input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  const log = $("#chat-log");
  addMsg(log, "user", esc(text));
  const box = addMsg(log, "bot", '<div class="src"></div><details class="think" hidden><summary>思考</summary><div class="thinking"></div></details><div class="body">…</div><div class="notes"></div>');
  const btn = $("#chat-form button[type=submit]");
  btn.disabled = true;
  let answer = "", thinking = "";
  await streamNDJSON("/api/chat", {message: text, history: chatHistory, use_rag: $("#chat-rag").checked}, (ev) => {
    if (ev.type === "sources") $(".src", box).innerHTML = sourcesHTML(ev.hits);
    else if (ev.type === "thinking") { thinking += ev.text; $(".think", box).hidden = false; $(".thinking", box).textContent = thinking; }
    else if (ev.type === "content") { answer += ev.text; $(".body", box).innerHTML = md(answer); }
    else if (ev.type === "warn") note($(".notes", box), "warn", ev.message);
    else if (ev.type === "error") note($(".notes", box), "error", ev.message);
  });
  if (!answer) $(".body", box).innerHTML = "";
  chatHistory.push({role: "user", content: text}, {role: "assistant", content: answer});
  btn.disabled = false;
});
$("#chat-clear").addEventListener("click", () => { chatHistory.length = 0; $("#chat-log").innerHTML = ""; });

// ── エージェント ───────────────────────────────────────────────────────────
const agentHistory = [];
$("#agent-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("#agent-input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  const log = $("#agent-log");
  addMsg(log, "user", esc(text));
  const box = addMsg(log, "bot", '<details class="trace" open><summary>調査の過程</summary><div class="steps"></div></details><div class="charts"></div><div class="src"></div><details class="think" hidden><summary>思考</summary><div class="thinking"></div></details><div class="body"></div><div class="notes"></div>');
  const steps = $(".steps", box);
  const btn = $("#agent-form button[type=submit]");
  btn.disabled = true;
  let answer = "", thinking = "";
  await streamNDJSON("/api/agent", {message: text, history: agentHistory}, (ev) => {
    if (ev.type === "agent") {
      steps.insertAdjacentHTML("beforeend", `<div class="hint">道具: ${esc(ev.tools.join(", "))} / 最大 ${ev.max_steps} 手${ev.cause ? " / 原因調査の形式" : ""}</div>`);
    } else if (ev.type === "step") {
      steps.insertAdjacentHTML("beforeend", `<div class="step" id="s${ev.n}-${box.dataset.k = (box.dataset.k || Date.now())}">
        <b>${ev.n}.</b> <span class="tool">${esc(ev.tool)}</span> ${esc(ev.thought)}
        <span class="res">…</span>
        <details><summary>引数・結果</summary><pre>${esc(JSON.stringify(ev.args, null, 1))}</pre><pre class="detail"></pre></details></div>`);
    } else if (ev.type === "observation") {
      const st = $(`#s${ev.n}-${box.dataset.k}`, steps);
      if (st) {
        $(".res", st).innerHTML = `<span class="${ev.ok ? "ok" : "ng"}">${ev.ok ? "✓" : "✗"}</span> ${esc(ev.summary)} <span class="hint">${ev.elapsed_ms} ms</span>`;
        $(".detail", st).textContent = ev.detail;
      }
    } else if (ev.type === "chart") {
      $(".charts", box).insertAdjacentHTML("beforeend", `<figure class="chart"><img alt="${esc(ev.title)}" src="${ev.src}"><figcaption>${esc(ev.title)} (手順 ${ev.n})</figcaption></figure>`);
    } else if (ev.type === "sources") {
      $(".src", box).innerHTML = sourcesHTML(ev.hits);
    } else if (ev.type === "thinking") {
      thinking += ev.text; $(".think", box).hidden = false; $(".thinking", box).textContent = thinking;
    } else if (ev.type === "content") {
      answer += ev.text; $(".body", box).innerHTML = md(answer);
      $(".trace", box).open = false;
    } else if (ev.type === "warn") note($(".notes", box), "warn", ev.message);
    else if (ev.type === "error") note($(".notes", box), "error", ev.message);
  });
  agentHistory.push({role: "user", content: text}, {role: "assistant", content: answer});
  btn.disabled = false;
});
$("#agent-clear").addEventListener("click", () => { agentHistory.length = 0; $("#agent-log").innerHTML = ""; });

// Ctrl+Enter で送信
for (const id of ["chat", "agent"]) {
  $(`#${id}-input`).addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); $(`#${id}-form`).requestSubmit(); }
  });
}

// ── 検索 ───────────────────────────────────────────────────────────────────
$("#search-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("#search-input").value.trim();
  if (!q) return;
  const out = $("#search-results");
  out.innerHTML = '<p class="hint">検索中…</p>';
  const t0 = performance.now();
  const res = await fetch("/api/search", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({query: q, top_k: Number($("#search-k").value) || 10})});
  if (!res.ok) { out.innerHTML = `<p class="note error">${esc(await res.text())}</p>`; return; }
  const {hits} = await res.json();
  out.innerHTML = `<p class="hint">${hits.length} 件 (${Math.round(performance.now() - t0)} ms)</p>` +
    hits.map((h, i) => `<div class="hit"><div class="meta">[${i + 1}] ${esc(h.path)} #${h.chunk_index}/${h.num_chunks}
      — ${h.score_type} ${h.score.toFixed(3)} (dense ${Number(h.dense_score).toFixed(3)})</div><div class="text">${esc(h.text)}</div></div>`).join("");
});

// ── 評価 ───────────────────────────────────────────────────────────────────
async function loadEval() {
  const res = await fetch("/api/eval/runs");
  const {runs} = await res.json();
  const pct = (v) => v == null ? "-" : (v * 100).toFixed(1) + "%";
  $("#eval-table").innerHTML = !runs.length ? '<p class="hint">まだ記録がありません。</p>' :
    '<table class="evaltbl"><thead><tr><th>日時</th><th>label</th><th>問</th><th>FILE@1</th><th>FILE@k</th><th>CHUNK@1</th><th>CHUNK@5</th><th>CHUNK@k</th><th>MRR</th><th>miss</th><th>ms/問</th><th>チャンク数</th></tr></thead><tbody>' +
    runs.map((r) => `<tr><td>${esc(r.run_at)}</td><td>${esc(r.label)}</td><td>${r.n_questions}</td><td>${pct(r.file_recall_1)}</td>
      <td class="main">${pct(r.file_recall_k)}</td><td>${pct(r.recall_1)}</td><td>${pct(r.recall_5)}</td><td>${pct(r.recall_k)}</td>
      <td>${r.mrr == null ? "-" : r.mrr.toFixed(3)}</td><td>${r.miss}</td><td>${r.latency_ms == null ? "-" : Math.round(r.latency_ms)}</td><td>${r.chunks_total ?? "-"}</td></tr>`).join("") +
    "</tbody></table>";
}

// ── タブ・状態 ─────────────────────────────────────────────────────────────
for (const b of document.querySelectorAll(".tab")) {
  b.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
    document.querySelectorAll(".pane").forEach((p) => p.classList.toggle("active", p.id === "tab-" + b.dataset.tab));
    if (b.dataset.tab === "eval") loadEval();
  });
}

fetch("/api/health").then((r) => r.json()).then((h) => {
  const r = h.retrieval || {}, l = h.llm || {}, d = h.db || {};
  $("#status").textContent = [
    r.error ? "索引: エラー" : `索引 ${r.total_chunks?.toLocaleString()} チャンク${r.reranker ? " + リランク" : ""}`,
    l.error ? "LLM: 未接続" : `LLM ${l.default_model}`,
    `DB ${d.tables} 表`,
  ].join(" · ");
}).catch(() => { $("#status").textContent = "状態を取得できません"; });
