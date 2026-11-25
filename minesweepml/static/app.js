import { BarChart, LineChart } from "/static/charts.js";
import { COVERED, EXPLODED, MineBoard, MiniBoards, parseCells } from "/static/boards.js";

const $id = (id) => document.getElementById(id);
const DIFFS = ["beginner", "intermediate", "expert"];
const DIFF_LABEL = { beginner: "Beginner", intermediate: "Intermediate", expert: "Expert" };
const BASE_RATE = 5; // moves per second at 1x
const ACTIVE = ["starting", "running", "paused", "stopping"];

const state = {
  info: null, status: { state: "idle" }, bots: [], checkpoints: [], selectedBot: null, chartBot: null,
  job: null, benchResults: [], benchShown: null, sim: null,
};

// ---------------------------------------------------------------- helpers

function fmtDuration(s) {
  if (s === null || s === undefined) return "–";
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}
const fmtInt = (n) => (n === null || n === undefined ? "–" : Number(n).toLocaleString());
const fmtNum = (n, d = 2) => (n === null || n === undefined ? "–" : Number(n).toFixed(d));
const fmtPct = (v, d = 1) => (v === null || v === undefined ? "–" : `${(100 * v).toFixed(d)}%`);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const fmtDate = (t) => new Date(t * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });

async function api(path, body, method) {
  const opts = body === undefined && !method ? {} : {
    method: method || "POST", headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  };
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { msg = (await res.json()).detail || msg; } catch { /* not JSON */ }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return res.headers.get("content-type")?.includes("application/json") ? res.json() : res;
}

function log(message, ts = Date.now()) {
  const list = $id("log");
  list.querySelector(".placeholder")?.remove();
  const li = document.createElement("li");
  const time = new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  li.innerHTML = `<time>${time}</time>${esc(message)}`;
  list.prepend(li);
  while (list.children.length > 200) list.lastChild.remove();
}

function setProgress(el, done, total) {
  el.hidden = false;
  el.querySelector(".progress-fill").style.width = `${total ? (100 * done) / total : 0}%`;
}

function winrates(result) {
  if (!result) return '<span class="muted">–</span>';
  const d = result.difficulties || {};
  return `<span class="winrates">${DIFFS.map((k) => (d[k] ? `<b>${fmtPct(d[k].win_rate, 0)}</b>` : "<span>–</span>")).join(" · ")}</span>`;
}

// ---------------------------------------------------------------- tabs

function showTab(name) {
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${name}`));
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
  if (name === "benchmark") refreshBenchmarks();
  if (name === "simulate") replay.board.schedule();
  if (name === "train") liveBoard.schedule();
}
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
window.addEventListener("hashchange", () => {
  const tab = location.hash.slice(1);
  if (["bots", "train", "simulate", "benchmark"].includes(tab)) showTab(tab);
});

// ---------------------------------------------------------------- bots

async function refreshBots() {
  [state.bots, state.checkpoints] = await Promise.all([api("/api/bots"), api("/api/checkpoints")]);
  renderBots();
  renderBotSelects();
  renderPlayerSelects();
}

function renderBots() {
  const tbody = $id("botTable").querySelector("tbody");
  $id("botCount").textContent = state.bots.length ? `(${state.bots.length})` : "";
  $id("botsEmpty").hidden = state.bots.length > 0;
  tbody.innerHTML = state.bots.map((b) => `
    <tr data-name="${esc(b.name)}" class="${b.name === state.selectedBot ? "selected" : ""}">
      <td>${esc(b.name)}</td>
      <td class="hide-narrow">${b.net.blocks}×${b.net.channels}</td>
      <td class="num">${fmtInt(b.games)}</td>
      <td class="num hide-narrow">${fmtDuration(b.elapsed)}</td>
      <td><span class="badge ${b.training ? "running" : ""}">${b.training ? "training" : esc(b.status)}</span></td>
      <td>${winrates(b.benchmark)}</td>
      <td><div class="bot-actions">
        <button type="button" class="small" data-act="train">Train</button>
        <button type="button" class="small" data-act="simulate" ${b.checkpoints ? "" : "disabled"}>Simulate</button>
      </div></td>
    </tr>`).join("");
  if (state.selectedBot && !state.bots.some((b) => b.name === state.selectedBot)) state.selectedBot = null;
  renderBotDetail();
}

$id("botTable").addEventListener("click", (e) => {
  const tr = e.target.closest("tr[data-name]");
  if (!tr) return;
  const name = tr.dataset.name;
  const act = e.target.closest("button")?.dataset.act;
  if (act === "train") return trainBot(name);
  if (act === "simulate") return simulateWith(latestCheckpoint(name));
  state.selectedBot = name;
  renderBots();
});

function botByName(name) { return state.bots.find((b) => b.name === name); }
function latestCheckpoint(name) { return botByName(name)?.checkpoint_list?.[0]?.label || null; }

const SETTING_LABELS = [
  ["mix", "Difficulty mix", (s) => DIFFS.map((d) => `${DIFF_LABEL[d][0]} ${Math.round(100 * s[`mix_${d}`] / mixTotal(s))}%`).join(" · ")],
  ["eps", "Exploration", (s) => `${fmtNum(100 * s.eps_start, 1)}% → ${fmtNum(100 * s.eps_end, 1)}% over ${fmtInt(s.eps_games)} games`],
  ["record_prob", "Positions stored", (s) => `${Math.round(100 * s.record_prob)}%`],
  ["forgive_share", "Continue after a mine", (s) => `${Math.round(100 * s.forgive_share)}% of games`],
  ["lr", "Learning rate", (s) => String(s.lr)],
  ["safe_neighbours", "First click", (s) => (s.safe_neighbours ? "opens an area" : "safe cell only")],
];
const mixTotal = (s) => (s.mix_beginner + s.mix_intermediate + s.mix_expert) || 1;

function renderBotDetail() {
  const b = botByName(state.selectedBot);
  $id("botDetail").hidden = !b;
  if (!b) return;
  $id("bdName").innerHTML = `${esc(b.name)} <span class="badge ${b.training ? "running" : ""}">${b.training ? "training" : esc(b.status)}</span>`;
  $id("bdStats").innerHTML = [
    ["Network", `${b.net.blocks} blocks × ${b.net.channels} channels`],
    ["Games", fmtInt(b.games)], ["Training steps", fmtInt(b.steps)], ["Training time", fmtDuration(b.elapsed)],
    ["Last preset", b.preset || "–"], ["Started from", b.origin ? b.origin.split("/").slice(-3).join("/") : "an untrained network"],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("");
  const s = b.settings;
  $id("bdSettings").innerHTML = SETTING_LABELS.map(([, label, f]) => `<dt>${label}</dt><dd>${esc(f(s))}</dd>`).join("");
  $id("bdDelete").disabled = b.training;
  const ckpts = b.checkpoint_list || [];
  $id("bdCkptCount").textContent = `(${ckpts.length})`;
  $id("bdCkpts").innerHTML = ckpts.length ? ckpts.map((c) => `
    <li data-label="${esc(c.label)}">
      <span class="ckpt-name">${esc(c.label.split("/").pop())}</span>
      <span class="ckpt-meta">${c.benchmark ? `benchmark ${winrates(c.benchmark)} on ${fmtInt(c.benchmark.boards)} boards` : "not benchmarked"}</span>
      <span class="ckpt-actions">
        <button type="button" data-act="simulate">Simulate</button>
        <button type="button" data-act="benchmark">Benchmark</button>
        <button type="button" data-act="fork">Fork</button>
      </span>
    </li>`).join("") : '<li class="muted">No checkpoints yet: train the bot first.</li>';
}

$id("bdCkpts").addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  const li = e.target.closest("li[data-label]");
  if (!btn || !li) return;
  const label = li.dataset.label;
  if (btn.dataset.act === "simulate") simulateWith(label);
  else if (btn.dataset.act === "benchmark") benchmarkWith(label);
  else if (btn.dataset.act === "fork") openBotDialog(null, label);
});
$id("bdTrain").addEventListener("click", () => trainBot(state.selectedBot));
$id("bdEdit").addEventListener("click", () => openBotDialog(state.selectedBot));
$id("bdDelete").addEventListener("click", () => {
  const b = botByName(state.selectedBot);
  $id("deleteText").innerHTML = `This permanently removes <b>${esc(b.name)}</b> with its ${fmtInt(b.checkpoints)} checkpoint(s),
    metrics and replay buffer (the folder <code>runs/${esc(b.name)}/</code>). It cannot be undone.`;
  $id("deleteError").textContent = "";
  $id("deleteDialog").showModal();
});
$id("deleteConfirm").addEventListener("click", async () => {
  try {
    await api(`/api/bots/${encodeURIComponent(state.selectedBot)}`, undefined, "DELETE");
    $id("deleteDialog").close();
    state.selectedBot = null;
    await refreshBots();
  } catch (err) { $id("deleteError").textContent = err.message; }
});

function trainBot(name) {
  $id("trainBot").value = name;
  syncTrainBot();
  state.chartBot = name;
  $id("chartBot").value = name;
  loadHistory(name);
  showTab("train");
}

function simulateWith(label) {
  if (!label) return;
  $id("simPlayer").value = label;
  showTab("simulate");
}

function benchmarkWith(label) {
  $id("benchPlayer").value = label;
  showTab("benchmark");
}

// -- bot dialog (create / edit)

let dialogEdit = null;

function openBotDialog(editName = null, fromLabel = "") {
  dialogEdit = editName;
  const b = editName ? botByName(editName) : null;
  const s = b ? b.settings : state.info.defaults;
  $id("botDialogTitle").textContent = b ? `Edit ${b.name}` : "New bot";
  $id("bfSubmit").textContent = b ? "Save" : "Create bot";
  $id("bfCreateOnly").hidden = !!b;
  $id("bfName").value = b ? b.name : "";
  const from = $id("bfFrom");
  from.innerHTML = '<option value="">An untrained network</option>' +
    state.checkpoints.map((c) => `<option value="${esc(c.label)}">${esc(c.label)}</option>`).join("");
  from.value = fromLabel;
  $id("bfSize").value = `${s.blocks}x${s.channels}`;
  if (!$id("bfSize").value) $id("bfSize").value = "6x32";
  const total = mixTotal(s);
  $id("bfMixB").value = Math.round((100 * s.mix_beginner) / total);
  $id("bfMixI").value = Math.round((100 * s.mix_intermediate) / total);
  $id("bfMixE").value = Math.round((100 * s.mix_expert) / total);
  $id("bfEpsStart").value = +(100 * s.eps_start).toFixed(2);
  $id("bfEpsEnd").value = +(100 * s.eps_end).toFixed(2);
  $id("bfEpsGames").value = s.eps_games;
  $id("bfRecord").value = Math.round(100 * s.record_prob);
  $id("bfForgive").value = Math.round(100 * s.forgive_share);
  $id("bfLr").value = s.lr;
  $id("bfOpen").checked = s.safe_neighbours;
  $id("botError").textContent = "";
  syncBotDialog();
  $id("botDialog").showModal();
}

function syncBotDialog() {
  const sum = ["bfMixB", "bfMixI", "bfMixE"].reduce((a, id) => a + (Number($id(id).value) || 0), 0);
  $id("bfMixSum").textContent = `(shares are normalised; now ${sum}%)`;
  $id("bfSize").disabled = !!$id("bfFrom").value;
}
["bfMixB", "bfMixI", "bfMixE"].forEach((id) => $id(id).addEventListener("input", syncBotDialog));
$id("bfFrom").addEventListener("change", syncBotDialog);
$id("btnNewBot").addEventListener("click", () => openBotDialog());
$id("bfCancel").addEventListener("click", () => $id("botDialog").close());

$id("botForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const num = (id, scale = 1) => Number($id(id).value) / scale;
  const settings = {
    mix_beginner: num("bfMixB", 100), mix_intermediate: num("bfMixI", 100), mix_expert: num("bfMixE", 100),
    eps_start: num("bfEpsStart", 100), eps_end: num("bfEpsEnd", 100), eps_games: Math.round(num("bfEpsGames")),
    record_prob: num("bfRecord", 100), forgive_share: num("bfForgive", 100), lr: num("bfLr"),
    safe_neighbours: $id("bfOpen").checked,
  };
  try {
    let name;
    if (dialogEdit) {
      const res = await api(`/api/bots/${encodeURIComponent(dialogEdit)}`, { name: $id("bfName").value.trim(), settings }, "PATCH");
      name = res.name;
    } else {
      const checkpoint = $id("bfFrom").value || null;
      if (!checkpoint) {
        const [blocks, channels] = $id("bfSize").value.split("x").map(Number);
        Object.assign(settings, { blocks, channels });
      }
      const res = await api("/api/bots", { name: $id("bfName").value.trim(), checkpoint, settings });
      name = res.name;
    }
    $id("botDialog").close();
    state.selectedBot = name;
    await refreshBots();
    showTab("bots");
  } catch (err) { $id("botError").textContent = err.message; }
});

// ---------------------------------------------------------------- train tab

function renderBotSelects() {
  const sel = $id("trainBot");
  const keep = sel.value;
  sel.innerHTML = '<option value="">+ New bot</option>' +
    state.bots.map((b) => `<option value="${esc(b.name)}">${esc(b.name)} (${fmtInt(b.games)} games)</option>`).join("");
  if (keep === "" && state.bots.length && !sel.dataset.touched) sel.value = state.bots[0].name;
  else sel.value = state.bots.some((b) => b.name === keep) ? keep : "";
  syncTrainBot();

  const chart = $id("chartBot");
  chart.innerHTML = state.bots.map((b) => `<option value="${esc(b.name)}">${esc(b.name)}</option>`).join("");
  if (!state.chartBot || !state.bots.some((b) => b.name === state.chartBot)) {
    state.chartBot = state.status.bot && ACTIVE.includes(state.status.state) ? state.status.bot : state.bots[0]?.name || null;
    if (state.chartBot) loadHistory(state.chartBot);
  }
  if (state.chartBot) chart.value = state.chartBot;
}

function syncTrainBot() {
  const name = $id("trainBot").value;
  $id("newBotNameRow").hidden = !!name;
  const b = botByName(name);
  $id("trainBotInfo").textContent = b
    ? `${b.net.blocks}×${b.net.channels} network · ${fmtInt(b.games)} games · ${SETTING_LABELS[0][2](b.settings)}. Start continues from its latest checkpoint.`
    : "A new bot with the default settings (change them on the Bots tab first if you like).";
}
$id("trainBot").addEventListener("change", (e) => { e.target.dataset.touched = "1"; syncTrainBot(); });

function updatePresetHint() {
  const preset = $id("preset").value;
  const v = state.info.presets[preset];
  $id("presetHint").textContent = `${v.workers} workers · ${v.threads} torch threads · batch ${v.batch_size} · ` +
    `nice ${v.nice} · buffer ${fmtInt(v.buffer_size)} (this machine has ${state.info.cpu_count} CPUs)`;
  for (const k of ["workers", "threads", "batch_size", "nice", "buffer_size"]) $id(k).placeholder = v[k];
}
$id("preset").addEventListener("change", updatePresetHint);

$id("trainForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  $id("trainError").textContent = "";
  const intOrNull = (id) => ($id(id).value === "" ? null : Number($id(id).value));
  const duration = document.querySelector("input[name=duration]:checked").value;
  const existing = $id("trainBot").value;
  const body = {
    bot: existing || $id("trainNewName").value.trim() || null, preset: $id("preset").value,
    workers: intOrNull("workers"), threads: intOrNull("threads"), batch_size: intOrNull("batch_size"),
    parallel_games: intOrNull("parallel_games"), nice: intOrNull("nice"), buffer_size: intOrNull("buffer_size"),
    eval_minutes: intOrNull("eval_minutes"), eval_boards: intOrNull("eval_boards"),
    duration, games: Number($id("games").value), time: $id("time").value, reset_lr: $id("resetLr").checked,
  };
  try {
    await api("/api/training/start", body);
    $id("resetLr").checked = false;
  } catch (err) { $id("trainError").textContent = err.message; }
});
$id("btnPause").addEventListener("click", () => api("/api/training/pause", {}));
$id("btnResume").addEventListener("click", () => api("/api/training/resume", {}));
$id("btnStop").addEventListener("click", () => api("/api/training/stop", {}));

function renderStatus(s) {
  const prev = state.status;
  state.status = s;
  const st = s.state || "idle";
  const badge = $id("stateBadge");
  badge.textContent = s.throttled && st === "running" ? "running · catching up" : st;
  badge.className = `badge ${st}`;
  const active = ACTIVE.includes(st);
  $id("btnStart").disabled = active;
  $id("btnPause").disabled = st !== "running";
  $id("btnResume").disabled = st !== "paused";
  $id("btnStop").disabled = !["running", "paused"].includes(st);
  const set = (id, v) => { $id(id).textContent = v ?? "–"; };
  set("st-bot", s.bot);
  set("st-preset", s.preset ? `${s.preset} (${s.workers} workers, ${s.threads} threads, batch ${s.batch_size})` : null);
  set("st-net", s.net);
  set("st-mix", s.mix);
  set("st-duration", s.duration);
  set("st-elapsed", s.elapsed !== undefined ? fmtDuration(s.elapsed) : null);
  set("st-eta", s.eta !== undefined && s.eta !== null ? fmtDuration(s.eta) : null);
  set("st-games", s.games !== undefined ? fmtInt(s.games) : null);
  set("st-gps", s.games_per_sec !== undefined ? fmtNum(s.games_per_sec, 2) : null);
  set("st-mps", s.moves_per_sec !== undefined ? fmtInt(Math.round(s.moves_per_sec)) : null);
  set("st-positions", s.positions !== undefined ? fmtInt(s.positions) : null);
  set("st-steps", s.steps !== undefined ? fmtInt(s.steps) : null);
  set("st-lr", s.lr !== undefined ? Number(s.lr).toExponential(1) : null);
  set("st-eps", s.eps !== undefined ? fmtPct(s.eps, 1) : null);
  set("st-buffer", s.buffer !== undefined ? fmtInt(s.buffer) : null);
  set("st-ckpt", s.checkpoint);
  set("st-eval", s.last_eval ? DIFFS.filter((d) => s.last_eval.difficulties[d])
    .map((d) => `${DIFF_LABEL[d][0]} ${fmtPct(s.last_eval.difficulties[d].win_rate, 0)}`).join(" · ") : null);
  if (s.bot && prev.bot !== s.bot && active) {
    refreshBots();
    if (state.chartBot !== s.bot) { state.chartBot = s.bot; $id("chartBot").value = s.bot; loadHistory(s.bot); }
  }
  if (ACTIVE.includes(prev.state) && ["completed", "stopped", "error"].includes(st)) refreshBots();
}

// -- live view

const liveBoard = new MineBoard($id("liveBoard"), { maxCell: 30, maxHeight: 520 });
const live = { focus: [0, 0] };
const liveGrid = new MiniBoards($id("liveGrid"), { onSelect: focusGame });

async function focusGame(worker, slot) {
  live.focus = [worker, slot];
  try { await api("/api/training/focus", { worker, slot }); } catch (err) { log(err.message); }
}

function renderLive(ev) {
  if (!ev) return;
  const cells = parseCells(ev.cells);
  liveBoard.set({ rows: ev.rows, cols: ev.cols, cells, probs: ev.probs, next: ev.next });
  $id("liveInfo").textContent = `worker ${ev.worker}, game ${ev.slot} · ${DIFF_LABEL[ev.difficulty] || ev.difficulty} · move ${ev.moves + 1}`;
}

function renderGrid(ev) {
  if (!ev) return;
  if (ev.focus) live.focus = ev.focus;
  liveGrid.render(ev.games, `${live.focus[0]}/${live.focus[1]}`);
  $id("gridInfo").textContent = ev.games.length ? `(${ev.games.length} shown)` : "";
}

// -- charts

const minutes = (v) => (v < 10 ? v.toFixed(1) : v.toFixed(0));
const pct = (v) => `${v.toFixed(0)}%`;
const charts = {
  loss: new LineChart($id("chLoss"), { series: ["loss"], xLabel: "min", xFmt: minutes }),
  win: new LineChart($id("chWin"), { series: DIFFS.map((d) => DIFF_LABEL[d]), xLabel: "game", integerX: true, yMin: 0, yMax: 100, yFmt: pct }),
  evals: new LineChart($id("chEval"), { series: DIFFS.map((d) => DIFF_LABEL[d]), xLabel: "games", integerX: true, yMin: 0, yMax: 100, yFmt: pct, markers: true }),
  cleared: new LineChart($id("chCleared"), { series: DIFFS.map((d) => DIFF_LABEL[d]), xLabel: "game", integerX: true, yMin: 0, yMax: 100, yFmt: pct }),
  gps: new LineChart($id("chGps"), { series: ["games / sec"], xLabel: "min", xFmt: minutes, yMin: 0 }),
  mps: new LineChart($id("chMps"), { series: ["moves / sec"], xLabel: "min", xFmt: minutes, yMin: 0 }),
  cpu: new LineChart($id("chCpu"), { series: ["CPU %"], xLabel: "min", xFmt: minutes, yMin: 0, yMax: 100, yFmt: pct }),
  cores: new BarChart($id("chCores"), { yMax: 100, yFmt: pct, labelFmt: (i) => `core ${i}` }),
};

async function loadHistory(name) {
  if (!name) return;
  let h;
  try { h = await api(`/api/bots/${encodeURIComponent(name)}/history`); } catch { return; }
  if (state.chartBot !== name) return;
  charts.loss.set(0, h.train.map((r) => [r.elapsed / 60, r.loss]));
  DIFFS.forEach((d, i) => {
    const rows = h.games.filter((r) => r.difficulty === d);
    charts.win.set(i, rows.map((r) => [r.game, 100 * r.win_rate]));
    charts.cleared.set(i, rows.map((r) => [r.game, 100 * r.cleared]));
    charts.evals.set(i, h.evals.filter((e) => e.difficulty === d).map((e) => [e.games, (100 * e.wins) / e.boards]));
  });
  charts.gps.set(0, h.system.map((r) => [r.elapsed / 60, r.games_per_sec]));
  charts.mps.set(0, h.system.map((r) => [r.elapsed / 60, r.moves_per_sec]));
  charts.cpu.set(0, h.system.map((r) => [r.elapsed / 60, r.cpu_total]));
  const last = h.system[h.system.length - 1];
  charts.cores.set(last ? last.cpu_per_core : []);
  $id("evalTable").innerHTML = h.evals.slice().reverse().map((e) => `<tr><td class="num">${fmtInt(e.games)}</td>
    <td>${esc(e.checkpoint)}</td><td>${DIFF_LABEL[e.difficulty] || esc(e.difficulty)}</td><td class="num">${e.boards}</td>
    <td class="num">${fmtPct(e.wins / e.boards)}</td><td class="num">${fmtPct(e.cleared)}</td><td class="num">${fmtNum(e.moves, 1)}</td></tr>`).join("")
    || '<tr><td colspan="7" class="muted">No benchmark during training yet.</td></tr>';
}
$id("chartBot").addEventListener("change", (e) => { state.chartBot = e.target.value; loadHistory(state.chartBot); });
setInterval(() => {  // the win-rate curves come from the database; refresh them while the bot trains
  if (state.chartBot && state.status.bot === state.chartBot && ACTIVE.includes(state.status.state)) loadHistory(state.chartBot);
}, 15000);

function onSystem(ev) {
  if (state.chartBot !== state.status.bot) return;
  charts.gps.push(0, [ev.elapsed / 60, ev.games_per_sec]);
  charts.mps.push(0, [ev.elapsed / 60, ev.moves_per_sec]);
  charts.cpu.push(0, [ev.elapsed / 60, ev.cpu_total]);
  charts.cores.set(ev.cpu_per_core);
}

// ---------------------------------------------------------------- players (simulate + benchmark)

function renderPlayerSelects() {
  for (const id of ["simPlayer", "benchPlayer"]) {
    const sel = $id(id);
    const keep = sel.value;
    const groups = {};
    for (const c of state.checkpoints) (groups[c.bot || "bundled"] ||= []).push(c);
    sel.innerHTML = '<optgroup label="Baselines"><option value="solver">Rule-based solver</option><option value="random">Random clicks</option></optgroup>' +
      Object.entries(groups).map(([bot, items]) => `<optgroup label="${esc(bot)}">${items.map((c) =>
        `<option value="${esc(c.label)}">${esc(c.label)}</option>`).join("")}</optgroup>`).join("");
    if (keep && [...sel.options].some((o) => o.value === keep)) sel.value = keep;
    else if (state.checkpoints.length) sel.value = state.checkpoints[0].label;
  }
}

function difficultyValue() {
  const d = $id("simDifficulty").value;
  if (d !== "custom") return d;
  return `${$id("simRows").value}x${$id("simCols").value}/${$id("simMines").value}`;
}
$id("simDifficulty").addEventListener("change", () => { $id("simCustom").hidden = $id("simDifficulty").value !== "custom"; });

// ---------------------------------------------------------------- simulate tab

$id("btnPlayOne").addEventListener("click", async () => {
  $id("simError").textContent = "";
  const seed = $id("simSeed").value === "" ? null : Number($id("simSeed").value);
  $id("btnPlayOne").disabled = true;
  try {
    const res = await api("/api/simulate/game", {
      player: $id("simPlayer").value, difficulty: difficultyValue(), seed, safe_neighbours: $id("simOpen").checked,
    });
    loadReplay(res.record, res.replay, true);
  } catch (err) { $id("simError").textContent = err.message; } finally { $id("btnPlayOne").disabled = false; }
});

$id("btnBatch").addEventListener("click", async () => {
  $id("simError").textContent = "";
  try {
    await api("/api/simulate/batch", {
      player: $id("simPlayer").value, difficulty: difficultyValue(), games: Number($id("simGames").value),
      seed: $id("simSeed").value === "" ? 0 : Number($id("simSeed").value), safe_neighbours: $id("simOpen").checked,
    });
  } catch (err) { $id("simError").textContent = err.message; }
});
$id("btnSimCancel").addEventListener("click", () => api("/api/jobs/cancel", {}));

async function showSimulation() {
  const sim = await api("/api/simulate/last");
  if (!sim) return;
  state.sim = sim;
  $id("simResults").hidden = false;
  const s = sim.summary;
  $id("simResultInfo").textContent = `${sim.player} · ${sim.difficulty}`;
  $id("simSummary").innerHTML = [
    ["Games", fmtInt(s.boards)], ["Won", `${fmtInt(s.wins)} (${fmtPct(s.win_rate)})`],
    ["95% interval", `${fmtPct(s.ci_low)} – ${fmtPct(s.ci_high)}`], ["Cleared on average", fmtPct(s.cleared)],
    ["Moves per game", fmtNum(s.moves, 1)],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  renderSimGames();
}

function renderSimGames() {
  if (!state.sim) return;
  const filter = $id("simFilter").value;
  const rows = state.sim.records.map((r, i) => ({ ...r, i })).filter((r) => filter === "all" || r.result === filter);
  $id("simGamesTable").innerHTML = rows.slice(0, 2000).map((r) => `<tr data-i="${r.i}">
    <td class="num">${r.seed}</td><td><span class="badge ${r.result}">${r.result}</span></td>
    <td class="num">${r.moves.length}</td><td class="num">${fmtPct(r.cleared, 0)}</td></tr>`).join("");
}
$id("simFilter").addEventListener("change", renderSimGames);
$id("simGamesTable").addEventListener("click", async (e) => {
  const tr = e.target.closest("tr[data-i]");
  if (!tr) return;
  document.querySelectorAll("#simGamesTable tr").forEach((r) => r.classList.toggle("selected", r === tr));
  const record = { ...state.sim.records[Number(tr.dataset.i)], path: state.sim.path };
  try { loadReplay(record, await api("/api/replay", { record }), false); } catch (err) { $id("simError").textContent = err.message; }
});

// -- replay viewer

const replay = {
  board: new MineBoard($id("rpBoard"), { maxCell: 34, maxHeight: 560 }),
  record: null, data: null, n: 0, frame: 0, cells: null, cellsFrame: 0,
  playing: false, acc: 0, last: 0, probs: null, probsKey: null,
};

function speed() { return $id("rpSpeed").value; }

function loadReplay(record, data, autoplay) {
  replay.record = record;
  replay.data = data;
  replay.n = data.moves.length;
  replay.probs = null;
  replay.probsKey = null;
  $id("rpScrub").max = replay.n;
  $id("rpInfo").textContent = `${record.player} · ${data.label} · seed ${data.seed}`;
  const badge = $id("rpBadge");
  badge.hidden = false;
  badge.textContent = data.result;
  badge.className = `badge ${data.result}`;
  const isNet = record.path && !["random", "solver"].includes(record.path);
  $id("rpOverlaySource").querySelector('[value="player"]').disabled = !isNet;
  if (!isNet) $id("rpOverlaySource").value = "solver";
  seek(0);
  if ($id("rpOverlay").checked) fetchProbs();
  if (autoplay) play();
  else pause();
}

function cellsAt(f) {
  const d = replay.data;
  if (!replay.cells || f < replay.cellsFrame) {
    replay.cells = new Int8Array(d.rows * d.cols).fill(COVERED);
    replay.cellsFrame = 0;
  }
  for (let k = replay.cellsFrame; k < f; k++) {
    for (const idx of d.opened[k]) replay.cells[idx] = d.mines[idx] ? EXPLODED : d.counts[idx];
  }
  replay.cellsFrame = f;
  return replay.cells;
}

function seek(f) {
  replay.frame = Math.max(0, Math.min(replay.n, f));
  drawReplay();
}

function drawReplay() {
  const d = replay.data;
  if (!d) return;
  const f = replay.frame;
  const end = f === replay.n;
  const probs = $id("rpOverlay").checked && replay.probs && f > 0 && !end ? replay.probs[f] : null;
  replay.board.set({
    rows: d.rows, cols: d.cols, cells: cellsAt(f), probs,
    next: !end && $id("rpNext").checked ? d.moves[f] : null,
    mines: end ? d.mines : null, won: d.result === "won",
  });
  $id("rpScrub").value = f;
  $id("rpFrame").textContent = f === 0 ? `start / ${replay.n}` : `move ${f} / ${replay.n}`;
  let revealed = 0;
  for (const v of replay.cells) if (v !== COVERED && v !== EXPLODED) revealed++;
  const safe = d.rows * d.cols - d.mines_total;
  const status = end ? (d.result === "won" ? `won in ${replay.n} moves` : `lost after ${replay.n} moves`) : `${fmtPct(revealed / safe, 0)} cleared`;
  const pr = probs && d.moves[f] ? probs[d.moves[f][0] * d.cols + d.moves[f][1]] : null;
  $id("rpMeta").innerHTML = `<span><b>${status}</b></span>` +
    (pr !== null && pr >= 0 ? `<span>next click: P(mine) ${pr}%</span>` : "") +
    (probs ? '<span class="legend-scale">safe <i></i> mine</span>' : "");
}

function play() {
  if (!replay.data) return;
  if (speed() === "instant") { pause(); seek(replay.n); return; }
  if (replay.frame >= replay.n) seek(0);
  replay.playing = true;
  replay.acc = 0;
  replay.last = performance.now();
  $id("rpPlay").textContent = "⏸";
  requestAnimationFrame(tick);
}

function pause() {
  replay.playing = false;
  $id("rpPlay").textContent = "▶";
}

function tick(now) {
  if (!replay.playing) return;
  const rate = BASE_RATE * Number(speed());
  replay.acc += ((now - replay.last) / 1000) * rate;
  replay.last = now;
  const steps = Math.floor(replay.acc);
  if (steps > 0) {  // at high speeds several moves pass between two animation frames; draw only the last
    replay.acc -= steps;
    seek(replay.frame + steps);
  }
  if (replay.frame >= replay.n) { pause(); return; }
  requestAnimationFrame(tick);
}

$id("rpPlay").addEventListener("click", () => (replay.playing ? pause() : play()));
$id("rpBack").addEventListener("click", () => { pause(); seek(replay.frame - 1); });
$id("rpFwd").addEventListener("click", () => { pause(); seek(replay.frame + 1); });
$id("rpStart").addEventListener("click", () => { pause(); seek(0); });
$id("rpEnd").addEventListener("click", () => { pause(); seek(replay.n); });
$id("rpScrub").addEventListener("input", (e) => { pause(); seek(Number(e.target.value)); });
$id("rpSpeed").addEventListener("change", () => { if (speed() === "instant" && replay.playing) play(); });
$id("rpNext").addEventListener("change", drawReplay);
$id("rpOverlay").addEventListener("change", () => { if ($id("rpOverlay").checked) fetchProbs(); drawReplay(); });
$id("rpOverlaySource").addEventListener("change", () => { if ($id("rpOverlay").checked) fetchProbs(); });
document.addEventListener("keydown", (e) => {
  if (!$id("tab-simulate").classList.contains("active") || !replay.data) return;
  if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName) || document.querySelector("dialog[open]")) return;
  if (e.key === " ") { e.preventDefault(); replay.playing ? pause() : play(); }
  else if (e.key === "ArrowLeft") { pause(); seek(replay.frame - 1); }
  else if (e.key === "ArrowRight") { pause(); seek(replay.frame + 1); }
});

function overlayPlayer() {
  return $id("rpOverlaySource").value === "player" && replay.record.path ? replay.record.path : "solver";
}

async function fetchProbs() {
  if (!replay.record) return;
  const player = overlayPlayer();
  const key = `${player}|${replay.record.seed}|${replay.record.moves.length}`;
  if (replay.probsKey === key) return;
  replay.probsKey = key;
  replay.probs = null;
  $id("rpMeta").textContent = "computing P(mine) for every move…";
  try {
    const res = await api("/api/replay/probabilities", { record: replay.record, player });
    if (replay.probsKey === key) { replay.probs = res.probs; drawReplay(); }
  } catch (err) { $id("simError").textContent = err.message; replay.probsKey = null; }
}

$id("btnGif").addEventListener("click", async () => {
  if (!replay.record) return;
  $id("btnGif").disabled = true;
  $id("gifStatus").textContent = "rendering…";
  try {
    const res = await api("/api/replay/gif", {
      record: replay.record, player: overlayPlayer(), overlay: $id("rpOverlay").checked,
      fps: Number($id("gifFps").value) || 8,
    });
    const blob = await res.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `minesweepml-${replay.data.spec.replace("/", "-")}-seed${replay.data.seed}.gif`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
    $id("gifStatus").textContent = `${(blob.size / 1e6).toFixed(1)} MB`;
  } catch (err) { $id("gifStatus").textContent = err.message; } finally { $id("btnGif").disabled = false; }
});

// ---------------------------------------------------------------- benchmark tab

async function refreshBenchmarks() {
  const d = await api("/api/benchmarks");
  state.benchResults = d.results;
  $id("benchProcs").placeholder = d.default_processes;
  renderJob(d.job);
  renderBenchHistory();
  if (!state.benchShown && d.results.length) showBenchResult(d.results[d.results.length - 1]);
}

$id("benchForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  $id("benchError").textContent = "";
  const difficulties = [...document.querySelectorAll("input[name=benchDiff]:checked")].map((i) => i.value);
  try {
    await api("/api/benchmarks", {
      player: $id("benchPlayer").value, difficulties, boards: Number($id("benchBoards").value),
      processes: $id("benchProcs").value === "" ? null : Number($id("benchProcs").value), quality: $id("benchQuality").checked,
    });
  } catch (err) { $id("benchError").textContent = err.message; }
});
$id("btnBenchCancel").addEventListener("click", () => api("/api/jobs/cancel", {}));

function baselineFor(name, boards) {
  return state.benchResults.filter((r) => r.player === name && r.boards === boards).pop() || null;
}

function showBenchResult(result) {
  state.benchShown = result;
  const rows = [result];
  for (const b of ["solver", "random"]) {
    const base = baselineFor(b, result.boards);
    if (base && base !== result) rows.push(base);
  }
  const diffs = DIFFS.filter((d) => result.difficulties[d]);
  const quality = rows.some((r) => r.quality);
  $id("benchResultInfo").textContent = `${result.player} · ${fmtInt(result.boards)} boards per difficulty · ${fmtDate(result.created)}`;
  $id("benchResult").innerHTML = `<table class="data-table"><thead><tr><th>Player</th><th>Difficulty</th>
    <th class="num">Won</th><th class="num">95% interval</th><th class="num">Cleared</th><th class="num">Moves</th>
    ${quality ? '<th class="num">Guesses</th><th class="num">Blunders</th>' : ""}</tr></thead><tbody>` +
    rows.flatMap((r, ri) => diffs.filter((d) => r.difficulties[d]).map((d) => {
      const s = r.difficulties[d];
      return `<tr class="${ri === 0 ? "selected" : ""}"><td>${esc(r.player)}</td><td>${DIFF_LABEL[d]}</td>
        <td class="num">${fmtPct(s.win_rate)}</td><td class="num">${fmtPct(s.ci_low)} – ${fmtPct(s.ci_high)}</td>
        <td class="num">${fmtPct(s.cleared)}</td><td class="num">${fmtNum(s.moves, 1)}</td>
        ${quality ? `<td class="num">${s.guesses !== undefined ? fmtNum(s.guesses, 2) : "–"}</td><td class="num">${s.blunders !== undefined ? fmtNum(s.blunders, 2) : "–"}</td>` : ""}</tr>`;
    })).join("") + "</tbody></table>";
  renderBenchHistory();
}

function renderBenchHistory() {
  const results = state.benchResults.map((r, i) => ({ r, i })).reverse();
  const cell = (r, d) => (r.difficulties[d] ? fmtPct(r.difficulties[d].win_rate) : "–");
  $id("benchHistory").innerHTML = results.map(({ r, i }) => `<tr data-i="${i}" class="${r === state.benchShown ? "selected" : ""}">
    <td>${fmtDate(r.created)}</td><td>${esc(r.player)}</td><td class="num">${fmtInt(r.boards)}</td>
    <td>${cell(r, "beginner")}</td><td>${cell(r, "intermediate")}</td><td>${cell(r, "expert")}</td></tr>`).join("")
    || '<tr><td colspan="6" class="muted">No benchmarks yet.</td></tr>';
}
$id("benchHistory").addEventListener("click", (e) => {
  const tr = e.target.closest("tr[data-i]");
  if (tr) showBenchResult(state.benchResults[Number(tr.dataset.i)]);
});

// ---------------------------------------------------------------- jobs (benchmark + simulation batches)

function renderJob(job) {
  state.job = job;
  const running = job && ["starting", "running"].includes(job.state);
  const kind = job?.kind;
  $id("btnBench").disabled = running;
  $id("btnBatch").disabled = running;
  $id("btnBenchCancel").hidden = !(running && kind === "benchmark");
  $id("btnSimCancel").hidden = !(running && kind === "simulate");
  const bench = $id("benchProgress"), sim = $id("simProgress");
  if (running && kind === "benchmark") {
    setProgress(bench, job.done, job.total);
    $id("benchStatus").textContent = `Benchmarking ${job.params.player.split("/").slice(-2).join("/")}: ${job.done} of ${job.total || "?"} chunks`;
  } else bench.hidden = true;
  if (running && kind === "simulate") setProgress(sim, job.done, job.total);
  else sim.hidden = true;
  if (!running && kind === "benchmark" && job.state !== "completed") $id("benchStatus").textContent = job.message || job.state;
  if (running && kind !== "benchmark") $id("benchStatus").textContent = "A simulation batch is running; the benchmark can start when it ends.";
  if (!running && kind === "benchmark" && job.state === "completed") $id("benchStatus").textContent = "";
}

async function onJobEvent(ev) {
  const job = state.job || { kind: ev.kind, params: {} };
  if (ev.event === "starting") return renderJob(ev.job);
  if (ev.event === "started") job.state = "running";
  if (ev.event === "progress") Object.assign(job, { done: ev.done, total: ev.total });
  if (["completed", "cancelled", "error"].includes(ev.event)) { job.state = ev.event; job.message = ev.message; }
  renderJob(job);
  if (ev.event === "completed" && ev.kind === "simulate") await showSimulation();
  if (ev.event === "completed" && ev.kind === "benchmark") {
    await refreshBenchmarks();
    showBenchResult(state.benchResults[state.benchResults.length - 1]);
    refreshBots();
  }
  if (ev.event === "error") (ev.kind === "benchmark" ? $id("benchError") : $id("simError")).textContent = ev.message;
}

// ---------------------------------------------------------------- live connection

function handleEvent(ev) {
  switch (ev.type) {
    case "hello":
      renderStatus(ev.status);
      renderLive(ev.live);
      renderGrid(ev.grid);
      if (ev.job) renderJob(ev.job);
      break;
    case "status": renderStatus(ev); break;
    case "log": log(ev.message); break;
    case "train":
      if (state.chartBot === state.status.bot) charts.loss.push(0, [ev.elapsed / 60, ev.loss]);
      break;
    case "system": onSystem(ev); break;
    case "eval":
    case "checkpoint":
      if (state.chartBot === state.status.bot) loadHistory(state.chartBot);
      refreshBots();
      break;
    case "live": renderLive(ev); break;
    case "grid": renderGrid(ev); break;
    case "job": onJobEvent(ev); break;
    default: break;
  }
}

function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  const conn = $id("conn");
  ws.onopen = () => { conn.textContent = "live"; conn.className = "conn ok"; };
  ws.onmessage = (m) => handleEvent(JSON.parse(m.data));
  ws.onclose = () => { conn.textContent = "reconnecting…"; conn.className = "conn bad"; setTimeout(connect, 2000); };
}

// ---------------------------------------------------------------- start

async function init() {
  state.info = await api("/api/info");
  updatePresetHint();
  renderStatus(state.info.status);
  renderLive(state.info.live);
  renderGrid(state.info.grid);
  await refreshBots();
  if (state.info.job) renderJob(state.info.job);
  const tab = location.hash.slice(1);
  showTab(["bots", "train", "simulate", "benchmark"].includes(tab) ? tab : "bots");
  connect();
}

init();
