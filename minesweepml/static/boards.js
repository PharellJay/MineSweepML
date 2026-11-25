// Minesweeper boards on <canvas>: one draw call per frame, so replays can run at any speed.
// Colours come from CSS custom properties, so the light and dark themes both work.

const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const TEXT = ".012345678*";

export const COVERED = -1;
export const EXPLODED = 9;

/** '.012345678*' text (as sent by the server) -> Int8Array of -1..9. */
export function parseCells(text) {
  const out = new Int8Array(text.length);
  for (let i = 0; i < text.length; i++) out[i] = TEXT.indexOf(text[i]) - 1;
  return out;
}

function palette() {
  const p = {};
  for (const k of ["covered", "edge", "open", "grid", "safe", "danger", "boom", "mine", "flag"]) p[k] = cssVar(`--cell-${k}`);
  p.n = [null, 1, 2, 3, 4, 5, 6, 7, 8].map((i) => (i ? cssVar(`--n${i}`) : null));
  p.accent = cssVar("--accent");
  return p;
}

export class MineBoard {
  // opts: { maxCell, minCell, numbers (draw digits), fixedCell (thumbnail size) }
  constructor(el, opts = {}) {
    this.el = el;
    this.opts = { maxCell: 34, minCell: 3, numbers: true, ...opts };
    this.canvas = document.createElement("canvas");
    el.innerHTML = "";
    el.appendChild(this.canvas);
    this.state = null;
    this.pending = false;
    if (!opts.fixedCell) new ResizeObserver(() => this.schedule()).observe(el);
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => this.schedule());
  }

  /** state: { rows, cols, cells, probs?, next?: [r, c], mines?: 0/1 array, won? } */
  set(state) { this.state = state; this.schedule(); }

  schedule() {
    if (this.pending) return;
    this.pending = true;
    requestAnimationFrame(() => { this.pending = false; this.draw(); });
  }

  cellSize() {
    const { rows, cols } = this.state;
    if (this.opts.fixedCell) return this.opts.fixedCell;
    const width = this.el.clientWidth || cols * 20;
    const byWidth = Math.floor(width / cols);
    const byHeight = this.opts.maxHeight ? Math.floor(this.opts.maxHeight / rows) : Infinity;
    return Math.max(this.opts.minCell, Math.min(this.opts.maxCell, byWidth, byHeight));
  }

  draw() {
    const s = this.state;
    if (!s) return;
    const { rows, cols, cells, probs, next, mines, won } = s;
    const c = this.cellSize();
    const dpr = window.devicePixelRatio || 1;
    const w = cols * c, h = rows * c;
    if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
      this.canvas.style.width = `${w}px`;
      this.canvas.style.height = `${h}px`;
    }
    const ctx = this.canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const p = palette();
    const numbers = this.opts.numbers && c >= 9;
    const gap = c >= 6 ? 1 : 0;
    ctx.fillStyle = p.grid;
    ctx.fillRect(0, 0, w, h);
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = `600 ${Math.round(c * 0.62)}px system-ui, sans-serif`;
    for (let r = 0; r < rows; r++) {
      for (let col = 0; col < cols; col++) {
        const i = r * cols + col;
        const v = cells[i];
        const x = col * c, y = r * c, size = c - gap;
        if (v === COVERED) {
          ctx.fillStyle = p.covered;
          ctx.fillRect(x, y, size, size);
          const pr = probs ? probs[i] : -1;
          if (pr !== undefined && pr >= 0) {
            ctx.globalAlpha = pr <= 2 ? 0.55 : 0.15 + 0.7 * Math.min(100, pr) / 100;
            ctx.fillStyle = pr <= 2 ? p.safe : p.danger;
            ctx.fillRect(x, y, size, size);
            ctx.globalAlpha = 1;
          } else if (c >= 8) {
            ctx.fillStyle = p.edge;
            ctx.fillRect(x, y, size, Math.max(1, c / 12));
            ctx.fillRect(x, y, Math.max(1, c / 12), size);
          }
          if (mines && mines[i]) (won ? this.flag(ctx, x, y, c, p) : this.mine(ctx, x, y, c, p.mine));
        } else if (v === EXPLODED) {
          ctx.fillStyle = p.boom;
          ctx.fillRect(x, y, size, size);
          this.mine(ctx, x, y, c, "#fff");
        } else {
          ctx.fillStyle = p.open;
          ctx.fillRect(x, y, size, size);
          if (v > 0) {
            if (numbers) {
              ctx.fillStyle = p.n[v];
              ctx.fillText(String(v), x + size / 2, y + size / 2 + 1);
            } else {
              ctx.fillStyle = p.n[v];
              ctx.globalAlpha = 0.55;
              ctx.fillRect(x + size * 0.3, y + size * 0.3, Math.max(1, size * 0.4), Math.max(1, size * 0.4));
              ctx.globalAlpha = 1;
            }
          }
        }
      }
    }
    if (next) {
      const [r, col] = next;
      ctx.strokeStyle = p.accent;
      ctx.lineWidth = Math.max(2, c / 10);
      ctx.strokeRect(col * c + 1, r * c + 1, c - gap - 2, c - gap - 2);
    }
  }

  mine(ctx, x, y, c, colour) {
    ctx.fillStyle = colour;
    ctx.beginPath();
    ctx.arc(x + c / 2, y + c / 2, Math.max(1, c * 0.24), 0, Math.PI * 2);
    ctx.fill();
  }

  flag(ctx, x, y, c, p) {
    const pole = x + c * 0.38;
    ctx.strokeStyle = p.mine;
    ctx.lineWidth = Math.max(1, c / 12);
    ctx.beginPath(); ctx.moveTo(pole, y + c * 0.22); ctx.lineTo(pole, y + c * 0.8); ctx.stroke();
    ctx.fillStyle = p.flag;
    ctx.beginPath();
    ctx.moveTo(pole, y + c * 0.22); ctx.lineTo(x + c * 0.78, y + c * 0.36); ctx.lineTo(pole, y + c * 0.5);
    ctx.closePath(); ctx.fill();
  }
}

/** A grid of clickable thumbnails (the live training games). */
export class MiniBoards {
  constructor(el, { onSelect }) {
    this.el = el;
    this.onSelect = onSelect;
    this.items = new Map();
  }

  render(games, selectedKey) {
    if (!games.length) {
      this.el.innerHTML = '<p class="mini-empty">Thumbnails of the running games appear here while training.</p>';
      this.items.clear();
      return;
    }
    if (this.el.querySelector(".mini-empty")) this.el.innerHTML = "";
    const keys = new Set();
    for (const g of games) {
      const key = `${g.worker}/${g.slot}`;
      keys.add(key);
      let item = this.items.get(key);
      if (!item) {
        const btn = document.createElement("button");
        btn.className = "mini";
        btn.type = "button";
        const holder = document.createElement("div");
        const label = document.createElement("span");
        label.className = "mini-label";
        btn.append(holder, label);
        btn.addEventListener("click", () => this.onSelect(g.worker, g.slot));
        this.el.appendChild(btn);
        item = { btn, label, board: new MineBoard(holder, { numbers: false, maxCell: 6, minCell: 2 }) };
        this.items.set(key, item);
      }
      item.btn.classList.toggle("selected", key === selectedKey);
      item.btn.title = `Worker ${g.worker}, game ${g.slot}: ${g.difficulty}, ${g.moves} moves`;
      item.label.textContent = `${g.difficulty[0].toUpperCase()} · ${g.moves}`;
      item.board.set({ rows: g.rows, cols: g.cols, cells: parseCells(g.cells) });
    }
    for (const [key, item] of this.items) {
      if (!keys.has(key)) { item.btn.remove(); this.items.delete(key); }
    }
  }
}
