"""`python -m minesweepml gif`: one game as an animated GIF, for the README.

Every move is a frame. Before each move the covered cells can be tinted by
the player's P(mine) (green: safe, orange: likely a mine) and the cell it is
about to click is outlined. The last frame holds the result: the exploded
mine and the other mines after a loss, flags on every mine after a win.
The colours are those of the web board's light theme.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .board import COVERED, EXPLODED, get_spec
from .players import NetPlayer, make_baseline, play_games
from .replay import GameRecord, frame_probabilities, frames

BG, INK, MUTED = (252, 252, 251), (11, 11, 11), (111, 110, 105)
GRID = (220, 219, 214)
COVERED_FILL = (196, 203, 212)
COVERED_EDGE = (214, 220, 227)
OPEN_FILL = (243, 243, 241)
ACCENT = (42, 120, 214)
SAFE = (27, 175, 122)
DANGER = (235, 104, 52)
BOOM = (214, 64, 52)
FLAG = (235, 104, 52)
NUMBER = {1: (42, 120, 214), 2: (24, 140, 92), 3: (214, 64, 52), 4: (31, 58, 147), 5: (150, 40, 40),
          6: (20, 140, 140), 7: (11, 11, 11), 8: (111, 110, 105)}
TINT_LEVELS = 12  # overlay colours are quantised to this many steps, which keeps the palette small


def _mix(a, b, t: float):
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def overlay_colour(pct: int):
    """Covered-cell colour for P(mine) in %: green when (nearly) safe, orange towards a mine."""
    if pct <= 2:
        return _mix(COVERED_FILL, SAFE, 0.55)
    step = round(min(100, pct) / 100 * TINT_LEVELS) / TINT_LEVELS
    return _mix(COVERED_FILL, DANGER, 0.15 + 0.7 * step)


class _Renderer:
    def __init__(self, rows: int, cols: int, cell: int):
        self.rows, self.cols, self.cell = rows, cols, cell
        self.pad = max(8, cell // 2)
        self.head = int(cell * 2.2)
        self.foot = int(cell * 1.4)
        self.width = cols * cell + 2 * self.pad
        self.height = self.head + rows * cell + self.foot
        self.num_font = ImageFont.load_default(size=max(10, int(cell * 0.7)))
        self.title_font = ImageFont.load_default(size=max(12, int(cell * 0.8)))
        self.font = ImageFont.load_default(size=max(10, int(cell * 0.6)))

    def frame(self, visible, title: str, status: str, footer: str, probs=None, target=None, mines=None,
              won: bool = False):
        img = Image.new("RGB", (self.width, self.height), BG)
        d = ImageDraw.Draw(img)
        d.text((self.pad, self.pad // 2), title, font=self.title_font, fill=INK)
        d.text((self.pad, self.pad // 2 + int(self.cell * 1.05)), status, font=self.font, fill=MUTED)
        c = self.cell
        for r in range(self.rows):
            for col in range(self.cols):
                x, y = self.pad + col * c, self.head + r * c
                v = int(visible[r, col])
                box = [x, y, x + c - 1, y + c - 1]
                if v == COVERED:
                    fill = COVERED_FILL
                    if probs is not None and probs[r * self.cols + col] >= 0:
                        fill = overlay_colour(probs[r * self.cols + col])
                    d.rectangle(box, fill=fill, outline=GRID)
                    d.line([x + 1, y + 1, x + c - 2, y + 1], fill=COVERED_EDGE)
                    d.line([x + 1, y + 1, x + 1, y + c - 2], fill=COVERED_EDGE)
                    if mines is not None and mines[r, col]:
                        if won:
                            self._flag(d, x, y)
                        else:
                            self._mine(d, x, y, INK)
                elif v == EXPLODED:
                    d.rectangle(box, fill=BOOM, outline=GRID)
                    self._mine(d, x, y, BG)
                else:
                    d.rectangle(box, fill=OPEN_FILL, outline=GRID)
                    if v > 0:
                        d.text((x + c / 2, y + c / 2 + 1), str(v), font=self.num_font, fill=NUMBER[v], anchor="mm")
        if target is not None:
            r, col = target
            x, y = self.pad + col * c, self.head + r * c
            d.rectangle([x - 1, y - 1, x + c, y + c], outline=ACCENT, width=2)
        d.text((self.pad, self.head + self.rows * c + self.foot // 2), footer, font=self.font, fill=MUTED,
               anchor="lm")
        return img

    def _mine(self, d, x, y, colour):
        c = self.cell
        r = c * 0.24
        cx, cy = x + c / 2, y + c / 2
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=colour)

    def _flag(self, d, x, y):
        c = self.cell
        pole = x + c * 0.38
        d.line([pole, y + c * 0.22, pole, y + c * 0.8], fill=INK, width=max(1, c // 12))
        d.polygon([(pole, y + c * 0.22), (x + c * 0.78, y + c * 0.36), (pole, y + c * 0.5)], fill=FLAG)


def _palette(renderer: _Renderer) -> Image.Image:
    """Every colour the frames use, so all frames share one exact palette."""
    colours = [BG, INK, MUTED, GRID, COVERED_FILL, COVERED_EDGE, OPEN_FILL, ACCENT, SAFE, DANGER, BOOM, FLAG,
               *NUMBER.values(), overlay_colour(0), *(overlay_colour(int(100 * k / TINT_LEVELS))
                                                    for k in range(TINT_LEVELS + 1))]
    # anti-aliased text needs the shades between the text colours and their backgrounds
    shades = []
    for fg in [INK, MUTED, BG, *NUMBER.values()]:
        for bg in (BG, OPEN_FILL, BOOM, COVERED_FILL):
            shades += [_mix(bg, fg, t / 6) for t in range(1, 6)]
    colours = list(dict.fromkeys(colours + shades))[:256]
    img = Image.new("P", (1, 1))
    flat = [v for c in colours for v in c]
    img.putpalette(flat + [0] * (768 - len(flat)))
    return img


def render_record(record: GameRecord, prob_player=None, title: str | None = None, fps: float = 8.0,
                  cell: int = 20, hold_ms: int = 3000) -> bytes:
    """A finished game as GIF bytes. `prob_player` (a NetPlayer or 'solver') draws the P(mine) overlay."""
    spec = get_spec(record.spec)
    renderer = _Renderer(spec.rows, spec.cols, cell)
    probs = frame_probabilities(record, prob_player) if prob_player is not None else None
    title = title or f"{record.player or 'player'} · {spec.label}"
    footer = f"seed {record.seed}" + (" · tint: P(mine)" if probs is not None else "")
    n = len(record.moves)
    images = []
    board = None
    for k, (board, _) in enumerate(frames(record)):
        if k < n:
            status = "opening click" if k == 0 else f"move {k} of {n} · {100 * board.cleared():.0f}% cleared"
            images.append(renderer.frame(board.visible(), title, status, footer,
                                         probs=probs[k] if probs is not None and k > 0 else None,
                                         target=record.moves[k]))
    won = board.state == "won"
    result = f"WON in {n} moves" if won else f"LOST after {n} moves · {100 * board.cleared():.0f}% cleared"
    images.append(renderer.frame(board.visible(), title, result, footer, mines=board.mines, won=won))
    palette = _palette(renderer)
    quantised = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    frame_ms = max(20, int(round(1000 / max(0.1, fps))))
    durations = [max(frame_ms, 800)] + [frame_ms] * (len(images) - 2) + [hold_ms]
    buf = io.BytesIO()
    quantised[0].save(buf, format="GIF", save_all=True, append_images=quantised[1:], duration=durations[:len(images)],
                      loop=0, optimize=False, disposal=1)
    return buf.getvalue()


def render_gif(player: str, out: str | Path, difficulty: str = "expert", seed: int = 0, title: str | None = None,
               fps: float = 8.0, overlay: bool = True, cell: int = 20, hold_ms: int = 3000,
               safe_neighbours: bool = True) -> dict:
    """Play one game with `player` ('random', 'solver' or a bot / checkpoint) and save it as a GIF."""
    from .runs import checkpoint_label, resolve_checkpoint

    if player in ("random", "solver"):
        obj = make_baseline(player, seed)
        label = player
    else:
        path = resolve_checkpoint(player)
        label = checkpoint_label(path)
        obj = NetPlayer(path, name=label)
    record = play_games(obj, difficulty, [seed], safe_neighbours, parallel=1)[0]
    record.player = label
    prob_player = None
    if overlay and player != "random":
        prob_player = "solver" if player == "solver" else obj
    data = render_record(record, prob_player=prob_player, title=title, fps=fps, cell=cell, hold_ms=hold_ms)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return {"path": out, "result": record.result, "moves": len(record.moves), "cleared": record.cleared,
            "bytes": len(data), "frames": len(record.moves) + 1, "record": record.to_dict()}
