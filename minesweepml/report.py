"""`python -m minesweepml report`: result charts and the README results section.

Everything is read from a bot's metrics database and the saved benchmark
results (runs/benchmarks.json), so the README only ever shows numbers that a
real run produced.
"""

from __future__ import annotations

import os
import platform
import re
import sys
from pathlib import Path

START, END = "<!-- results:start -->", "<!-- results:end -->"

# reference palette, light mode: categorical slots in fixed order (as in the web UI)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
DIFFS = ("beginner", "intermediate", "expert")
LABEL = {"beginner": "Beginner", "intermediate": "Intermediate", "expert": "Expert"}


def machine_info() -> dict:
    """Hardware and software of the current machine."""
    import psutil
    import torch

    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": "none (CPU only)" if not torch.cuda.is_available() else torch.cuda.get_device_name(0),
    }


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def _save(plt, fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="#fcfcfb")
    plt.close(fig)


def _legend(ax):
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED)


def make_charts(h: dict, out_dir: Path) -> list[tuple[str, str]]:
    plt = _plt()
    out_dir.mkdir(parents=True, exist_ok=True)
    charts = []

    if h["train"]:
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        ax.plot([r["elapsed"] / 60 for r in h["train"]], [r["loss"] for r in h["train"]], color=COLORS[0], lw=2)
        _style(ax, "Training loss (cross-entropy over covered cells)", "training time (minutes)", "loss")
        _save(plt, fig, out_dir / "loss.png")
        charts.append(("loss.png", "Training loss"))

    evals = h["evals"]
    if evals:
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        for i, d in enumerate(DIFFS):
            rows = [e for e in evals if e["difficulty"] == d]
            if rows:
                ax.plot([e["games"] for e in rows], [100 * e["wins"] / e["boards"] for e in rows], color=COLORS[i],
                        lw=2, marker="o", ms=4, label=LABEL[d])
        ax.set_ylim(0, 100)
        _style(ax, f"Benchmark win rate per checkpoint ({evals[0]['boards']} boards each)", "training games",
               "won %")
        _legend(ax)
        _save(plt, fig, out_dir / "benchmark.png")
        charts.append(("benchmark.png", "Benchmark win rate of the checkpoints during training"))

    if h["games"]:
        for key, title, file in (("win_rate", "Training games won, last 100 per difficulty", "winrate.png"),
                                 ("cleared", "Safe cells cleared in training games, last 100 per difficulty",
                                  "cleared.png")):
            fig, ax = plt.subplots(figsize=(6.4, 3.4))
            for i, d in enumerate(DIFFS):
                rows = [g for g in h["games"] if g["difficulty"] == d]
                if rows:
                    ax.plot([g["game"] for g in rows], [100 * g[key] for g in rows], color=COLORS[i], lw=1.6,
                            label=LABEL[d])
            ax.set_ylim(0, 100)
            _style(ax, title, "training games", "%")
            _legend(ax)
            _save(plt, fig, out_dir / file)
            charts.append((file, title))

    if h["system"]:
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        t = [r["elapsed"] / 60 for r in h["system"]]
        ax.plot(t, [r["moves_per_sec"] for r in h["system"]], color=COLORS[0], lw=1, label="moves / s")
        _style(ax, "Moves played per second (all workers)", "training time (minutes)", "moves / s")
        _save(plt, fig, out_dir / "throughput.png")
        charts.append(("throughput.png", "Moves per second"))

        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        ax.plot(t, [r["cpu_total"] for r in h["system"]], color=COLORS[0], lw=1)
        ax.set_ylim(0, 100)
        _style(ax, "CPU usage, all cores", "training time (minutes)", "%")
        _save(plt, fig, out_dir / "cpu.png")
        charts.append(("cpu.png", "Overall CPU usage"))
    return charts


def _pct(s: dict) -> str:
    return f"{100 * s['win_rate']:.1f}% ({100 * s['ci_low']:.1f}-{100 * s['ci_high']:.1f})"


def benchmark_section(bot_dir: Path) -> tuple[list[str], dict | None]:
    """The newest, largest benchmark of one of the bot's checkpoints, next to the baselines on as many boards."""
    from .benchmark import load_results

    results = load_results()
    mine = [r for r in results if r.get("path") and Path(r["path"]).resolve().parent.parent == bot_dir.resolve()]
    if not mine:
        return [], None
    boards = max(r["boards"] for r in mine)
    top = [r for r in mine if r["boards"] == boards]
    choice = best_checkpoint(bot_dir, {"evals": []})  # the best of these, as exported to models/best.pt
    best = next((r for r in reversed(top) if choice and Path(r["path"]).name == choice[0]), top[-1])
    rows = [(f"`{best['player']}`", best)]
    for name, label in (("solver", "rule-based solver"), ("random", "random clicks")):
        base = [r for r in results if r["player"] == name and r["boards"] == boards]
        if base:
            rows.append((label, base[-1]))
    diffs = [d for d in DIFFS if d in best["difficulties"]]
    lines = [f"### Benchmark: {boards:,} fixed boards per difficulty", "",
             "Win rate with its Wilson 95% interval. Every player plays the same seeded boards, opened on the "
             "centre cell; training never uses these seeds.", "",
             "| Player | " + " | ".join(LABEL[d] for d in diffs) + " |", "|---|" + "---|" * len(diffs)]
    for label, r in rows:
        lines.append(f"| {label} | " + " | ".join(_pct(r["difficulties"][d]) if d in r["difficulties"] else "-"
                                                   for d in diffs) + " |")
    lines += ["", "| Player | " + " | ".join(f"{LABEL[d]} cleared" for d in diffs) + " |",
              "|---|" + "---|" * len(diffs)]
    for label, r in rows:
        lines.append(f"| {label} | " + " | ".join(f"{100 * r['difficulties'][d]['cleared']:.1f}%"
                                                   if d in r["difficulties"] else "-" for d in diffs) + " |")
    if any("guesses" in r["difficulties"].get(d, {}) for _, r in rows for d in diffs):
        lines += ["", "Forced guesses (moves with no provably safe cell) and blunders (mines hit while a "
                  "provably safe cell existed) per game, judged by the solver:", "",
                  "| Player | " + " | ".join(f"{LABEL[d]} guesses / blunders" for d in diffs) + " |",
                  "|---|" + "---|" * len(diffs)]
        for label, r in rows:
            cells = []
            for d in diffs:
                s = r["difficulties"].get(d, {})
                cells.append(f"{s['guesses']:.2f} / {s['blunders']:.2f}" if "guesses" in s else "-")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines, best


def results_markdown(bot_dir: Path, h: dict, charts, image_dir: str, hardware: dict, hardware_note: str,
                     exported: dict | None = None) -> str:
    from .config import format_duration
    from .network import MineNet, NetConfig, count_parameters
    from .runs import read_json

    cfg = read_json(bot_dir / "config.json", {}) or {}
    state = read_json(bot_dir / "state.json", {}) or {}
    elapsed = state.get("elapsed", 0.0)
    games, positions, steps = state.get("games", 0), state.get("positions", 0), state.get("steps", 0)
    params = count_parameters(MineNet(NetConfig(cfg.get("blocks", 6), cfg.get("channels", 32))))
    lines = [START, "", f"Generated by `python -m minesweepml report` from bot `{bot_dir.name}`.", ""]

    lines += ["### The run", "", "| | |", "|---|---|"]
    lines.append(f"| Hardware | {hardware['cpu']}, {hardware['logical_cpus']} logical CPUs, "
                 f"{hardware['ram_gb']} GB RAM, GPU: {hardware['gpu']} |")
    lines.append(f"| Software | {hardware['os']}, Python {hardware['python']}, torch {hardware['torch']} |")
    lines.append(f"| Preset | `{cfg.get('preset')}`: {cfg.get('workers')} workers x {cfg.get('parallel_games')} games, "
                 f"{cfg.get('threads')} trainer threads, batch {cfg.get('batch_size')}, nice {cfg.get('nice')} |")
    lines.append(f"| Network | {cfg.get('blocks')} residual blocks x {cfg.get('channels')} channels, "
                 f"{params:,} parameters |")
    mix = ", ".join(f"{LABEL[d]} {100 * cfg.get(f'mix_{d}', 0) / max(1e-9, sum(cfg.get(f'mix_{k}', 0) for k in DIFFS)):.0f}%"
                    for d in DIFFS if cfg.get(f"mix_{d}", 0) > 0)
    lines.append(f"| Training mix | {mix} |")
    starts = [s["detail"] for s in h["sessions"] if s["event"] == "start"]
    lines.append(f"| Training sessions | {len(starts)}: " + "; ".join(starts) + " |")
    lines.append(f"| Training time | {format_duration(elapsed)} |")
    suspends = [s for s in h["sessions"] if s["event"] == "suspended"]
    if suspends:
        lines.append(f"| Computer suspended | {len(suspends)} time(s) during training; " +
                     "; ".join(s["detail"] for s in suspends) + " |")
    lines.append(f"| Games played | {games:,} |")
    lines.append(f"| Positions stored | {positions:,} |")
    lines.append(f"| Training steps | {steps:,} (batch {cfg.get('batch_size')}) |")
    if h["train"]:
        first, last = h["train"][0], h["train"][-1]
        lines.append(f"| Loss, first -> last logged | {first['loss']:.4f} -> {last['loss']:.4f} |")
    if hardware_note:
        lines += ["", hardware_note]
    lines.append("")

    bench_lines, _ = benchmark_section(bot_dir)
    lines += bench_lines

    evals = h["evals"]
    if evals:
        by_ckpt: dict[str, dict] = {}
        for e in evals:
            by_ckpt.setdefault(e["checkpoint"], {"games": e["games"]})[e["difficulty"]] = e
        diffs = [d for d in DIFFS if any(d in v for v in by_ckpt.values())]
        lines += [f"### Checkpoints during training ({evals[0]['boards']} benchmark boards per difficulty)", "",
                  "| Checkpoint | Games | " + " | ".join(f"{LABEL[d]} won" for d in diffs) + " |",
                  "|---|---|" + "---|" * len(diffs)]
        for name, v in by_ckpt.items():
            lines.append(f"| `{name}` | {v['games']:,} | " + " | ".join(
                f"{100 * v[d]['wins'] / v[d]['boards']:.1f}%" if d in v else "-" for d in diffs) + " |")
        lines.append("")
    if exported:
        lines += [f"The bundled `models/best.pt` is `{exported['source']}`, the checkpoint with the best "
                  f"mix-weighted win rate in {exported['basis']}, saved without optimiser state "
                  f"({exported['size_mb']:.2f} MB).", ""]

    if charts:
        lines += ["### Charts", ""]
        for file, alt in charts:
            lines.append(f"![{alt}]({image_dir}/{file})")
        lines.append("")
    lines.append(END)
    return "\n".join(lines)


def best_checkpoint(bot_dir: Path, h: dict) -> tuple[str, float, str] | None:
    """The checkpoint with the best mix-weighted win rate, and where that score comes from.

    Full benchmarks (runs/benchmarks.json) of the bot's checkpoints win over the
    smaller evaluations during training: the largest benchmark size decides.
    """
    from .benchmark import load_results
    from .config import TrainConfig
    from .runs import read_json

    cfg = TrainConfig.from_dict(read_json(bot_dir / "config.json", {}) or {})
    mix = cfg.mix()
    mine = [r for r in load_results() if r.get("path") and Path(r["path"]).resolve().parent.parent == bot_dir.resolve()
            and Path(r["path"]).exists() and all(d in r["difficulties"] for d in mix)]
    if mine:
        boards = max(r["boards"] for r in mine)
        ranked = [(sum(w * r["difficulties"][d]["win_rate"] for d, w in mix.items()), Path(r["path"]).name)
                  for r in mine if r["boards"] == boards]
        score, name = max(ranked)
        return name, score, f"the full benchmark ({boards:,} boards per difficulty)"
    scores: dict[str, list[float]] = {}
    for e in h["evals"]:
        s = scores.setdefault(e["checkpoint"], [0.0, 0.0])
        w = mix.get(e["difficulty"], 0.0)
        s[0] += w * e["wins"] / max(1, e["boards"])
        s[1] += w
    ranked = [(v[0] / v[1], k) for k, v in scores.items() if v[1] > 0 and (bot_dir / "checkpoints" / k).exists()]
    if not ranked:
        return None
    score, name = max(ranked)
    return name, score, "the evaluations during training"


def export_best_checkpoint(bot_dir: Path, h: dict) -> dict | None:
    """Copy the best evaluated checkpoint (weights only) to models/best.pt."""
    from .network import load_checkpoint, model_from_checkpoint, save_checkpoint
    from .runs import MODELS_DIR

    best = best_checkpoint(bot_dir, h)
    if best is None:
        return None
    name, score, basis = best
    data = load_checkpoint(str(bot_dir / "checkpoints" / name))
    MODELS_DIR.mkdir(exist_ok=True)
    dest = MODELS_DIR / "best.pt"
    save_checkpoint(str(dest), model_from_checkpoint(data), games=data.get("games"), steps=data.get("steps"),
                    elapsed=data.get("elapsed"), bot=bot_dir.name, source=name, score=score,
                    config=data.get("config"))
    return {"path": dest, "source": name, "score": score, "basis": basis, "size_mb": dest.stat().st_size / 1e6}


def generate_report(bot: str | None, readme: Path, out_dir: Path, export_best: bool = False) -> None:
    from .metrics import read_history
    from .runs import list_bots, read_json, resolve_bot

    if bot:
        bot_dir = resolve_bot(bot)
    else:
        bots = sorted(list_bots(), key=lambda b: b["games"], reverse=True)
        if not bots:
            sys.exit("No bots found. Train first: python -m minesweepml train")
        bot_dir = Path(bots[0]["path"])
    h = read_history(bot_dir, limit=5000)
    hardware = read_json(bot_dir / "hardware.json")
    note = ""
    if not hardware:
        hardware = machine_info()
        note = "_Hardware details describe the machine that generated this report (the bot did not record its own)._"
    exported = export_best_checkpoint(bot_dir, h) if export_best else None
    charts = make_charts(h, out_dir)
    try:
        image_dir = out_dir.resolve().relative_to(readme.resolve().parent).as_posix()
    except ValueError:
        image_dir = out_dir.as_posix()
    section = results_markdown(bot_dir, h, charts, image_dir, hardware, note, exported)
    text = readme.read_text() if readme.exists() else f"# MineSweepML\n\n## Results\n\n{START}\n{END}\n"
    if START in text and END in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: section, text, flags=re.S)
    else:
        text = text.rstrip() + "\n\n## Results\n\n" + section + "\n"
    readme.write_text(text)
    print(f"Updated {readme} from bot {bot_dir.name}; charts in {out_dir}")
    if exported:
        print(f"Exported {exported['source']} to {exported['path']} ({exported['size_mb']:.1f} MB)")
    elif export_best:
        print("No evaluated checkpoint to export")
