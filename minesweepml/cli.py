"""Command line interface: python -m minesweepml {train,evaluate,simulate,gif,serve,report,list}."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from .config import CPU_COUNT, PRESET_NAMES, TrainConfig, all_presets, format_duration, parse_duration


def _add_training_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("intensity (preset values can be overridden individually)")
    g.add_argument("--preset", choices=PRESET_NAMES, default="balanced")
    g.add_argument("--workers", type=int, help="worker processes that play games")
    g.add_argument("--threads", type=int, help="torch threads for the trainer process")
    g.add_argument("--batch-size", type=int, help="training batch size")
    g.add_argument("--nice", type=int, help="process niceness of trainer and workers (0-19)")
    g.add_argument("--buffer-size", type=int, help="replay buffer size in positions (about 1 KB each at Expert)")
    d = p.add_argument_group("duration (default: until stopped)").add_mutually_exclusive_group()
    d.add_argument("--games", type=int, help="stop after N games")
    d.add_argument("--time", help="stop after a wall-clock time such as 2h, 90m, 1h30m")
    d.add_argument("--until-stop", action="store_true", help="run until Ctrl+C (default)")
    b = p.add_argument_group("bot (created if it does not exist; settings given here are saved with it)")
    b.add_argument("--bot", "--name", dest="bot", help="bot to train (default: a new bot with a generated name)")
    b.add_argument("--fork", metavar="CHECKPOINT", help="start a new bot from a copy of this checkpoint")
    b.add_argument("--blocks", type=int, help="residual blocks (new bots only)")
    b.add_argument("--channels", type=int, help="channels per block (new bots only)")
    b.add_argument("--mix", help="share of each difficulty, e.g. beginner=0.2,intermediate=0.3,expert=0.5")
    b.add_argument("--eps-start", type=float, help="exploration at the start (share of random clicks)")
    b.add_argument("--eps-end", type=float, help="exploration after --eps-games games")
    b.add_argument("--eps-games", type=int, help="games over which exploration decays")
    b.add_argument("--record-prob", type=float, help="share of positions stored for training")
    b.add_argument("--forgive-share", type=float,
                   help="share of games that continue after a mine, for data (results still count as lost)")
    b.add_argument("--lr", type=float, help="learning rate")
    b.add_argument("--first-click", choices=("open", "safe-cell"),
                   help="open: the first click and its neighbours are mine-free (default); safe-cell: only the cell")
    a = p.add_argument_group("advanced")
    a.add_argument("--parallel-games", type=int, help="games played together per worker")
    a.add_argument("--reset-lr", action="store_true", help="undo earlier learning-rate drops for this session")
    a.add_argument("--checkpoint-minutes", type=float, help="minutes between checkpoints")
    a.add_argument("--eval-minutes", type=float, help="minutes between evaluations (0 = off)")
    a.add_argument("--eval-boards", type=int, help="benchmark boards per difficulty in an evaluation")
    a.add_argument("--seed", type=int)


def parse_mix(text: str) -> dict:
    out = {}
    for part in text.split(","):
        name, _, value = part.partition("=")
        name = name.strip().lower()
        if name not in ("beginner", "intermediate", "expert"):
            raise ValueError(f"unknown difficulty {name!r} in --mix")
        out[f"mix_{name}"] = float(value)
    for name in ("beginner", "intermediate", "expert"):
        out.setdefault(f"mix_{name}", 0.0)
    return out


def config_from_args(args) -> tuple[TrainConfig, dict]:
    """The session configuration, and the bot settings given explicitly on the command line."""
    bot = {"blocks": args.blocks, "channels": args.channels, "eps_start": args.eps_start, "eps_end": args.eps_end,
           "eps_games": args.eps_games, "record_prob": args.record_prob, "forgive_share": args.forgive_share,
           "lr": args.lr}
    if args.mix:
        bot.update(parse_mix(args.mix))
    if args.first_click:
        bot["safe_neighbours"] = args.first_click == "open"
    bot = {k: v for k, v in bot.items() if v is not None}
    overrides = {
        "workers": args.workers, "threads": args.threads, "batch_size": args.batch_size, "nice": args.nice,
        "buffer_size": args.buffer_size, "parallel_games": args.parallel_games,
        "checkpoint_minutes": args.checkpoint_minutes, "eval_minutes": args.eval_minutes,
        "eval_boards": args.eval_boards, "seed": args.seed, "max_games": args.games,
        "max_seconds": parse_duration(args.time) if args.time else None,
        "reset_lr": True if args.reset_lr else None, **bot,
    }
    return TrainConfig.from_preset(args.preset, **overrides), bot


class ConsolePrinter:
    """Prints training events: a status line every few seconds plus logs and evaluations."""

    def __init__(self, every: float = 10.0):
        self.every = every
        self.last = 0.0
        self.loss = None

    def __call__(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "train":
            self.loss = event
        elif kind == "log":
            print(f"[{time.strftime('%H:%M:%S')}] {event['message']}", flush=True)
        elif kind == "status":
            if event["state"] in ("running", "paused") and time.time() - self.last < self.every:
                return
            self.last = time.time()
            loss = f"loss {self.loss['loss']:.4f}" if self.loss else "loss -"
            eta = f" | eta {format_duration(event['eta'])}" if event.get("eta") is not None else ""
            print(f"[{format_duration(event['elapsed'])}] {event['state']} | games {event['games']:,} "
                  f"({event['games_per_sec']:.1f}/s, {event['moves_per_sec']:.0f} moves/s) | steps "
                  f"{event['steps']:,} | {loss} | eps {event['eps']:.3f} | buffer {event['buffer']:,}{eta}",
                  flush=True)


def cmd_train(args) -> int:
    from .trainer import Control, Trainer

    try:
        cfg, bot_overrides = config_from_args(args)
        cfg.mix()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    fork = None
    if args.fork:
        from .runs import resolve_checkpoint

        fork = str(resolve_checkpoint(args.fork))
    control = Control()
    presses = 0

    def on_signal(signum, frame):
        nonlocal presses
        presses += 1
        if presses == 1:
            print("\nStopping gracefully: discarding unfinished games and saving a checkpoint "
                  "(press Ctrl+C again to quit immediately)", flush=True)
            control.stop.set()
        else:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    print(f"Training: preset {cfg.preset} ({CPU_COUNT} CPUs) workers={cfg.workers} threads={cfg.threads} "
          f"batch={cfg.batch_size} nice={cfg.nice} | {cfg.duration_text()}", flush=True)
    trainer = Trainer(cfg, control, ConsolePrinter(), bot=args.bot, fork=fork, bot_overrides=bot_overrides)
    try:
        reason = trainer.run()
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Bot directory: {trainer.bot_dir}")
    return 0 if reason in ("completed", "stopped") else 1


def _print_benchmark(result: dict) -> None:
    quality = result.get("quality")
    head = f"{'difficulty':<14} {'boards':>6} {'won':>7} {'95% interval':>17} {'cleared':>8} {'moves':>7}"
    print(head + (f" {'guesses':>8} {'blunders':>9}" if quality else ""))
    for d, s in result["difficulties"].items():
        line = (f"{d:<14} {s['boards']:>6} {100 * s['win_rate']:>6.1f}% "
                f"{100 * s['ci_low']:>7.1f}% - {100 * s['ci_high']:>5.1f}% {100 * s['cleared']:>7.1f}% "
                f"{s['moves']:>7.1f}")
        if quality:
            line += f" {s['guesses']:>8.2f} {s['blunders']:>9.2f}"
        print(line)


def cmd_evaluate(args) -> int:
    from .benchmark import run_benchmark, save_result
    from .runs import resolve_checkpoint

    player = args.player
    if player not in ("random", "solver"):
        try:
            player = str(resolve_checkpoint(player))
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    difficulties = [d.strip() for d in args.difficulties.split(",")]
    print(f"Benchmark: {player} on {args.boards} boards per difficulty ({', '.join(difficulties)}), "
          f"{args.processes} processes", flush=True)

    def progress(done, total, partial):
        print(f"\r  {done}/{total} chunks", end="", flush=True)

    result = run_benchmark(player, difficulties, args.boards, args.processes, quality=args.quality,
                           safe_neighbours=args.first_click == "open", nice=args.nice, progress=progress)
    print(f"\r{result['player']}: {result['seconds']:.0f}s")
    _print_benchmark(result)
    if not args.no_save:
        save_result(result)
    return 0


def cmd_simulate(args) -> int:
    from .players import make_player, play_games
    from .runs import resolve_checkpoint

    import torch

    torch.set_num_threads(args.threads)
    spec = args.player
    if spec not in ("random", "solver"):
        spec = str(resolve_checkpoint(spec))
    player = make_player(spec, seed=args.seed)
    seeds = [args.seed + i for i in range(args.games)]
    start = time.time()
    records = play_games(player, args.difficulty, seeds, safe_neighbours=args.first_click == "open")
    secs = time.time() - start
    wins = sum(r.result == "won" for r in records)
    moves = sum(len(r.moves) for r in records)
    print(f"{player.name} on {args.difficulty}: {wins}/{len(records)} won ({100 * wins / len(records):.1f}%), "
          f"{sum(r.cleared for r in records) / len(records):.1%} cleared on average, "
          f"{moves / len(records):.1f} moves per game, {moves / secs:.0f} moves/s")
    if args.out:
        Path(args.out).write_text(json.dumps([r.to_dict() for r in records]))
        print(f"Game records -> {args.out}")
    return 0


def cmd_gif(args) -> int:
    from .gif import render_gif

    try:
        res = render_gif(args.player, args.out, args.difficulty, args.seed, title=args.title, fps=args.fps,
                         overlay=args.overlay, cell=args.cell, hold_ms=args.hold_ms,
                         safe_neighbours=args.first_click == "open")
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"{res['result']} after {res['moves']} moves ({100 * res['cleared']:.0f}% cleared) -> {res['path']} "
          f"({res['bytes'] / 1e6:.1f} MB, {res['frames']} frames)")
    return 0


def cmd_serve(args) -> int:
    from .server import serve

    serve(args.host, args.port, open_browser=not args.no_browser)
    return 0


def cmd_report(args) -> int:
    from .report import generate_report

    generate_report(bot=args.bot, readme=Path(args.readme), out_dir=Path(args.out), export_best=args.export_best)
    return 0


def cmd_list(args) -> int:
    from .runs import list_bots, list_checkpoints

    print("Presets on this machine:")
    for name, values in all_presets().items():
        print(f"  {name:<9} " + " ".join(f"{k}={v}" for k, v in values.items()))
    print("\nBots:")
    for b in list_bots():
        net = b["net"]
        print(f"  {b['name']:<28} {b['status']:<10} games={b['games']:<8} steps={b['steps']:<7} "
              f"time={format_duration(b['elapsed'])} net={net['blocks']}x{net['channels']} "
              f"checkpoints={b['checkpoints']}")
    print("\nCheckpoints:")
    for c in list_checkpoints():
        print(f"  {c['label']}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m minesweepml", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    procs = max(1, CPU_COUNT - 2)

    p = sub.add_parser("train", help="train a bot on games it plays itself")
    _add_training_args(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="benchmark a bot, checkpoint or baseline on fixed seeded boards")
    p.add_argument("player", help="random, solver, a bot name (latest checkpoint), checkpoint file or label")
    p.add_argument("--boards", type=int, default=1000, help="boards per difficulty (default 1000)")
    p.add_argument("--difficulties", default="beginner,intermediate,expert")
    p.add_argument("--quality", action="store_true", help="also count forced guesses and blunders (slower)")
    p.add_argument("--processes", type=int, default=procs, help=f"parallel processes (default {procs})")
    p.add_argument("--nice", type=int, default=10)
    p.add_argument("--first-click", choices=("open", "safe-cell"), default="open")
    p.add_argument("--no-save", action="store_true", help="do not add the result to runs/benchmarks.json")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("simulate", help="play games and print a summary")
    p.add_argument("player", help="random, solver, a bot name, checkpoint file or label")
    p.add_argument("--difficulty", default="expert", help="beginner, intermediate, expert or ROWSxCOLS/MINES")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--seed", type=int, default=0, help="seed of the first board; game i uses seed + i")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--first-click", choices=("open", "safe-cell"), default="open")
    p.add_argument("--out", help="save the game records as JSON")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("gif", help="render one game as an animated GIF")
    p.add_argument("player", help="random, solver, a bot name (latest checkpoint), checkpoint file or label")
    p.add_argument("--difficulty", default="expert")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--title", help="text above the board (default: the player and difficulty)")
    p.add_argument("--fps", type=float, default=8.0, help="moves per second")
    p.add_argument("--overlay", action=argparse.BooleanOptionalAction, default=True,
                   help="tint covered cells by the network's P(mine) (networks and the solver)")
    p.add_argument("--cell", type=int, default=20, help="cell size in pixels")
    p.add_argument("--hold-ms", type=int, default=3000, help="how long the final frame is shown")
    p.add_argument("--first-click", choices=("open", "safe-cell"), default="open")
    p.add_argument("--out", default="docs/games/game.gif")
    p.set_defaults(func=cmd_gif)

    p = sub.add_parser("serve", help="start the web interface")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("report", help="regenerate result charts and the README results section")
    p.add_argument("--bot", help="bot to report on (default: the bot with the most games)")
    p.add_argument("--readme", default="README.md")
    p.add_argument("--out", default="docs/results", help="directory for chart images")
    p.add_argument("--export-best", action="store_true",
                   help="also copy the best evaluated checkpoint (weights only) to models/best.pt")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("list", help="show presets, bots and checkpoints")
    p.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
