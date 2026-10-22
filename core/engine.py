"""Paced workload runner behind train.py, simulate.py, solve.py and search.py.

Nothing here computes anything real. Each job prints the log lines, progress bars and
converging metrics defined in core/jobs.py, paced so the run ends exactly at --walltime.
In a terminal it runs full-screen with an editor beside the log; see core/workspace.py.
"""
import argparse
import math
import os
import platform
import random
import re
import string
import sys
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import TextIO

from core.console import Display, Keyboard
from core.jobs import DEVICE, JOBS, OPTION_CHANGE, PROGRAM, VERSION
from core.workspace import Workspace

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WALLTIME = 60.0
MIN_WALLTIME = 5.0
SETUP_SHARE = 0.12  # of walltime; the main loop gets MAIN_SHARE, teardown the rest
MAIN_SHARE = 0.80
TICK = 0.1  # seconds between redraws
TYPING_TICK = 0.03  # faster redraws while the editor is live, so typing feels instant
LOG_INTERVAL = {0: 5.0, 1: 2.5, 2: 1.0}  # seconds between main-loop lines, by -v level
MIN_MAIN_LINES = 6  # short runs log more often so they never look empty
DEBUG_RATE = {0: 0.0, 1: 0.5, 2: 3.0}  # debug lines per second, by -v level
EVENT_CHANCE = 0.15
BAR_WIDTH = 24
EXIT_INTERRUPTED = 130

_PARTIALS = " ▏▎▍▌▋▊▉"
_UNIT_SECONDS = {"h": 3600, "m": 60, "s": 1}
_TIMESTAMP = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3} ", re.MULTILINE)
_FORMATTER = string.Formatter()


def parse_walltime(text: str) -> float:
    """Seconds from 'HH:MM:SS', 'MM:SS', '90', '90s', '5m' or '1h30m'."""
    text = text.strip().lower()
    if ":" in text:
        parts = text.split(":")
        if len(parts) > 3 or not all(part.isdigit() for part in parts):
            raise ValueError(f"bad walltime {text!r}")
        seconds = 0
        for part in parts:
            seconds = seconds * 60 + int(part)
        return float(seconds)
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return float(text)
    pieces = re.findall(r"(\d+(?:\.\d+)?)([hms])", text)
    if not pieces or "".join(n + u for n, u in pieces) != text:
        raise ValueError(f"bad walltime {text!r}")
    return sum(float(n) * _UNIT_SECONDS[u] for n, u in pieces)


def fmt_duration(seconds: float, always_hours: bool = False) -> str:
    total = int(round(max(0.0, seconds)))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours or always_hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def nice_stride(raw: float) -> int:
    """Round a log stride to the nearest 1/2/5 x 10^k, like real log intervals."""
    if raw <= 1:
        return 1
    magnitude = 10 ** math.floor(math.log10(raw))
    candidates = [m * magnitude for m in (1, 2, 5, 10)]
    return int(min(candidates, key=lambda c: abs(math.log(c / raw))))


def bar_text(label: str, done: int, total: int, elapsed: float, unit: str,
             ascii_only: bool = False) -> str:
    """A tqdm-style progress line."""
    frac = min(1.0, done / total) if total else 1.0
    cells = frac * BAR_WIDTH
    full = int(cells)
    if ascii_only:
        bar = "#" * full + " " * (BAR_WIDTH - full)
    else:
        bar = "█" * full
        if full < BAR_WIDTH:
            bar += _PARTIALS[int((cells - full) * len(_PARTIALS))] + " " * (BAR_WIDTH - full - 1)
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0.0
    return (f"{label}: {int(frac * 100):3d}%|{bar}| {done:,}/{total:,} "
            f"[{fmt_duration(elapsed)}<{fmt_duration(eta)}, {rate:,.2f}{unit}/s]")


def curve_value(spec: dict, p: float, rng: random.Random) -> float:
    """A metric's value at progress p in [0, 1], with noise that settles as p -> 1."""
    kind, start = spec["kind"], spec["start"]
    end = spec.get("end", start)
    if kind == "exp":
        value = end + (start - end) * math.exp(-spec.get("rate", 4.0) * p)
    elif kind == "log":
        lo, hi = math.log(max(start, 1e-300)), math.log(max(end, 1e-300))
        value = math.exp(lo + (hi - lo) * p)
    elif kind == "cosine":
        value = end + (start - end) * 0.5 * (1 + math.cos(math.pi * p))
    elif kind == "linear":
        value = start + (end - start) * p
    elif kind == "steady":
        value = start
    else:
        raise ValueError(f"unknown metric kind {kind!r}")
    settle = 1.0 if kind == "steady" else 0.2 + 0.8 * (1 - p)
    value *= 1 + spec.get("noise", 0.0) * settle * rng.gauss(0, 1)
    return min(spec.get("max", math.inf), max(spec.get("min", -math.inf), value))


class Context(dict):
    """Template values. Unknown keys like {hex8}, {int3}, {ms} are generated fresh each use."""

    _RANDOM = {
        "ms": lambda rng: rng.uniform(0.4, 38.0),
        "mb": lambda rng: rng.uniform(4.0, 512.0),
        "gb": lambda rng: rng.uniform(0.6, 14.0),
        "pct": lambda rng: rng.uniform(3.0, 27.0),
    }

    def __init__(self, rng: random.Random, values: dict) -> None:
        super().__init__(values)
        self.rng = rng

    def __missing__(self, key: str) -> object:
        if key in self._RANDOM:
            return self._RANDOM[key](self.rng)
        if key == "rank":
            return self.rng.randrange(max(1, self["devices"]))
        match = re.fullmatch(r"(hex|int)(\d+)", key)
        if not match:
            raise KeyError(f"template uses unknown value {{{key}}}")
        digits = int(match.group(2))
        if match.group(1) == "hex":
            return "".join(self.rng.choice("0123456789abcdef") for _ in range(digits))
        return self.rng.randrange(10 ** (digits - 1), 10 ** digits)


def _int_arg(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return int(float(text))  # accept 1e9
    except OverflowError:
        raise ValueError(f"out of range: {text}") from None


_int_arg.__name__ = "int"


def _walltime_arg(text: str) -> float:
    try:
        seconds = parse_walltime(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid walltime {text!r} (use HH:MM:SS, MM:SS, 90s, 5m or 1h30m)") from None
    if seconds <= 0:
        raise argparse.ArgumentTypeError("walltime must be positive")
    return seconds


def _coerce(text: str) -> int | float | str:
    for convert in (int, float):
        try:
            return convert(text)
        except ValueError:
            pass
    return text


def _parse_extras(tokens: list[str]) -> dict:
    """Unknown --flags become extra config entries, so anything typed looks accepted."""
    extras, i = {}, 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--") and len(token) > 2:
            key, has_value, value = token[2:].partition("=")
            if not has_value:
                if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                    value = tokens[i + 1]
                    i += 1
                else:
                    value = "true"
            extras[key.replace("-", "_")] = _coerce(value)
        i += 1
    return extras


def _paced_params(job: dict) -> tuple[str, ...]:
    """Params whose default is worked out from --walltime so the run moves at a real pace."""
    return tuple(filter(None, (job["units"], job.get("inner"))))


def parse_args(name: str, job: dict,
               argv: list[str] | None) -> tuple[argparse.Namespace, dict]:
    parser = argparse.ArgumentParser(prog=f"{name}.py", description=job["description"],
                                     allow_abbrev=False)
    paced = _paced_params(job)
    for key, (default, help_text) in job["params"].items():
        kind = {int: _int_arg, float: float}.get(type(default), str)
        shown = "scales with --walltime" if key in paced else default
        parser.add_argument("--" + key.replace("_", "-"), dest=key, type=kind,
                            default=None if key in paced else default,
                            metavar=type(default).__name__.upper(),
                            help=f"{help_text} (default: {shown})")
    parser.add_argument("--walltime", type=_walltime_arg, default=DEFAULT_WALLTIME,
                        help="wall-clock budget: HH:MM:SS, MM:SS, 90s, 5m, 1h30m (default: 00:01:00)")
    parser.add_argument("--devices", type=_int_arg, default=8, help="device count (default: 8)")
    parser.add_argument("--seed", type=_int_arg, default=None, help="run seed (default: random)")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="more output; repeat for debug-level logs")
    args, unknown = parser.parse_known_args(argv)
    return args, _parse_extras(unknown)


class Session:
    def __init__(self, name: str, argv: list[str] | None = None, out: TextIO | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 keyboard: Keyboard | None = None) -> None:
        self.name, self.job = name, JOBS[name]
        args, self.extras = parse_args(name, self.job, argv)
        self.verbosity = min(2, args.verbose)
        self.seed = args.seed if args.seed is not None else random.randrange(1, 100_000)
        self.rng = random.Random(self.seed)
        self.walltime = max(MIN_WALLTIME, args.walltime)
        self.display = Display(out or sys.stdout, "█" + _PARTIALS)
        self.clock, self.sleep = clock, sleep
        self.t0, self.t0_wall = clock(), datetime.now()
        self.last_pump = self.t0
        self.bar: str | None = None
        self.chatter_credit = 0.0

        self.params = {key: getattr(args, key) for key in self.job["params"]}
        self._pace_params()
        self.curves = {key: {k: v(self.params) if callable(v) else v for k, v in spec.items()}
                       for key, spec in self.job["metrics"].items()}
        self.units = self.params[self.job["units"]]
        inner_key = self.job.get("inner")
        self.inner = self.params[inner_key] if inner_key else 1
        self.ctx = Context(self.rng, {
            **self.params, **self.extras,
            "job": name, "seed": self.seed, "devices": max(1, args.devices), "device": DEVICE,
            "host": platform.node() or "node0", "cores": os.cpu_count() or 8, "pid": os.getpid(),
            "i": 0, "n": self.units, "step": 0, "steps": self.units * self.inner,
            "option": "", "value": "",
        })
        self.ctx["run_id"] = self.ctx["hex8"]
        self._set_progress(0.0)
        self._check_templates()
        sources = self._load_sources()
        # last, so a startup failure above never leaves the terminal in raw-key mode
        self.keyboard = keyboard if keyboard is not None else Keyboard(enabled=self.display.live)
        self.view: Display | Workspace = self.display
        if self.keyboard.active and sources:
            try:
                self.view = Workspace(self.display.out, sources, self.job.get("options", []),
                                      self.display.ascii_only)
            except BaseException:
                self.keyboard.close()
                raise
        self.tick = TYPING_TICK if self.keyboard.active else TICK

    def _load_sources(self) -> list[tuple[str, str]]:
        """The files open in the editor, read now so a missing one fails before the run."""
        try:
            return [(path, (ROOT / path).read_text(encoding="utf-8"))
                    for path in self.job.get("sources", [])]
        except (OSError, UnicodeDecodeError) as error:
            raise SystemExit(f"core/jobs.py: job {self.name!r}, sources: {error}") from None

    def _pace_params(self) -> None:
        """Unless given, unit counts follow the walltime: a train epoch takes ~unit_seconds."""
        main_budget = self.walltime * MAIN_SHARE
        units_key, inner_key = self.job["units"], self.job.get("inner")
        if self.params[units_key] is None:
            self.params[units_key] = round(main_budget / self.job["unit_seconds"])
        if inner_key and self.params[inner_key] is None:
            self.params[inner_key] = round(self.job["unit_seconds"] * self.job["inner_rate"])
        for key in _paced_params(self.job):
            self.params[key] = max(1, self.params[key])  # so config, lines and summary agree

    def _check_templates(self) -> None:
        """Fail before the run starts, not mid-demo, if an edited template is broken."""
        scratch = Context(random.Random(0), dict(self.ctx))
        job = self.job
        templates = [item["bar"] if isinstance(item, dict) else item
                     for item in job["setup"] + job["teardown"]]
        templates += [job["line"], *job["events"], *job["debug"]]
        templates += [template for _, template in job["summary"]] + [OPTION_CHANGE]
        for template in templates:
            try:
                _FORMATTER.vformat(template, (), scratch)
            except Exception as error:
                raise SystemExit(f"core/jobs.py: job {self.name!r}, template {template!r}: "
                                 f"{type(error).__name__}: {error}") from None

    def render(self, template: str) -> str:
        return _FORMATTER.vformat(template, (), self.ctx)

    def run(self) -> int:
        try:
            self._header()
            self._timed(self.job["setup"], self.walltime * SETUP_SHARE)
            self._main(self.walltime * MAIN_SHARE)
            self._timed(self.job["teardown"], self.walltime - self._elapsed())
            self._summary()
            return 0
        except KeyboardInterrupt:
            self._abort()
            return EXIT_INTERRUPTED
        finally:
            self.keyboard.close()
            self.view.close()

    def _elapsed(self) -> float:
        return self.clock() - self.t0

    def _set_progress(self, p: float) -> None:
        for key, spec in self.curves.items():
            self.ctx[key] = curve_value(spec, p, self.rng)
        for key, derive in self.job.get("derived", {}).items():
            self.ctx[key] = derive(self.ctx)

    def _log(self, template: str, level: str = "INFO") -> None:
        message = self.render(template)
        if message.startswith("WARNING "):
            level, message = "WARNING", message[len("WARNING "):]
        stamp = self.t0_wall + timedelta(seconds=self._elapsed())
        self.view.line(f"{stamp:%Y-%m-%d %H:%M:%S}.{stamp.microsecond // 1000:03d} "
                          f"{level:<7} {self.job['logger']}: {message}")

    # --- waiting, and everything that happens while waiting -------------------------------

    def _wait(self, seconds: float) -> None:
        self.sleep(seconds)
        self._pump()

    def _sleep_until(self, deadline: float) -> None:
        while (now := self.clock()) < deadline:
            self._wait(min(self.tick, deadline - now))

    def _pump(self) -> None:
        """One tick: keys into the workspace, -v debug chatter, redraw."""
        now = self.clock()
        dt = max(0.0, now - self.last_pump)
        self.last_pump = now
        for key in self.keyboard.read():
            change = self.view.press(key)
            if change:  # an option was picked in the panel
                option, value = change.split("=", 1)
                self.ctx.update(option=option, value=value)
                self._log(OPTION_CHANGE)
        self.chatter_credit += DEBUG_RATE[self.verbosity] * dt
        while self.chatter_credit >= 1:
            self.chatter_credit -= 1
            self._log(self.rng.choice(self.job["debug"]), level="DEBUG")
        self._refresh()

    def _refresh(self) -> None:
        self.view.show(self.bar)

    # --- phases -------------------------------------------------------------------------

    def _header(self) -> None:
        config = {**self.params, **self.extras, "devices": self.ctx["devices"],
                  "walltime": fmt_duration(self.walltime, always_hours=True)}
        self.view.line(f"{PROGRAM} {VERSION} | job {self.name} | run {self.ctx['run_id']} "
                          f"| seed {self.seed} | pid {self.ctx['pid']}")
        self.view.line("config:")
        for key in sorted(config):
            value = config[key]
            self.view.line(f"  {key:<20} {value:g}" if isinstance(value, float)
                              else f"  {key:<20} {value}")

    def _timed(self, items: list, budget: float) -> None:
        """Spread setup/teardown items over the budget; dict items animate a progress bar."""
        if not items:
            self._sleep_until(self.clock() + max(0.0, budget))
            return
        weights = [(2.0 if isinstance(item, dict) else 1.0) * self.rng.uniform(0.5, 1.5)
                   for item in items]
        scale = max(0.0, budget) / sum(weights)
        deadline = self.clock()
        for item, weight in zip(items, weights):
            deadline += weight * scale
            if isinstance(item, dict):
                self._bar(item, deadline)
            else:
                self._log(item)
                self._sleep_until(deadline)

    def _bar(self, item: dict, deadline: float) -> None:
        label, total, unit = self.render(item["bar"]), item["total"], item.get("unit", "it")
        start = self.clock()
        span = max(deadline - start, 1e-9)
        while (now := self.clock()) < deadline:
            done = int((now - start) / span * total)
            self.bar = bar_text(label, done, total, now - start, unit, self.display.ascii_only)
            self._wait(min(self.tick, deadline - now))
        self.bar = None
        self._refresh()
        self.view.line(bar_text(label, total, total, self.clock() - start, unit,
                                   self.display.ascii_only))

    def _main(self, budget: float) -> None:
        """Main loop over units x inner steps; logs every few seconds at round step counts.
        With inner steps (epochs of steps) the bar shows the current unit, like tqdm does."""
        n, inner = self.units, self.inner
        total = n * inner
        interval = min(LOG_INTERVAL[self.verbosity], budget / MIN_MAIN_LINES)
        log_every = nice_stride(total * interval / budget)
        start = self.clock()
        next_log = min(log_every, total)
        while True:
            now = self.clock()
            p = 1.0 if start + budget - now <= 1e-6 else (now - start) / budget
            done = total if p >= 1.0 else int(p * total)
            while next_log <= done:
                self._step_line(next_log)
                next_log = total + 1 if next_log == total else min(total, next_log + log_every)
            if p >= 1.0:
                break
            unit = min(n, done // inner + 1)
            label = f"{self.job['unit_label']} {unit}/{n}"
            if inner > 1:
                bar_done, bar_total = done - (unit - 1) * inner, inner
                bar_elapsed = now - (start + (unit - 1) * inner * budget / total)
            else:
                bar_done, bar_total, bar_elapsed = done, total, now - start
            self.bar = bar_text(label, bar_done, bar_total, bar_elapsed,
                                self.job["inner_label"], self.display.ascii_only)
            self._wait(min(self.tick, start + budget - now))
        self.bar = None
        self._refresh()

    def _step_line(self, step: int) -> None:
        self.ctx["step"] = step
        self.ctx["i"] = min(self.units, (step - 1) // self.inner + 1)
        self._set_progress(step / (self.units * self.inner))
        self._log(self.job["line"])
        if self.rng.random() < EVENT_CHANCE:
            self._log(self.rng.choice(self.job["events"]))

    def _summary(self) -> None:
        rows = [(label, self.render(template)) for label, template in self.job["summary"]]
        rows.append(("wall time", fmt_duration(self._elapsed(), always_hours=True)))
        label_width = max(len(label) for label, _ in rows) + 4
        rule = "-" * (label_width + max(len(value) for _, value in rows) + 2)
        self.view.line(rule)
        self.view.line(" summary")
        self.view.line(rule)
        for label, value in rows:
            self.view.line(f" {label:<{label_width}}{value}")
        self.view.line(rule)

    def _abort(self) -> None:
        self.bar = None
        self.view.show(None)
        self._log("WARNING received SIGINT, shutting down workers")
        self._log("flushing in-flight state -> checkpoints/interrupt_{run_id}.ckpt")
        self.view.line(f"aborted after {fmt_duration(self._elapsed(), always_hours=True)} "
                          f"(exit {EXIT_INTERRUPTED})")


def strip_timestamps(text: str) -> str:
    return _TIMESTAMP.sub("", text)


def main(name: str) -> None:
    try:
        sys.exit(Session(name).run())
    except KeyboardInterrupt:  # Ctrl+C during startup or a second one during shutdown
        sys.exit(EXIT_INTERRUPTED)
