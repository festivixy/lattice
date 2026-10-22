import contextlib
import io
import os
import unittest
import unittest.mock

from core import console, engine
from core.jobs import JOBS


class FakeClock:
    """Stands in for time.monotonic/time.sleep so a 5-minute run finishes instantly."""

    def __init__(self, interrupt_after=None):
        self.now = 0.0
        self.sleeps = 0
        self.interrupt_after = interrupt_after

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps += 1
        if self.interrupt_after is not None and self.sleeps >= self.interrupt_after:
            raise KeyboardInterrupt
        self.now += max(0.0, seconds)


class FakeKeyboard:
    """Replays scripted typing: (time, text) pairs, released as the fake clock passes them."""

    def __init__(self, clock, script):
        self.clock = clock
        self.script = sorted(script)
        self.active = True
        self.closed = False

    def read(self):
        keys = []
        while self.script and self.script[0][0] <= self.clock.now:
            keys.extend(self.script.pop(0)[1])
        return keys

    def close(self):
        self.closed = True


def run_job(name, argv, clock=None, typing=None):
    clock = clock or FakeClock()
    out = io.StringIO()
    keyboard = FakeKeyboard(clock, typing) if typing is not None else None
    session = engine.Session(name, argv, out=out, clock=clock, sleep=clock.sleep,
                             keyboard=keyboard)
    code = session.run()
    return session, code, out.getvalue(), clock


def all_templates(job):
    for item in job["setup"] + job["teardown"]:
        yield item["bar"] if isinstance(item, dict) else item
    yield job["line"]
    yield from job["events"]
    yield from job["debug"]
    yield from (template for _, template in job["summary"])


class ParseWalltimeTest(unittest.TestCase):
    def test_accepts_clock_and_unit_formats(self):
        cases = {"00:05:00": 300, "5:00": 300, "1:02:03": 3723, "90": 90, "90s": 90,
                 "5m": 300, "1h30m": 5400, "2.5m": 150}
        for text, seconds in cases.items():
            with self.subTest(text=text):
                self.assertEqual(engine.parse_walltime(text), seconds)

    def test_rejects_garbage(self):
        for text in ("", "abc", "5x", "1:2:3:4", "5m garbage", "-5"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                engine.parse_walltime(text)

    def test_bad_walltime_flag_exits_like_argparse(self):
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
            engine.Session("train", ["--walltime", "soon"])
        self.assertIn("invalid walltime", err.getvalue())


class SessionTest(unittest.TestCase):
    def test_every_job_completes_and_every_template_renders(self):
        for name, job in JOBS.items():
            with self.subTest(job=name):
                session, code, text, _ = run_job(name, ["--walltime", "20", "-vv", "--seed", "7"])
                self.assertEqual(code, 0)
                self.assertIn(" summary", text)
                for template in all_templates(job):
                    session.render(template)

    def test_run_ends_on_walltime(self):
        for walltime in (5, 37, 300):
            with self.subTest(walltime=walltime):
                *_, clock = run_job("solve", ["--walltime", str(walltime)])
                self.assertGreaterEqual(clock.now, walltime - 1e-6)
                self.assertLessEqual(clock.now, walltime + engine.TICK)

    def test_walltime_below_minimum_is_raised_to_minimum(self):
        *_, clock = run_job("train", ["--walltime", "1"])
        self.assertAlmostEqual(clock.now, engine.MIN_WALLTIME, delta=engine.TICK)

    def test_unit_count_flag_drives_main_loop(self):
        _, _, text, _ = run_job("train", ["--epochs", "7", "--walltime", "10"])
        self.assertRegex(text, r"epoch +1/7 \| step")
        self.assertRegex(text, r"epoch +7/7 \| step")

    def test_default_unit_counts_follow_walltime_at_a_real_pace(self):
        for walltime, epochs in ((60, 2), (600, 19)):
            with self.subTest(walltime=walltime):
                session, *_ = run_job("train", ["--walltime", str(walltime)])
                self.assertEqual(session.params["epochs"], epochs)
                steps_per_second = session.units * session.inner / (walltime * engine.MAIN_SHARE)
                self.assertLess(steps_per_second, 10)

    def test_main_lines_arrive_every_few_seconds(self):
        _, _, text, _ = run_job("solve", ["--walltime", "300"])
        lines = text.count("lattice.krylov: iter ")
        budget = 300 * engine.MAIN_SHARE
        self.assertGreater(lines, budget / (2 * engine.LOG_INTERVAL[0]))
        self.assertLess(lines, 2 * budget / engine.LOG_INTERVAL[0])

    def test_short_runs_still_log_several_main_lines(self):
        _, _, text, _ = run_job("simulate", ["--walltime", "5", "--steps", "5000"])
        self.assertGreaterEqual(text.count("lattice.nbody: step "), engine.MIN_MAIN_LINES // 2)

    def test_debug_volume_follows_verbosity(self):
        counts = [run_job("solve", ["--walltime", "30", "--seed", "3", *flag])[2].count(" DEBUG ")
                  for flag in ([], ["-v"], ["-vv"])]
        self.assertEqual(counts[0], 0)
        self.assertLess(counts[1], counts[2])

    def test_int_flags_accept_scientific_notation(self):
        session, *_ = run_job("simulate", ["--particles", "1e9", "--walltime", "5"])
        self.assertEqual(session.params["particles"], 10**9)

    def test_unknown_flags_show_in_config(self):
        _, code, text, _ = run_job("search", ["--walltime", "5", "--gpu-arch", "sm90",
                                              "--tag=demo", "--dry"])
        self.assertEqual(code, 0)
        self.assertRegex(text, r"gpu_arch\s+sm90")
        self.assertRegex(text, r"tag\s+demo")
        self.assertRegex(text, r"dry\s+true")

    def test_same_seed_gives_same_run(self):
        runs = [run_job("simulate", ["--seed", "42", "--walltime", "15", "-v"])[2]
                for _ in range(2)]
        self.assertEqual(engine.strip_timestamps(runs[0]), engine.strip_timestamps(runs[1]))

    def test_ctrl_c_aborts_cleanly(self):
        _, code, text, _ = run_job("train", ["--walltime", "60"],
                                   clock=FakeClock(interrupt_after=40))
        self.assertEqual(code, engine.EXIT_INTERRUPTED)
        self.assertIn("received SIGINT", text)
        self.assertNotIn(" summary", text)


class EdgeCaseTest(unittest.TestCase):
    def test_broken_template_fails_before_run_with_clear_message(self):
        broken = {**JOBS["train"], "events": ["checkpoint {los:.4f}"]}
        with unittest.mock.patch.dict(JOBS, {"train": broken}):
            with self.assertRaises(SystemExit) as caught:
                engine.Session("train", ["--walltime", "5"], out=io.StringIO())
        self.assertIn("{los:.4f}", str(caught.exception.code))

    def test_empty_setup_still_ends_on_walltime(self):
        bare = {**JOBS["train"], "setup": [], "teardown": []}
        with unittest.mock.patch.dict(JOBS, {"train": bare}):
            _, code, _, clock = run_job("train", ["--walltime", "20"])
        self.assertEqual(code, 0)
        self.assertAlmostEqual(clock.now, 20, delta=engine.TICK)

    def test_degenerate_numbers_do_not_crash(self):
        for name, argv in [("solve", ["--tol", "0"]), ("train", ["--epochs", "0"]),
                           ("train", ["--steps-per-epoch", "-3"]), ("search", ["--devices", "0"])]:
            with self.subTest(argv=argv):
                session, code, text, _ = run_job(name, argv + ["--walltime", "5"])
                self.assertEqual(code, 0)
        self.assertIn("world_size=1", run_job("train", ["--devices", "0", "--walltime", "5"])[2])

    def test_infinite_int_is_an_argparse_error(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            engine.Session("train", ["--epochs", "inf"])
        self.assertEqual(caught.exception.code, 2)


class LiveDisplayTest(unittest.TestCase):
    def test_log_lines_print_above_the_live_region(self):
        class FakeTerminal(io.StringIO):
            def isatty(self):
                return True

        out = FakeTerminal()
        display = console.Display(out, "#")
        display.set_live(["bar", "lattice> hi"])
        display.line("log")
        wipe = "\r\x1b[2K\x1b[1A\x1b[2K"
        self.assertEqual(out.getvalue(), "bar\nlattice> hi" + wipe + "log\nbar\nlattice> hi")
        display.clear_live()
        self.assertTrue(out.getvalue().endswith(wipe))

    def test_unchanged_region_is_not_redrawn(self):
        class FakeTerminal(io.StringIO):
            def isatty(self):
                return True

        out = FakeTerminal()
        display = console.Display(out, "#")
        display.set_live(["same"])
        before = out.getvalue()
        display.set_live(["same"])
        self.assertEqual(out.getvalue(), before)

    def test_region_never_taller_than_the_window(self):
        class FakeTerminal(io.StringIO):
            def isatty(self):
                return True

        out = FakeTerminal()
        display = console.Display(out, "#")
        with unittest.mock.patch("shutil.get_terminal_size",
                                 return_value=os.terminal_size((80, 4))):
            display.set_live([f"line {n}" for n in range(10)])
        self.assertEqual(display.region, ["line 7", "line 8", "line 9"])

    def test_posix_keys_survive_escape_codes_split_across_reads(self):
        keys, leftover = console.split_posix_keys("ab\x1b[2")
        self.assertEqual((keys, leftover), (["a", "b"], "\x1b[2"))
        keys, leftover = console.split_posix_keys(leftover + "4~c\x1b[Ad")
        self.assertEqual((keys, leftover), ([console.F12, "c", console.UP, "d"], ""))

    def test_wide_characters_are_fitted_by_screen_columns(self):
        self.assertEqual(console.fit_head("训练模型abc", 5), "训练")
        self.assertEqual(console.fit_head("short", 40), "short")


class HelpersTest(unittest.TestCase):
    def test_nice_stride_rounds_to_1_2_5(self):
        cases = {0.3: 1, 1: 1, 1.8: 2, 4: 5, 6.25: 5, 13: 10, 180: 200, 3700: 5000}
        for raw, stride in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(engine.nice_stride(raw), stride)

    def test_bar_text_fills_proportionally(self):
        empty = engine.bar_text("x", 0, 100, 0.0, "it")
        half = engine.bar_text("x", 50, 100, 5.0, "it")
        full = engine.bar_text("x", 100, 100, 10.0, "it", ascii_only=True)
        self.assertIn("  0%", empty)
        self.assertIn("█" * (engine.BAR_WIDTH // 2), half)
        self.assertIn("10.00it/s", half)
        self.assertIn("100%|" + "#" * engine.BAR_WIDTH + "|", full)

    def test_curves_hit_their_endpoints_without_noise(self):
        rng = engine.random.Random(0)
        for kind in ("log", "cosine", "linear"):
            spec = {"kind": kind, "start": 1.0, "end": 1e-6}
            with self.subTest(kind=kind):
                self.assertAlmostEqual(engine.curve_value(spec, 0.0, rng), 1.0)
                self.assertAlmostEqual(engine.curve_value(spec, 1.0, rng), 1e-6)

    def test_curve_clamps(self):
        spec = {"kind": "steady", "start": 0.99, "noise": 5.0, "max": 1.0, "min": 0.0}
        rng = engine.random.Random(1)
        values = [engine.curve_value(spec, 0.5, rng) for _ in range(200)]
        self.assertTrue(all(0.0 <= v <= 1.0 for v in values))


if __name__ == "__main__":
    unittest.main()
