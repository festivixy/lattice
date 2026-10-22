import io
import os
import unittest
import unittest.mock

from core import console, engine
from core.console import DELETE, DOWN, END, F2, F3, F12, HOME, LEFT, RIGHT, UP, cell_width
from core.editor import Buffer
from core.jobs import JOBS
from core.workspace import Workspace
from tests.test_engine import FakeClock, FakeKeyboard, run_job

SOURCE = "def f(x):\n    y = x + 1\n    return y\n\n\ndef g():\n    pass\n"


class FakeTerminal(io.StringIO):
    def isatty(self):
        return True


def text_of(buffer):
    return "\n".join(buffer.lines)


class BufferTest(unittest.TestCase):
    def test_opens_partly_written_with_cursor_at_the_end(self):
        buffer = Buffer("src/f.py", SOURCE)
        self.assertEqual(buffer.lines, ["def f(x):", "    y = x + 1", "    return y"])
        self.assertEqual((buffer.row, buffer.col), (2, len("    return y")))

    def test_mashing_types_the_rest_of_the_file_exactly(self):
        buffer = Buffer("src/f.py", SOURCE)
        while buffer.pos < len(SOURCE):
            buffer.press("q", real=False)
        self.assertEqual(text_of(buffer), SOURCE)

    def test_one_key_brings_a_new_line_with_its_indentation(self):
        buffer = Buffer("src/f.py", "a:\n    b\n    c\n")
        buffer.lines, buffer.row, buffer.col, buffer.pos = ["a:"], 0, 2, 2
        buffer.press("x", real=False)
        self.assertEqual(buffer.lines, ["a:", "    "])

    def test_backspace_after_typing_untypes_and_retyping_restores(self):
        buffer = Buffer("src/f.py", SOURCE)
        for key in "abcd":
            buffer.press(key, real=False)
        typed = text_of(buffer)
        buffer.press("\b", real=False)
        buffer.press("k", real=False)
        self.assertEqual(text_of(buffer), typed)

    def test_arrows_move_and_editing_happens_at_the_cursor(self):
        buffer = Buffer("src/f.py", SOURCE)
        buffer.press(UP, real=False)
        buffer.press(HOME, real=False)  # to the indentation
        self.assertEqual((buffer.row, buffer.col), (1, 4))
        buffer.press(DELETE, real=False)  # the "y"
        self.assertEqual(buffer.lines[1], "     = x + 1")
        buffer.press("\b", real=False)  # cursor moved, so this deletes a real character
        self.assertEqual(buffer.lines[1], "    = x + 1")
        buffer.press("z", real=True)
        self.assertEqual(buffer.lines[1], "   z = x + 1")
        buffer.press(END, real=False)
        buffer.press(RIGHT, real=False)
        self.assertEqual((buffer.row, buffer.col), (2, 0))
        buffer.press(LEFT, real=False)
        self.assertEqual((buffer.row, buffer.col), (1, len("   z = x + 1")))

    def test_mashing_away_from_the_write_point_types_a_comment_line(self):
        buffer = Buffer("src/f.py", SOURCE)
        buffer.press(UP, real=False)
        buffer.press(HOME, real=False)  # mid-line: the note still goes on a line of its own
        for _ in range(12):
            buffer.press("q", real=False)
        self.assertEqual(buffer.lines[1], "    y = x + 1")
        self.assertEqual(buffer.lines[2], "    # TODO: fus")
        self.assertEqual(buffer.lines[3], "    return y")

    def test_one_step_left_of_the_write_point_still_types_a_whole_comment(self):
        buffer = Buffer("src/f.py", SOURCE)
        buffer.press(LEFT, real=False)
        for _ in range(8):
            buffer.press("q", real=False)
        self.assertEqual(buffer.lines[-1], "    # TODO:")

    def test_returning_to_the_write_point_resumes_the_real_code(self):
        buffer = Buffer("src/f.py", SOURCE)
        buffer.press(UP, real=False)
        for _ in range(5):
            buffer.press("q", real=False)  # a note above the write point
        for _ in range(2):
            buffer.press(DOWN, real=False)
        buffer.press(END, real=False)  # back where the writing left off
        while buffer.pos < len(SOURCE):
            buffer.press("q", real=False)
        without_note = [line for line in buffer.lines if "# TO" not in line]
        self.assertEqual("\n".join(without_note), SOURCE)

    def test_finished_file_keeps_going_with_comments_not_a_repeat(self):
        buffer = Buffer("src/k.cu", "int a;\nint b;\n")
        for _ in range(30):
            buffer.press("q", real=False)
        self.assertTrue(text_of(buffer).startswith("int a;\nint b;\n"))
        self.assertIn("// TODO", text_of(buffer))
        self.assertEqual(text_of(buffer).count("int a;"), 1)

    def test_backspace_inside_a_comment_retypes_the_same_comment(self):
        buffer = Buffer("src/f.py", SOURCE)
        buffer.press(UP, real=False)
        for _ in range(6):
            buffer.press("q", real=False)
        typed = text_of(buffer)
        buffer.press("\b", real=False)
        buffer.press("q", real=False)
        self.assertEqual(text_of(buffer), typed)

    def test_enter_keeps_indentation_and_delete_joins_lines(self):
        buffer = Buffer("src/f.py", SOURCE)
        buffer.press("\n", real=False)
        self.assertEqual(buffer.lines[-1], "    ")
        buffer.press(UP, real=False)
        buffer.press(END, real=False)
        buffer.press(DELETE, real=False)
        self.assertEqual(buffer.lines[-1], "    return y    ")

    def test_view_scrolls_to_keep_the_cursor_visible(self):
        buffer = Buffer("src/long.py", "\n".join(f"line {n}" for n in range(100)))
        rows, (row, col) = buffer.view(10, 40, "|")
        self.assertEqual(len(rows), 10)
        self.assertEqual(row, 9)
        self.assertTrue(rows[-1].endswith(f"line {buffer.row}"))
        self.assertEqual(rows[-1][col - len(f"line {buffer.row}"):], f"line {buffer.row}")
        for _ in range(3):
            buffer.press("<PGUP>", real=False)
        rows, (row, _) = buffer.view(10, 40, "|")
        self.assertEqual(row, 0)


class WorkspaceTest(unittest.TestCase):
    FILES = [("src/a.py", SOURCE), ("src/b.py", SOURCE), ("configs/c.yaml", "k: v\n")]
    OPTIONS = [("precision", ["bf16", "fp8"]), ("optimizer", ["adamw", "lion"])]

    def setUp(self):
        patcher = unittest.mock.patch("shutil.get_terminal_size",
                                      return_value=os.terminal_size((100, 24)))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.out = FakeTerminal()
        self.ws = Workspace(self.out, self.FILES, self.OPTIONS, ascii_only=False)

    def test_takes_over_the_screen_and_gives_it_back_with_the_log(self):
        self.assertTrue(self.out.getvalue().startswith("\x1b[?1049h"))
        self.ws.line("first log line")
        self.ws.close()
        self.assertTrue(self.out.getvalue().endswith("\x1b[?1049lfirst log line\n"))

    def test_every_row_fills_the_screen_width_exactly(self):
        self.ws.line("x" * 300)  # wraps in the log pane
        self.ws.show("epoch 1/2:  50%|bar| 5/10")
        self.assertEqual(len(self.ws.frame), 24)
        for row in self.ws.frame:
            self.assertEqual(sum(cell_width(ch) for ch in row), 100)
        joined = "\n".join(self.ws.frame)
        for expected in ("[a.py]", " b.py ", "FILES", "OPTIONS", "precision", "epoch 1/2"):
            self.assertIn(expected, joined)

    def test_panel_picks_options_and_files(self):
        self.ws.press(F2)
        for _ in range(3):  # past the three files
            self.ws.press(DOWN)
        self.assertEqual(self.ws.press("\n"), "precision=fp8")
        self.assertEqual(self.ws.press("\n"), "precision=bf16")
        self.ws.press(UP)  # back to the last file
        self.assertIsNone(self.ws.press("\n"))
        self.assertEqual(self.ws.active, 2)
        self.assertFalse(self.ws.in_panel)

    def test_letters_in_the_panel_do_not_edit(self):
        before = list(self.ws.buffers[0].lines)
        self.ws.press(F2)
        self.ws.press("q")
        self.assertEqual(self.ws.buffers[0].lines, before)

    def test_f3_cycles_files_and_f12_switches_to_real_typing(self):
        self.ws.press(F3)
        self.assertEqual(self.ws.active, 1)
        self.ws.press(F12)
        self.ws.press("Z")
        self.assertTrue(self.ws.buffers[1].lines[-1].endswith("Z"))

    def test_cursor_is_placed_in_the_editor_and_hidden_in_the_panel(self):
        self.ws.render()
        self.assertTrue(self.out.getvalue().endswith("\x1b[?25h"))
        self.ws.press(F2)
        self.ws.render()
        self.assertFalse(self.out.getvalue().endswith("\x1b[?25h"))


class SessionWorkspaceTest(unittest.TestCase):
    def setUp(self):
        patcher = unittest.mock.patch("shutil.get_terminal_size",
                                      return_value=os.terminal_size((120, 30)))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_keys_reach_the_editor_and_the_run_ends_with_the_plain_log(self):
        session, code, text, _ = run_job("train", ["--walltime", "10"], typing=[(3, "abc")])
        self.assertEqual(code, 0)
        self.assertIsInstance(session.view, Workspace)
        self.assertEqual(len(session.view.buffers[0].typed), 3)
        self.assertTrue(session.keyboard.closed)
        after_screen = text[text.rindex("\x1b[?1049l"):]
        self.assertIn(" summary", after_screen)
        self.assertIn("initializing process group", after_screen)

    def test_picking_an_option_logs_a_quiet_config_line(self):
        files = len(JOBS["train"]["sources"])
        keys = [F2] + [DOWN] * files + ["\n"]
        _, _, text, _ = run_job("train", ["--walltime", "10"], typing=[(3, keys)])
        self.assertIn("config change queued: precision=fp8 (applies at next checkpoint)", text)

    def test_job_without_files_runs_plainly_even_in_a_terminal(self):
        bare = {**JOBS["train"], "sources": []}
        with unittest.mock.patch.dict(JOBS, {"train": bare}):
            session, code, _, _ = run_job("train", ["--walltime", "5"], typing=[(2, "abc")])
        self.assertEqual(code, 0)
        self.assertIsInstance(session.view, console.Display)

    def test_no_workspace_without_a_terminal(self):
        session, *_ = run_job("train", ["--walltime", "5"])
        self.assertIsInstance(session.view, console.Display)

    def test_every_job_has_files_and_options(self):
        for name, job in JOBS.items():
            with self.subTest(job=name):
                self.assertGreaterEqual(len(job["sources"]), 3)
                self.assertGreaterEqual(len(job["options"]), 3)
                for path in job["sources"]:
                    self.assertTrue((engine.ROOT / path).is_file(), path)


if __name__ == "__main__":
    unittest.main()
