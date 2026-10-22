"""Full-screen view: the job's log on the left, an editor and a FILES / OPTIONS panel on the
right. It takes over the terminal (like vim or htop) while the job runs; on close the screen
is restored and the whole log is printed, as if the job had been run plainly.

Keys: in the editor, letters type code (or exactly what you press after F12), arrows / Home /
End / PgUp / PgDn move, Backspace / Delete delete, Enter starts an indented line, F3 opens the
next file. F2 moves to the panel: arrows pick, Enter opens a file or changes an option, Esc or
F2 goes back to the editor.
"""
import shutil
from typing import TextIO

from core.console import DOWN, ESC, F2, F3, F12, UP, enable_vt_codes, pad
from core.editor import Buffer

LEFT_SHARE = 0.55  # of the screen width given to the job log
MIN_WIDTH, MIN_HEIGHT = 60, 12
MAX_LOG_LINES = 5000

_ENTER_SCREEN = "\x1b[?1049h\x1b[2J"
_LEAVE_SCREEN = "\x1b[?25h\x1b[?1049l"
_HIDE_CURSOR, _SHOW_CURSOR = "\x1b[?25l", "\x1b[?25h"


class Workspace:
    def __init__(self, out: TextIO, files: list[tuple[str, str]],
                 options: list[tuple[str, list[str]]], ascii_only: bool) -> None:
        self.out = out
        self.buffers = [Buffer(name, text) for name, text in files]
        self.active = 0
        self.options = [[name, list(values), 0] for name, values in options if values]
        self.items = [("file", i) for i in range(len(self.buffers))] + \
                     [("option", i) for i in range(len(self.options))]
        self.selected = 0
        self.in_panel = False
        self.real = False
        self.rule, self.vbar = ("-", "|") if ascii_only else ("─", "│")
        self.log: list[str] = []
        self.bar: str | None = None
        self.frame: list[str] = []
        self.size = (0, 0)
        enable_vt_codes()
        self.out.write(_ENTER_SCREEN)
        self.out.flush()

    # --- what the engine calls --------------------------------------------------------------

    def line(self, text: str) -> None:
        self.log.append(text)
        del self.log[:-MAX_LOG_LINES]
        self.render()

    def show(self, bar: str | None) -> None:
        self.bar = bar
        self.render()

    def press(self, key: str) -> str | None:
        """Handle one key. Returns "name=value" when an option was changed."""
        if key == F2 or (key == ESC and self.in_panel):
            self.in_panel = not self.in_panel
            return None
        if key == F12:
            self.real = not self.real
        elif key == F3:
            self.active = (self.active + 1) % len(self.buffers)
        elif self.in_panel:
            return self._panel_key(key)
        else:
            self.buffers[self.active].press(key, self.real)
        return None

    def close(self) -> None:
        """Give the terminal back and print the whole log, like a plain run would leave."""
        self.out.write(_LEAVE_SCREEN + "".join(line + "\n" for line in self.log))
        self.out.flush()

    # --- panel ----------------------------------------------------------------------------

    def _panel_key(self, key: str) -> str | None:
        if key in (UP, DOWN):
            self.selected = (self.selected + (1 if key == DOWN else -1)) % len(self.items)
        elif key == "\n":
            kind, index = self.items[self.selected]
            if kind == "file":
                self.active, self.in_panel = index, False
            else:
                option = self.options[index]
                option[2] = (option[2] + 1) % len(option[1])
                return f"{option[0]}={option[1][option[2]]}"
        return None

    def _panel_rows(self, width: int) -> list[str]:
        rows = [self._title("FILES", width)]
        for position, (kind, index) in enumerate(self.items):
            if kind == "option" and self.items[position - 1][0] == "file":
                rows.append(self._title("OPTIONS", width))
            marker = ">" if self.in_panel and position == self.selected else " "
            if kind == "file":
                name = self.buffers[index].name
                current = "*" if index == self.active else " "
                rows.append(f"{marker}{current} {name}")
            else:
                name, values, choice = self.options[index]
                rows.append(f"{marker}  {name:<18} {values[choice]}")
        return rows

    def _title(self, text: str, width: int) -> str:
        return f"{self.rule * 2} {text} " + self.rule * max(0, width - len(text) - 4)

    # --- drawing --------------------------------------------------------------------------

    def render(self) -> None:
        size = shutil.get_terminal_size((120, 30))
        width, height = max(MIN_WIDTH, size.columns), max(MIN_HEIGHT, size.lines)
        if (width, height) != self.size:  # resized: redraw everything
            self.size, self.frame = (width, height), []
            self.out.write("\x1b[2J")
        left = int(width * LEFT_SHARE)
        right = width - left - 1
        log_rows = self._log_rows(left, height)
        side, cursor = self._right_rows(right, height)
        frame = [pad(log_rows[i], left) + self.vbar + pad(side[i], right) for i in range(height)]
        changed = "".join(f"\x1b[{i + 1};1H{row}" for i, row in enumerate(frame)
                          if i >= len(self.frame) or self.frame[i] != row)
        self.frame = frame
        if cursor:
            place = f"\x1b[{cursor[0] + 1};{left + 2 + cursor[1]}H" + _SHOW_CURSOR
        else:
            place = ""
        self.out.write(_HIDE_CURSOR + changed + place)
        self.out.flush()

    def _log_rows(self, width: int, height: int) -> list[str]:
        """The newest log lines, wrapped to the pane, with the progress bar at the bottom."""
        room = height - (1 if self.bar else 0)
        rows: list[str] = []
        for text in reversed(self.log):
            wrapped = [text[i:i + width] for i in range(0, max(1, len(text)), width)]
            rows[:0] = wrapped
            if len(rows) >= room:
                break
        rows = rows[-room:] if room else []
        rows += [""] * (room - len(rows))
        return rows + ([self.bar] if self.bar else [])

    def _right_rows(self, width: int, height: int) -> tuple[list[str], tuple[int, int] | None]:
        buffer = self.buffers[self.active]
        tabs = " ".join(f"[{b.name.rsplit('/', 1)[-1]}]" if i == self.active
                        else f" {b.name.rsplit('/', 1)[-1]} " for i, b in enumerate(self.buffers))
        panel = self._panel_rows(width)
        editor_height = max(3, height - 2 - len(panel))
        view, (cursor_row, cursor_col) = buffer.view(editor_height, width, self.vbar)
        view += [""] * (editor_height - len(view))
        where = f" {buffer.name}  Ln {buffer.row + 1}, Col {buffer.col + 1} "
        status = f"{self.rule * 2}{where}" + self.rule * max(0, width - len(where) - 2)
        rows = ([tabs] + view + [status] + panel)[:height]
        rows += [""] * (height - len(rows))
        cursor = None if self.in_panel else (1 + cursor_row, min(cursor_col, width - 1))
        return rows, cursor
