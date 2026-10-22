"""One open file in the editor pane.

Each file opens partly written, and remembers where its writing left off. In code mode every
letter you press there types the next character of the real file, so mashing keys comes out as
clean code. Mashing anywhere else starts a new line and types a believable comment instead;
going back to where the writing left off picks the real code up again. In real mode (F12)
keys appear exactly as pressed. Either way you can move around and edit like a normal editor.
"""
from core.console import DELETE, DOWN, END, HOME, LEFT, PGDN, PGUP, RIGHT, UP, fit_head

PREFILL_SHARE = 0.45  # of a file's lines already written when it opens
PAGE = 10  # lines moved by PgUp / PgDn
NOTES = [
    "TODO: fuse this with the rope kernel once shapes settle",
    "NOTE: keep this in fp32, bf16 drifts here",
    "FIXME: revisit after the next profiling pass",
    "perf: ~2.1x faster than the unfused path on H100",
    "numerically stable variant, see issue 412",
    "TODO: guard against empty shards",
    "keep the layout contiguous for the all-gather",
    "NOTE: matches the reference implementation to 1e-6",
]
_SLASH_COMMENTS = (".cu", ".c", ".cc", ".cpp", ".h", ".hpp", ".js", ".ts", ".rs", ".go")


class Buffer:
    def __init__(self, name: str, text: str) -> None:
        self.name = name
        self.source = text.replace("\r\n", "\n").replace("\t", "    ")
        self.comment = "//" if name.endswith(_SLASH_COMMENTS) else "#"
        source_lines = self.source.split("\n")
        written = max(1, int(len(source_lines) * PREFILL_SHARE))
        self.lines = source_lines[:written]
        self.row, self.col = written - 1, len(self.lines[-1])
        self.pos = len("\n".join(self.lines))  # next character of the source to type
        self.anchor = self.pos  # where the writing left off, as an index into the text
        self.top = self.left = 0  # scroll offsets
        self.typed: list[tuple[str, bool]] = []  # code typed since the cursor last moved
        self.note = ""  # rest of the comment being typed away from the anchor
        self.notes_used = 0

    # --- typing ---------------------------------------------------------------------------

    def type_code(self) -> None:
        """One keypress in code mode. A newline brings the next line's indentation with it,
        like an editor's auto-indent."""
        if not self.note and self._index() == self.anchor and self.pos < len(self.source):
            size = 1
            if self.source[self.pos] == "\n":
                while self.source.startswith(" ", self.pos + size):
                    size += 1
            chunk = self.source[self.pos:self.pos + size]
            self.pos += size
            self._insert(chunk)
            self.typed.append((chunk, True))
            return
        if not self.note:
            self.col = len(self.lines[self.row])  # notes go on a line of their own
            line = self.lines[self.row]
            indent = line[: len(line) - len(line.lstrip(" "))]
            text = NOTES[self.notes_used % len(NOTES)]
            self.notes_used += 1
            self.note = f"\n{indent}{self.comment} {text}"
        size = len(self.note) - len(self.note[1:].lstrip(" ")) if self.note[0] == "\n" else 1
        chunk, self.note = self.note[:size], self.note[size:]
        self._insert(chunk)
        self.typed.append((chunk, False))

    def type_char(self, ch: str) -> None:
        self._insert(ch)
        self._forget()

    def newline(self) -> None:
        line = self.lines[self.row]
        indent = line[: len(line) - len(line.lstrip(" "))]
        self._insert("\n" + indent)
        self._forget()

    def backspace(self) -> None:
        """Deletes the character before the cursor; right after code typing it un-types, so
        pressing on again retypes the same code."""
        if not self.typed:
            self._delete_back()
            return
        chunk, from_source = self.typed.pop()
        for _ in chunk:
            self._delete_back()
        if from_source:
            self.pos -= len(chunk)
        else:
            self.note = chunk + self.note

    def delete(self) -> None:
        line = self.lines[self.row]
        if self.col < len(line):
            self.lines[self.row] = line[: self.col] + line[self.col + 1:]
        elif self.row + 1 < len(self.lines):
            self.lines[self.row] = line + self.lines.pop(self.row + 1)
        else:
            return
        if self._index() < self.anchor:
            self.anchor -= 1
        self._forget()

    def _forget(self) -> None:
        self.typed.clear()
        self.note = ""

    def _index(self) -> int:
        return sum(len(line) + 1 for line in self.lines[: self.row]) + self.col

    def _insert(self, text: str) -> None:
        if self._index() <= self.anchor:
            self.anchor += len(text)
        line = self.lines[self.row]
        head, tail = line[: self.col], line[self.col:]
        parts = (head + text).split("\n")
        self.lines[self.row:self.row + 1] = parts[:-1] + [parts[-1] + tail]
        self.row += len(parts) - 1
        self.col = len(parts[-1])

    def _delete_back(self) -> None:
        if self.col > 0:
            line = self.lines[self.row]
            self.lines[self.row] = line[: self.col - 1] + line[self.col:]
            self.col -= 1
        elif self.row > 0:
            previous = self.lines[self.row - 1]
            self.lines[self.row - 1] = previous + self.lines.pop(self.row)
            self.row -= 1
            self.col = len(previous)
        else:
            return
        if self._index() < self.anchor:
            self.anchor -= 1

    # --- moving ---------------------------------------------------------------------------

    def move(self, key: str) -> None:
        if key == LEFT:
            if self.col > 0:
                self.col -= 1
            elif self.row > 0:
                self.row -= 1
                self.col = len(self.lines[self.row])
        elif key == RIGHT:
            if self.col < len(self.lines[self.row]):
                self.col += 1
            elif self.row + 1 < len(self.lines):
                self.row, self.col = self.row + 1, 0
        elif key in (UP, DOWN, PGUP, PGDN):
            step = {UP: -1, DOWN: 1, PGUP: -PAGE, PGDN: PAGE}[key]
            self.row = min(len(self.lines) - 1, max(0, self.row + step))
            self.col = min(self.col, len(self.lines[self.row]))
        elif key == HOME:
            line = self.lines[self.row]
            indent = len(line) - len(line.lstrip(" "))
            self.col = 0 if self.col == indent else indent  # like most editors
        elif key == END:
            self.col = len(self.lines[self.row])
        self._forget()

    def press(self, key: str, real: bool) -> None:
        if key in (LEFT, RIGHT, UP, DOWN, PGUP, PGDN, HOME, END):
            self.move(key)
        elif key == "\b":
            self.backspace()
        elif key == DELETE:
            self.delete()
        elif key == "\n":
            self.newline()
        elif len(key) == 1 and key.isprintable():
            if real:
                self.type_char(key)
            else:
                self.type_code()

    # --- drawing --------------------------------------------------------------------------

    def view(self, height: int, width: int, rule: str) -> tuple[list[str], tuple[int, int]]:
        """Visible lines with line numbers, scrolled to keep the cursor in sight, plus the
        cursor's (row, column) inside the view."""
        if self.row < self.top:
            self.top = self.row
        elif self.row >= self.top + height:
            self.top = self.row - height + 1
        gutter = len(str(len(self.lines))) + 3
        text_width = max(1, width - gutter)
        if self.col < self.left:
            self.left = max(0, self.col - text_width // 2)
        elif self.col >= self.left + text_width:
            self.left = self.col - text_width + 1
        rows = []
        for number in range(self.top, min(len(self.lines), self.top + height)):
            text = fit_head(self.lines[number][self.left:], text_width)
            rows.append(f"{number + 1:>{gutter - 3}} {rule} {text}")
        return rows, (self.row - self.top, gutter + self.col - self.left)
