"""Terminal I/O for the runner: keyboard input read without blocking, and the plain line
display used when there is no interactive terminal (or for piped output)."""
import codecs
import os
import re
import shutil
import sys
import unicodedata
from typing import TextIO

# Special keys come back from Keyboard.read() as these tokens; plain characters as themselves,
# with Enter as "\n" and Backspace as "\b".
UP, DOWN, LEFT, RIGHT = "<UP>", "<DOWN>", "<LEFT>", "<RIGHT>"
HOME, END, PGUP, PGDN, DELETE = "<HOME>", "<END>", "<PGUP>", "<PGDN>", "<DELETE>"
ESC, F2, F3, F12 = "<ESC>", "<F2>", "<F3>", "<F12>"

_KEY_ALIASES = {"\r": "\n", "\x7f": "\b", "\x08": "\b", "\t": " ", "\x1b": ESC}
# msvcrt: special keys arrive as "\x00" or "\xe0" followed by one of these codes
_WINDOWS_KEYS = {"H": UP, "P": DOWN, "K": LEFT, "M": RIGHT, "G": HOME, "O": END, "I": PGUP,
                 "Q": PGDN, "S": DELETE, "<": F2, "=": F3, "\x86": F12}
# POSIX terminals send escape sequences
_POSIX_KEYS = {"[A": UP, "[B": DOWN, "[C": RIGHT, "[D": LEFT, "[H": HOME, "[F": END,
               "OH": HOME, "OF": END, "[1~": HOME, "[4~": END, "[7~": HOME, "[8~": END,
               "[5~": PGUP, "[6~": PGDN, "[3~": DELETE, "OQ": F2, "[12~": F2, "OR": F3,
               "[13~": F3, "[24~": F12}
_ESCAPE_SEQUENCE = re.compile(r"\x1b(\[[0-9;?]*[A-Za-z~]|O.)")
_UNFINISHED_ESCAPE = re.compile(r"\x1b(\[[0-9;?]*|O)$")


class Display:
    """Plain output: log lines scroll, and the progress bar is redrawn in place under them."""

    def __init__(self, out: TextIO, block_chars: str) -> None:
        self.out = out
        self.live = hasattr(out, "isatty") and out.isatty()
        self.ascii_only = not _can_encode(out, block_chars)
        self.region: list[str] = []
        if self.live:
            enable_vt_codes()

    def line(self, text: str) -> None:
        self.out.write(self._wipe() + text + "\n" + self._draw())
        self.out.flush()

    def show(self, bar: str | None) -> None:
        self.set_live([bar] if bar else [])

    def press(self, key: str) -> str | None:
        return None

    def close(self) -> None:
        self.clear_live()

    def set_live(self, lines: list[str]) -> None:
        if not self.live:
            return
        size = shutil.get_terminal_size((100, 24))  # follow resizes
        width = max(10, size.columns)
        # never taller than the window: lines scrolled off the top could not be wiped again
        fitted = [fit_head(line, width - 1) for line in lines][-max(1, size.lines - 1):]
        if fitted == self.region:
            return  # nothing changed; skip the redraw to avoid flicker
        wipe = self._wipe()
        self.region = fitted
        self.out.write(wipe + self._draw())
        self.out.flush()

    def clear_live(self) -> None:
        if self.region:
            self.out.write(self._wipe())
            self.region = []
            self.out.flush()

    def _wipe(self) -> str:
        """Erase the live region, leaving the cursor where its first line began."""
        if not self.region:
            return ""
        return "\r\x1b[2K" + "\x1b[1A\x1b[2K" * (len(self.region) - 1)

    def _draw(self) -> str:
        return "\n".join(self.region)


class Keyboard:
    """Keys typed while a job runs, read without blocking and without echo."""

    def __init__(self, enabled: bool = True) -> None:
        self.active = enabled and sys.stdin is not None and sys.stdin.isatty()
        self._saved_mode = None
        self._pending = ""  # POSIX: start of an escape sequence still arriving
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
        if self.active and os.name != "nt":
            import termios
            import tty
            self._saved_mode = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())  # no echo, no line buffering; Ctrl+C still works

    def read(self) -> list[str]:
        if not self.active:
            return []
        if os.name == "nt":
            keys = _read_windows()
        else:
            text = self._pending + self._decoder.decode(_read_posix_bytes())
            keys, self._pending = split_posix_keys(text)
        if "\x03" in keys:
            raise KeyboardInterrupt
        return [_KEY_ALIASES.get(key, key) for key in keys]

    def close(self) -> None:
        """Restore the terminal and drop unread keys so they don't spill onto the prompt."""
        if not self.active:
            return
        if os.name == "nt":
            self.active = False
            _read_windows()
            return
        import termios
        try:
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        finally:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._saved_mode)
            self.active = False


def _read_windows() -> list[str]:
    import msvcrt
    keys = []
    while msvcrt.kbhit():
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):  # special key: a second code follows
            token = _WINDOWS_KEYS.get(msvcrt.getwch())
            if token:
                keys.append(token)
        else:
            keys.append(ch)
    return keys


def _read_posix_bytes() -> bytes:
    import select
    fd, data = sys.stdin.fileno(), b""
    while select.select([fd], [], [], 0)[0]:
        chunk = os.read(fd, 1024)
        if not chunk:
            break
        data += chunk
    return data


def split_posix_keys(text: str) -> tuple[list[str], str]:
    """Keys in text, plus any unfinished escape sequence at the end to carry into the next
    read (over ssh/tmux a key like F12 can arrive split across reads). A lone ESC at the end
    is the Esc key itself."""
    unfinished = _UNFINISHED_ESCAPE.search(text)
    leftover = unfinished.group() if unfinished else ""
    text = text[: len(text) - len(leftover)]
    keys, index = [], 0
    for match in _ESCAPE_SEQUENCE.finditer(text):
        keys.extend(text[index:match.start()])
        token = _POSIX_KEYS.get(match.group(1))
        if token:
            keys.append(token)
        index = match.end()
    keys.extend(text[index:])
    return keys, leftover


def cell_width(ch: str) -> int:
    """Terminal columns a character takes: CJK and other wide characters take two."""
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def fit_head(text: str, cells: int) -> str:
    """The longest start of text that fits in cells columns."""
    used = 0
    for index, ch in enumerate(text):
        used += cell_width(ch)
        if used > cells:
            return text[:index]
    return text


def pad(text: str, cells: int) -> str:
    """text cut or space-padded to exactly cells columns."""
    text = fit_head(text, cells)
    return text + " " * (cells - sum(cell_width(ch) for ch in text))


def _can_encode(out: TextIO, chars: str) -> bool:
    try:
        chars.encode(getattr(out, "encoding", None) or "utf-8")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def enable_vt_codes() -> None:
    """Windows consoles only honour cursor-movement codes once this mode is switched on."""
    if os.name != "nt":
        return
    import ctypes
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
    mode = ctypes.c_uint32()
    if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
