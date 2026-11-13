# Lattice Typer

Mash keys, get real code. Open a file from your project and press **Ctrl+Alt+Y**
(**Ctrl+Cmd+Y** on a Mac). The file is swapped for a copy that
is partly written, and every key you press types the next character of the real file.

| Key | |
|---|---|
| letters | type the real code, one character per key |
| Backspace / Ctrl+Z | right after typing: un-type (typing again retypes the same code) |
| arrows, mouse | move around; mashing away from where the writing stopped types comments |
| Enter | a normal new line |
| F12 | type exactly what you press, and back |
| Ctrl+Alt+Y | stop: the copy closes and your real file comes back |

When a file is finished, the next one from your project opens. Your real files are never
changed: the copies live only in memory.

Doesn't work alongside the Vim extension, which also takes over typing.

## Install

From the repo root, `.\install.ps1` (Windows) or `sh install.sh` (macOS / Linux) installs it
when VS Code's `code` command is available. Or in VS Code: Extensions, `...` menu,
**Install from VSIX...**, and pick `vscode/lattice-typer.vsix`.

## Develop

    npm install
    npm test                # draft.js unit tests
    npm run test:vscode     # end-to-end, in a downloaded VS Code
    npm run package         # rebuild lattice-typer.vsix
