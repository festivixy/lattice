#!/bin/sh
# Adds the train / simulate / solve / search commands to zsh and bash: sh install.sh
set -e
dir=$(cd "$(dirname "$0")" && pwd)
quoted="'$(printf '%s' "$dir" | sed "s/'/'\\\\''/g")'"
line="LATTICE_DIR=$quoted; [ -f \"\$LATTICE_DIR/shortcuts.sh\" ] && . \"\$LATTICE_DIR/shortcuts.sh\""

added=0
for rc in "$HOME/.zshrc" "$HOME/.bashrc" "$HOME/.bash_profile"; do
    [ -f "$rc" ] || continue
    added=1
    if grep -qF "LATTICE_DIR=$quoted" "$rc"; then
        echo "already set up: $rc"
    else
        printf '\n%s\n' "$line" >> "$rc"
        echo "added to: $rc"
    fi
done
if [ "$added" = 0 ]; then  # fresh account: start the shell's own startup file
    case "$(basename "${SHELL:-sh}")" in
        zsh) rc="$HOME/.zshrc" ;;
        *) rc="$HOME/.bashrc" ;;
    esac
    printf '%s\n' "$line" >> "$rc"
    echo "added to: $rc"
fi

# the fake-typing VS Code extension (vscode/), when VS Code's `code` command is available
if command -v code >/dev/null 2>&1; then
    if code --install-extension "$dir/vscode/lattice-typer.vsix" --force >/dev/null 2>&1; then
        echo "installed the VS Code extension (Ctrl+Cmd+Y on a Mac, Ctrl+Alt+Y on Linux)"
    else
        echo "could not install the VS Code extension; see vscode/README.md"
    fi
fi
echo "done - open a new terminal and run: train 5m"
