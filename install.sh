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
echo "done - open a new terminal and run: train 5m"
