# One-word commands for the lattice jobs, sourced from ~/.zshrc / ~/.bashrc (see install.sh),
# which sets LATTICE_DIR first. `train 30m -vv` or `train -30m -vv` runs train.py with a
# 30-minute walltime from any folder; everything after the duration passes straight through.

_lattice() {
    _lattice_job=$1
    shift
    if command -v python3 >/dev/null 2>&1; then
        python3 "$LATTICE_DIR/$_lattice_job.py" "$@"
    else
        python "$LATTICE_DIR/$_lattice_job.py" "$@"
    fi
}

train() { _lattice train "$@"; }
simulate() { _lattice simulate "$@"; }
solve() { _lattice solve "$@"; }
search() { _lattice search "$@"; }
