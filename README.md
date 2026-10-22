# lattice

Distributed compute workloads behind one runner: transformer pretraining, gravitational
N-body simulation, preconditioned sparse Krylov solves and hash-target search.

## Run

    python train.py --epochs 200 --walltime 00:05:00
    python simulate.py --particles 1e9 --steps 20000 -v
    python solve.py --dim 8e6 --tol 1e-12 --walltime 10m
    python search.py --bits 64 -vv

Every script takes `--help` for its own flags.

## Common flags

| Flag | |
|---|---|
| `--walltime` | wall-clock budget: `HH:MM:SS`, `MM:SS`, `90s`, `5m`, `1h30m` (default 1 min) |
| `-v`, `-vv` | more log output; `-vv` adds per-rank debug logs |
| `--seed` | fix the run's seed |
| `--devices` | device count |

Other `--key value` flags (e.g. `--config configs/train.yaml`) pass through into the run config.
## Layout

| Path | |
|---|---|
| `train.py`, `simulate.py`, `solve.py`, `search.py` | entrypoints |
| `core/engine.py` | runner: scheduling, logging, progress |
| `core/jobs.py` | job definitions |
| `src/` | model, N-body, Krylov and search kernels |
| `configs/` | presets |

## Tests

    python -m unittest
