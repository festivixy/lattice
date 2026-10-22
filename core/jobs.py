"""Everything the jobs print. Edit freely; run `python -m unittest` afterwards to check it.

Each entry in JOBS is one script: train.py runs JOBS["train"]. Fields:

  description   shown by --help
  logger        name shown on every log line
  params        flag -> (default, help). Each becomes --flag (underscores turn into dashes),
                appears in the config dump and --help, and is usable in templates as {flag}.
                Ints accept 1e9-style input.
  units         the param counting main-loop units (epochs, steps, ...)
  unit_seconds  real-world seconds per unit; unless --<units> is given, the count is set from
                --walltime so the run moves at that pace (the listed default only sets the type)
  unit_label    word in the progress bar ("epoch 3/12")
  inner         optional param: steps inside each unit (steps per epoch); the bar then shows
                progress through the current unit
  inner_rate    inner steps per second, used the same way as unit_seconds
  inner_label   rate unit in the progress bar ("it" -> "4.12it/s")
  sources       files (relative to the project) open as tabs in the editor beside the log
  options       (name, [values]) rows in the OPTIONS panel; picking one cycles its value and
                logs OPTION_CHANGE ({option} and {value} are the pick)
  metrics       name -> curve, recomputed for each main-loop line. kind is one of
                  exp     start -> end, fast at first, then levelling off
                  log     start -> end evenly in log space (residuals: 1 -> 1e-10)
                  cosine  start -> end along a cosine (learning-rate schedules)
                  linear  start -> end in a straight line
                  steady  stays near start (throughputs, temperatures)
                noise is relative jitter (it settles as the run finishes, except for steady);
                min/max clamp. start/end can be a function of the params dict.
  derived       name -> function of the template values, recomputed alongside the metrics
  setup         lines spread over the first part of the run. A dict
                {"bar": label, "total": count, "unit": word} animates a progress bar instead.
  line          the main-loop line, logged every few seconds; {step}/{steps} count inner steps
  events        one of these sometimes follows a main-loop line
  debug         interleaved with -v (occasionally) and -vv (constantly)
  teardown      like setup, at the end
  summary       (label, template) rows of the final table; wall time is added automatically

Templates are Python format strings ("{loss:.4f}"). Besides params, metrics and derived values:
  {hex8} {hex16} {hex64}   random hex digits (any length)
  {int2} {int5} ...        random number with that many digits
  {ms} {mb} {gb} {pct}     random timing / size / percentage
  {rank}                   random device index
  {i} {n}                  current unit and unit count; {step} {steps} for inner steps
  {host} {cores} {device} {devices} {run_id} {seed} {pid} {job}
A line starting with "WARNING " is logged at WARNING level.

To add a job: add an entry here and a script next to train.py that calls main("<name>").
"""
import math

PROGRAM = "lattice"
VERSION = "0.9.3"
DEVICE = "NVIDIA H100 80GB HBM3"
OPTION_CHANGE = "config change queued: {option}={value} (applies at next checkpoint)"
JOBS = {
    "train": {
        "description": "Distributed transformer pretraining.",
        "logger": "lattice.train",
        "params": {
            "epochs": (100, "number of training epochs"),
            "steps_per_epoch": (250, "optimizer steps per epoch"),
            "batch_size": (512, "global batch size in sequences"),
            "lr": (3e-4, "peak learning rate"),
            "model": ("lt-7b", "model preset"),
            "precision": ("bf16", "mixed-precision dtype"),
        },
        "units": "epochs",
        "unit_seconds": 25.0,
        "unit_label": "epoch",
        "inner": "steps_per_epoch",
        "inner_rate": 4.0,
        "inner_label": "it",
        "sources": ["src/model/attention.py", "src/model/transformer.py", "src/train/loop.py",
                    "configs/train.yaml"],
        "options": [("precision", ["bf16", "fp8", "fp16"]),
                    ("optimizer", ["adamw", "lion", "sophia"]),
                    ("scheduler", ["cosine", "wsd", "linear"]),
                    ("grad_checkpointing", ["on", "off"])],
        "metrics": {
            "loss": {"kind": "exp", "start": 4.61, "end": 0.284, "rate": 4.0, "noise": 0.03},
            "acc": {"kind": "exp", "start": 0.08, "end": 0.913, "rate": 3.5, "noise": 0.015,
                    "max": 0.999},
            "lr_now": {"kind": "cosine", "start": lambda p: p["lr"],
                       "end": lambda p: p["lr"] * 0.01},
            "gnorm": {"kind": "steady", "start": 0.92, "noise": 0.2, "min": 0.05},
            "tok_s": {"kind": "steady", "start": 412_000, "noise": 0.03},
        },
        "derived": {
            "ppl": lambda c: math.exp(c["loss"]),
            "tokens": lambda c: c["step"] * c["batch_size"] * 4096,
        },
        "setup": [
            "initializing process group (backend=nccl, world_size={devices})",
            "rank 0 on {host}: {cores} host cores, {devices}x {device}",
            "loading tokenizer bpe-128k (128256 merges)",
            {"bar": "indexing shards", "total": 512, "unit": "shard"},
            "building {model}: 32 layers, d_model 4096, 32 heads, rope theta 500000",
            "parameters: 6.74B total, 6.74B trainable",
            "sharding FULL_SHARD, precision {precision}, activation checkpointing on",
            {"bar": "compiling graph (mode=max-autotune)", "total": 48, "unit": "kernel"},
            "schedule cosine, warmup 2000 steps, peak lr {lr:.1e}",
        ],
        "line": "epoch {i:>3}/{n} | step {step:>6,}/{steps:,} | loss {loss:.4f} | acc {acc:.4f} | lr {lr_now:.2e} "
                "| gnorm {gnorm:.3f} | {tok_s:,.0f} tok/s",
        "events": [
            "checkpoint saved -> checkpoints/{model}/ep{i:04d}.safetensors ({gb:.1f} GB)",
            "eval: val_loss {loss:.4f} | ppl {ppl:.3f}",
            "WARNING grad norm spike at step {int5}, clipped to 1.0",
            "dataloader: prefetch queue {int2}/64, rebalancing workers",
            "nccl allreduce p99 {ms:.2f} ms",
        ],
        "debug": [
            "[rank {rank}] fwd {ms:.2f} ms | bwd {ms:.2f} ms | opt {ms:.2f} ms",
            "[rank {rank}] allreduce bucket {int3}: {mb:.1f} MB in {ms:.2f} ms",
            "[rank {rank}] mem allocated {gb:.2f} GB / 80.00 GB",
            "[sched] step {int6} lr {lr_now:.3e}",
        ],
        "teardown": [
            "final eval on held-out split",
            {"bar": "evaluating", "total": 64, "unit": "batch"},
            "writing final checkpoint -> checkpoints/{model}/final.safetensors",
            "flushing metrics -> runs/{run_id}/events.jsonl",
        ],
        "summary": [
            ("final loss", "{loss:.4f}"),
            ("final accuracy", "{acc:.4f}"),
            ("perplexity", "{ppl:.3f}"),
            ("tokens processed", "{tokens:,.0f}"),
            ("throughput", "{tok_s:,.0f} tok/s"),
        ],
    },
    "simulate": {
        "description": "Gravitational N-body simulation (Barnes-Hut).",
        "logger": "lattice.nbody",
        "params": {
            "particles": (1_000_000, "number of bodies"),
            "steps": (5000, "integration steps"),
            "dt": (1e-3, "time step"),
            "integrator": ("leapfrog", "time integrator"),
            "box": (512.0, "periodic box side length"),
        },
        "units": "steps",
        "unit_seconds": 0.4,
        "unit_label": "step",
        "inner_label": "step",
        "sources": ["src/nbody/forces.cu", "src/nbody/octree.py", "configs/simulate.yaml"],
        "options": [("integrator", ["leapfrog", "yoshida4", "rk4"]),
                    ("opening_angle", ["0.5", "0.4", "0.7"]),
                    ("snapshot_every", ["500", "250", "1000"])],
        "metrics": {
            "drift": {"kind": "log", "start": 1e-12, "end": 4e-8, "noise": 0.2},
            "temp": {"kind": "steady", "start": 300.0, "noise": 0.004},
            "press": {"kind": "steady", "start": 1.013, "noise": 0.02},
            "virial": {"kind": "steady", "start": 1.0, "noise": 0.008},
            "ips": {"kind": "steady", "start": 2.1e12, "noise": 0.05},
        },
        "derived": {
            "t": lambda c: c["i"] * c["dt"],
            "mem_gb": lambda c: c["particles"] * 96 / 1e9 / max(1, c["devices"]),
        },
        "setup": [
            "initializing {devices} ranks on {host} ({cores} host cores), device {device}",
            "allocating particle buffers: {particles:,} bodies, {mem_gb:.2f} GB per rank",
            "initial conditions: plummer sphere, virial ratio 1.0, seed {seed}",
            {"bar": "partitioning domain", "total": 256, "unit": "cell"},
            "building octree (barnes-hut, theta 0.5, leaf size 16)",
            "neighbor lists: rcut 2.5 sigma, skin 0.3 sigma",
            {"bar": "computing initial forces", "total": 64, "unit": "tile"},
            "integrator {integrator}, dt {dt:g}, box {box:g}^3, periodic",
        ],
        "line": "step {i:>6}/{n} | t {t:9.3f} | dE/E {drift:.3e} | T {temp:7.2f} K "
                "| P {press:.3f} bar | {ips:.2e} pair/s",
        "events": [
            "snapshot written -> out/snap_{i:06d}.h5 ({gb:.2f} GB)",
            "WARNING load imbalance {pct:.1f}% across ranks, repartitioning domain",
            "rebuilt neighbor lists ({int2} pairs/cell avg)",
            "virial check: 2K/|W| = {virial:.4f}",
        ],
        "debug": [
            "[rank {rank}] force kernel {ms:.2f} ms, {int8} interactions",
            "[rank {rank}] halo exchange {mb:.1f} MB in {ms:.2f} ms",
            "[tree] rebuild depth {int2}, {int7} nodes",
            "[rank {rank}] kick-drift-kick {ms:.3f} ms",
        ],
        "teardown": [
            "writing final snapshot -> out/snap_final.h5",
            {"bar": "reducing diagnostics", "total": 32, "unit": "block"},
            "energy conservation |dE/E| = {drift:.2e} over {n:,} steps",
        ],
        "summary": [
            ("steps", "{n:,}"),
            ("simulated time", "{t:.3f}"),
            ("energy drift", "{drift:.3e}"),
            ("mean temperature", "{temp:.2f} K"),
            ("throughput", "{ips:.2e} pair/s"),
        ],
    },
    "solve": {
        "description": "Preconditioned Krylov solve of a large sparse system.",
        "logger": "lattice.krylov",
        "params": {
            "dim": (2_000_000, "matrix dimension"),
            "maxiter": (800, "maximum Krylov iterations"),
            "tol": (1e-10, "relative residual tolerance"),
            "method": ("gmres", "Krylov method"),
            "restart": (50, "restart length"),
            "precond": ("amg", "preconditioner"),
        },
        "units": "maxiter",
        "unit_seconds": 0.15,
        "unit_label": "iter",
        "inner_label": "it",
        "sources": ["src/krylov/gmres.py", "src/krylov/amg.py", "configs/solve.yaml"],
        "options": [("preconditioner", ["amg", "ilu0", "jacobi"]),
                    ("restart", ["50", "30", "100"]),
                    ("orthogonalization", ["cgs2", "mgs"])],
        "metrics": {
            "res": {"kind": "log", "start": 1.0, "end": lambda p: p["tol"] * 0.8, "noise": 0.25},
            "orth": {"kind": "log", "start": 1e-16, "end": 4e-14, "noise": 0.1},
            "gflops": {"kind": "steady", "start": 1840.0, "noise": 0.04},
            "cond": {"kind": "steady", "start": 3.4e8},
        },
        "derived": {
            "nnz": lambda c: c["dim"] * 27,
        },
        "setup": [
            "initializing {devices} ranks on {host} ({cores} host cores), device {device}",
            "assembling sparse operator: n = {dim:,}, nnz = {nnz:,} (27-point stencil)",
            "reordering: reverse cuthill-mckee, bandwidth {int5} -> {int3}",
            {"bar": "AMG setup (coarsening)", "total": 12, "unit": "level"},
            "estimating spectrum: lanczos, 40 steps",
            "condition number estimate {cond:.2e}",
            {"bar": "factorizing coarse level", "total": 96, "unit": "panel"},
            "method {method}(restart={restart}), preconditioner {precond}, tol {tol:.0e}",
        ],
        "line": "iter {i:>5}/{n} | ||r||/||b|| {res:.3e} | orth {orth:.1e} "
                "| {gflops:,.0f} GFLOP/s",
        "events": [
            "restart cycle complete, Krylov basis reset",
            "WARNING stagnation detected, raising smoother sweeps to 3",
            "residual replacement at iter {i}",
            "checkpoint: out/x_{i:05d}.bin ({gb:.2f} GB)",
        ],
        "debug": [
            "[spmv] {ms:.3f} ms, {gflops:,.0f} GFLOP/s",
            "[orth] cgs2 pass {ms:.3f} ms",
            "[amg] v-cycle level {int1}: {ms:.2f} ms",
            "[rank {rank}] allreduce {mb:.1f} MB in {ms:.2f} ms",
        ],
        "teardown": [
            "converged: relative residual {res:.3e} < tol {tol:.0e}",
            {"bar": "backward error check", "total": 32, "unit": "block"},
            "writing solution -> out/x.bin ({gb:.2f} GB)",
        ],
        "summary": [
            ("iterations", "{n:,}"),
            ("final residual", "{res:.3e}"),
            ("condition estimate", "{cond:.2e}"),
            ("throughput", "{gflops:,.0f} GFLOP/s"),
        ],
    },
    "search": {
        "description": "Distributed hash-target search over a 64-bit nonce space.",
        "logger": "lattice.search",
        "params": {
            "bits": (48, "leading zero bits required"),
            "blocks": (4000, "nonce blocks to scan"),
            "algo": ("sha256d", "hash function"),
            "threads": (64, "worker threads"),
        },
        "units": "blocks",
        "unit_seconds": 1.5,
        "unit_label": "block",
        "inner_label": "blk",
        "sources": ["src/search/sha256d.cu", "src/search/scheduler.py", "configs/search.yaml"],
        "options": [("threads", ["64", "128", "32"]),
                    ("work_stealing", ["on", "off"]),
                    ("kernel_block", ["256", "512", "128"])],
        "metrics": {
            "ghs": {"kind": "steady", "start": 4.71, "noise": 0.04},
            "best": {"kind": "linear", "start": 12, "end": lambda p: p["bits"]},
        },
        "derived": {
            "hashes": lambda c: c["i"] * 2**32,
            "scanned": lambda c: 100 * c["i"] / c["n"],
        },
        "setup": [
            "initializing {devices} devices on {host} ({cores} host cores), device {device}",
            "target: {bits} leading zero bits, algo {algo}",
            "spawning {threads} worker threads",
            {"bar": "precomputing midstates", "total": 4096, "unit": "state"},
            "nonce space 2^64 partitioned into {n:,} blocks",
            {"bar": "warming up kernels", "total": 16, "unit": "kernel"},
        ],
        "line": "block {i:>5}/{n} | {ghs:.2f} GH/s | best {best:.0f}/{bits} zero bits "
                "| {hashes:.3e} hashes",
        "events": [
            "share found: {hex16} ({best:.0f} bits)",
            "WARNING worker {int2} stalled, rescheduling block {int4}",
            "rotated extranonce -> {hex8}",
            "{scanned:.1f}% of nonce space scanned",
        ],
        "debug": [
            "[w{int2}] block {int5} nonce 0x{hex16} -> {hex16}...",
            "[sched] queue depth {int3}, steals {int2}",
            "[dev {rank}] kernel {ms:.2f} ms, occupancy {int2}%",
        ],
        "teardown": [
            "MATCH nonce=0x{hex16}",
            "digest {hex64}",
            {"bar": "verifying match", "total": 8, "unit": "round"},
            "result written -> results/{run_id}.json",
        ],
        "summary": [
            ("best", "{best:.0f}/{bits} zero bits"),
            ("hashes", "{hashes:.3e}"),
            ("hash rate", "{ghs:.2f} GH/s"),
        ],
    },
}
