# Runtime Scaling for `run_one_rep.py`

This document explains how `run_one_rep.py` options drive wall-time and how to tune for faster throughput.

## Runtime structure per repetition

Each repetition has three recurring costs:

1. **Model fitting** (`fit_gp_models`): done once initially and once per iteration.
2. **Acquisition search** (`acq.qpots(...)`): inner optimization work each iteration.
3. **True function evaluation** (`f(unnormalize(...))`): cost depends on the selected benchmark function and `q`.

As `N = ntrain + iters*q` grows, later iterations become slower because model refits are more expensive.

## Option-by-option runtime impact

Strong impact:

- `--iters`: directly multiplies outer-loop work.
- `--ntrain`: increases initial fit time and raises baseline for all later refits.
- `--q`: increases per-iteration evaluation count and accelerates dataset growth.
- `--nobj`: increases output/model complexity.

Medium to strong impact:

- `--dim`: increases optimization and modeling complexity.
- `--ngen`: increases acquisition inner-loop work per iteration.
- `--nystrom`: changes the acquisition internals; can trade quality and speed depending on setup.

Low direct runtime impact:

- `--rep`, `--start_seed`, `--outdir`, `--ref_point` (except indirect effects through problem geometry).

Compatibility flags (accepted, limited/no effect in current flow):

- `--max_nsga_multiplier`, `--threshold`, `--memory_log_every`.

Unsupported in current pip-only flow:

- `--use_partial`.

## Practical runtime tuning

If runtime is too long:

1. Lower `--iters` first (largest direct reduction).
2. Lower `--ngen` to reduce inner-search cost.
3. Lower `--q` if batch size is not required.
4. Lower `--ntrain` if initial fit is too heavy.
5. Lower `--dim` / `--nobj` if scientifically acceptable.

If you need better solution quality under a fixed budget:

1. Keep `q` modest.
2. Increase `iters` gradually.
3. Increase `ngen` only after confirming acquisition quality bottlenecks.

## Throughput tips for Slurm arrays

- Prefer many moderate array tasks over one very large task.
- Match `SLURM_CPUS_PER_TASK` with thread env vars (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, etc.).
- Avoid over-allocation of CPU threads when model sizes are small.
- Benchmark 1-2 pilot repetitions before submitting large arrays.

## Interpreting iteration logs

Per-iteration output includes:

- elapsed time,
- true hypervolume,
- `x_new` shape.

Use these to detect whether runtime is rising smoothly (expected with growing data) or showing sudden spikes (often acquisition/model-fit instability or resource contention).

## Recommended benchmark workflow

1. Run with reduced settings (small `ntrain`, small `iters`) to verify correctness.
2. Measure median per-iteration time from logs.
3. Estimate full-run wall-time:
   - `total ≈ setup_time + sum(iter_times)`
4. Scale `iters`/array width to meet deadline and queue constraints.
