# Memory Scaling for `run_one_rep.py`

This document explains how `run_one_rep.py` options influence memory usage and where memory pressure typically appears.


## Where memory is spent

Primary memory consumers in this workflow:

1. **GP training structures** created when fitting `ModelObject` each iteration.
2. **Accumulated datasets** (`train_x_full`, `train_y_full`, and coupled outputs).
3. **Acquisition-side temporaries** during `acq.qpots(...)`.
4. **Hypervolume partitioning buffers** (diagnostic path in `compute_true_hypervolume`).

The dominant growth pattern comes from repeatedly refitting GPs as the dataset grows.


## Growth model

Approximate total design points over one repetition:

- `N ≈ ntrain + iters * q`

Stored tensors grow roughly linearly with `N`:

- `train_x_full`: shape `(N, dim)`
- `train_y_full`: shape `(N, nobj)`
- `coupled_train_y`: shape `(N, nobj)` in unconstrained runs

Even though these arrays are modest by themselves, GP fitting overhead grows much faster as `N` increases.


## Option-by-option memory impact

High impact:

- `--ntrain`: largest immediate increase; directly increases initial fit size.
- `--iters`: increases the number of refits and final dataset size.
- `--q`: increases growth rate per iteration (`+q` points each iteration).
- `--nobj`: increases output width and model complexity.

Medium impact:

- `--dim`: increases input tensor width and acquisition complexity.
- `--ngen`: can increase temporary allocation in inner search.

Lower direct memory impact:

- `--rep`, `--start_seed`, `--outdir`, `--nychoice`.

Compatibility flags (accepted, limited effect in current flow):

- `--max_nsga_multiplier`, `--threshold`, `--memory_log_every`.

Unsupported in current pip-only flow:

- `--use_partial` (raises `NotImplementedError`).


## Practical memory tuning order

If memory is tight:

1. Reduce `--ntrain`.
2. Reduce `--q`.
3. Reduce `--iters`.
4. Reduce `--nobj` and/or `--dim`.
5. Reduce `--ngen` if acquisition-side pressure is visible.

Then increase Slurm `--mem` only if needed after parameter-side reductions.

Note: Use `--log_memory` for hyper-volume (HV) and memory (MEM) diagnostics. These logs help identify whether memory spikes are happening during HV diagnostics versus GP/acquisition steps.


## Suggested starting points by objective count

- **2-3 objectives**: start moderate (`ntrain` 50-200, `q` 1-4).
- **4-8 objectives**: keep `ntrain` conservative and increase `iters` gradually.
- **9+ objectives**: expect steeper memory growth; scale slowly and monitor logs.


## OOM troubleshooting checklist

1. Check `.err` for abrupt termination without Python traceback (possible OOM kill).
2. Re-run with `--log_memory` and inspect `[HV]` and `[MEM]` diagnostics in `.out`.
3. Lower `ntrain`, `q`, and `iters` in that order.
4. Confirm each repetition completes before increasing array width.
5. Increase `--mem` once parameter tuning no longer meets experiment needs.
