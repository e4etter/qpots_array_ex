#!/usr/bin/env python3
"""
Run exactly one optimization repetition (REP) using pip-installed qpots.

This is designed for Slurm job arrays where each array element runs one REP.
No MPI is used; parallelism is handled by Slurm scheduling many array tasks.
"""

import argparse
import os
import resource
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning
from botorch.utils.transforms import unnormalize
from linear_operator.utils.warnings import NumericalWarning

from qpots.acquisition import Acquisition
from qpots.function import Function
from qpots.model_object import ModelObject
from qpots.utils.utils import expected_hypervolume

# Suppress a known benign GPyTorch warning emitted repeatedly during GP refits.
# qpots configures very small noise and GPyTorch safely clamps it to 1e-6.
warnings.filterwarnings(
    "ignore",
    message="Very small noise values detected.*Rounding small noise values up to 1e-06.",
    category=NumericalWarning,
)


def configure_torch_threads() -> tuple[int, int]:
    """Configure PyTorch CPU threading from Slurm/env settings."""
    # Total CPU cores allocated to this Slurm task.
    n_cores = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    # Inter-op threads = how many independent PyTorch ops can run concurrently.
    # Allow explicit override for tuning experiments.
    if "TORCH_NUM_INTEROP_THREADS" in os.environ:
        n_interop = max(1, int(os.environ["TORCH_NUM_INTEROP_THREADS"]))
    elif n_cores <= 1:
        n_interop = 1
    elif n_cores == 2:
        n_interop = 2
    else:
        # Heuristic: keep inter-op modest (2-4) on larger core counts so each
        # op still has enough intra-op parallelism for BLAS/OpenMP kernels.
        n_interop = min(4, max(2, n_cores // 2))

    # Intra-op threads = threads used within a single op.
    # Split total cores across inter-op workers to reduce oversubscription.
    n_intra = max(1, (n_cores + n_interop - 1) // n_interop)
    torch.set_num_threads(n_intra)
    try:
        # Must be set before parallel work starts; otherwise PyTorch may raise.
        torch.set_num_interop_threads(n_interop)
    except RuntimeError as exc:
        print(f"[THREADS] warning: could not set interop threads ({exc})", flush=True)

    return torch.get_num_threads(), torch.get_num_interop_threads()


def log_memory(stage: str, extra: str = "") -> None:
    """
    Emit process-level memory checkpoints similar to legacy `[MEM]` logs.

    On Linux, `ru_maxrss` is in KB and represents peak RSS so far.
    """
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_gb = rss_kb / (1024 * 1024)
    suffix = f" | {extra}" if extra else ""
    print(f"[MEM] stage={stage} peak_rss_gb={rss_gb:.3f}{suffix}", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for a single-repetition optimization run."""
    parser = argparse.ArgumentParser(description="Single-repetition qPOTS runner for Slurm job arrays.")
    parser.add_argument("--rep", type=int, required=True, help="Repetition index (typically SLURM_ARRAY_TASK_ID).")
    parser.add_argument("--start_seed", type=int, default=1023, help="Base seed; effective seed is start_seed + rep.")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory for npy files.")

    # Problem setup
    parser.add_argument("--func", type=str, default="dtlz2")
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument("--nobj", type=int, default=10)
    parser.add_argument("--ncons", type=int, default=0)
    parser.add_argument("--ref_point", type=float, nargs="+", required=True)

    # Optimization setup
    parser.add_argument("--ntrain", type=int, default=300)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument("--ngen", type=int, default=100)
    parser.add_argument("--nystrom", type=int, default=0)
    parser.add_argument("--nychoice", type=str, default="pareto", choices=["pareto", "random"])

    # Backward-compatible args from prior custom runner.
    # These are accepted so existing Slurm scripts still work, but they do not
    # change behavior in this pip-only implementation.
    parser.add_argument("--max_nsga_multiplier", type=int, default=1)
    parser.add_argument("--use_partial", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--log_memory", action="store_true")
    parser.add_argument("--memory_log_every", type=int, default=1)
    return parser.parse_args()


def fit_gp_models(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    bounds: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> ModelObject:
    """Build and fit independent GP models from pip-installed qpots."""
    # qpots' ModelObject stores one GP per objective/constraint output.
    gps = ModelObject(
        train_x=train_x,
        train_y=train_y,
        bounds=bounds,
        nobj=args.nobj,
        ncons=args.ncons,
        device=device,
    )
    gps.fit_gp()
    return gps


def compute_true_hypervolume(
    train_y: torch.Tensor,
    ref_point: torch.Tensor,
    nobj: int,
    ncons: int,
    debug_mem: bool = False,
) -> float:
    """
    Compute hypervolume directly from observed outcomes.

    When ``debug_mem`` is enabled, prints light-weight diagnostics that help
    identify potential memory pressure in the hypervolume path.
    """
    t0 = time.time()
    if ncons > 0:
        # Only objective columns from feasible rows contribute to constrained HV.
        feasible = (train_y[..., -ncons:] >= 0).all(dim=-1)
        feasible_obj = train_y[feasible][..., :nobj].double()
        if debug_mem:
            print(
                f"[HV] feasible points: {feasible_obj.shape[0]}/{train_y.shape[0]} "
                f"(nobj={nobj}, ncons={ncons})",
                flush=True,
            )
        if feasible_obj.numel() == 0:
            if debug_mem:
                print("[HV] no feasible points, returning 0.0", flush=True)
            return 0.0
        y_for_hv = feasible_obj
    else:
        # Unconstrained case: all rows contribute.
        y_for_hv = train_y[..., :nobj].double()

    est_bytes = y_for_hv.numel() * y_for_hv.element_size()
    if debug_mem:
        print(
            f"[HV] tensor shape={tuple(y_for_hv.shape)}, dtype={y_for_hv.dtype}, "
            f"approx_tensor_mem={est_bytes / (1024**2):.2f} MB",
            flush=True,
        )
        if y_for_hv.shape[0] > 5000:
            print("[HV] large point set detected; partitioning may cause memory spikes.", flush=True)

    t1 = time.time()
    # FastNondominatedPartitioning is the heavy step and most likely to expose
    # memory pressure as the candidate set grows.
    partitioning = FastNondominatedPartitioning(ref_point.double(), y_for_hv)
    hv = float(partitioning.compute_hypervolume())
    t2 = time.time()
    if debug_mem:
        print(
            f"[HV] partition+compute_time={t2 - t1:.3f}s total_time={t2 - t0:.3f}s hv={hv:.6g}",
            flush=True,
        )
    return hv


def save_outputs(
    outdir: Path,
    stem: str,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    coupled_y: torch.Tensor,
    hvs: list[float],
    true_hvs: list[float],
    times: list[float],
) -> None:
    """Persist all per-repetition output arrays with a consistent naming scheme."""
    # Keep naming stable so merge_results.py can aggregate per-rep files.
    np.save(outdir / f"{stem}_train_x.npy", train_x.cpu().numpy())
    np.save(outdir / f"{stem}_train_y.npy", train_y.cpu().numpy())
    np.save(outdir / f"{stem}_coupled_y.npy", coupled_y.cpu().numpy())
    np.save(outdir / f"{stem}_hv.npy", np.array(hvs, dtype=float))
    np.save(outdir / f"{stem}_true_hv.npy", np.array(true_hvs, dtype=float))
    np.save(outdir / f"{stem}_times.npy", np.array(times, dtype=float))
    # Keep these files for compatibility with merge_results.py.
    np.save(outdir / f"{stem}_nsga_expansions.npy", np.array([0], dtype=int))
    np.save(outdir / f"{stem}_non_invertible_count.npy", np.array([0], dtype=int))


def main() -> None:
    args = parse_args()
    if args.use_partial:
        raise NotImplementedError("Partial evaluation mode is unavailable with pip-installed qpots.")
    if args.log_memory:
        print(
            "[INFO] --log_memory enabled: emitting process-level [MEM] checkpoints "
            "plus [HV] diagnostics.",
            flush=True,
        )
        log_memory("rep:start")

    n_intra, n_interop = configure_torch_threads()
    
    # qpots internals assume double precision for stable GP fitting.
    dtype = torch.double
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rep_seed = args.start_seed + args.rep

    print(
        f"[REP {args.rep}] device={device}, torch_intra_threads={n_intra}, "
        f"torch_interop_threads={n_interop}, seed={rep_seed}",
        flush=True,
    )

    tf = Function(args.func, dim=args.dim, nobj=args.nobj)
    f = tf.evaluate
    bounds = tf.get_bounds()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(rep_seed)
    # Initial design is sampled in the normalized [0, 1]^d search space.
    train_x = torch.rand([args.ntrain, args.dim], dtype=dtype)
    # Function evaluation expects physical-domain points; unnormalize first.
    train_y = f(unnormalize(train_x, bounds))
    if args.log_memory:
        log_memory("rep:after_initial_data", extra=f"train_x_shape={tuple(train_x.shape)}")

    # `_full` tracks the data used to train each subsequent GP refit.
    train_x_full = train_x.clone()
    train_y_full = train_y.clone()
    # `coupled_train_y` remains fully observed and is used only for "true" HV.
    coupled_train_y = train_y.clone()

    ref_point = torch.tensor(args.ref_point, dtype=dtype)
    true_hvs = [
        compute_true_hypervolume(
            coupled_train_y,
            ref_point,
            args.nobj,
            args.ncons,
            debug_mem=args.log_memory,
        )
    ]

    gps = fit_gp_models(train_x_full, train_y_full, bounds, args, device)
    acq = Acquisition(tf, gps, device=device, q=args.q)
    # qpots utility returns (hv, pareto_front); keep scalar HV only.
    hvs = [float(expected_hypervolume(gps, ref_point=ref_point)[0])]
    times = []

    for it in range(args.iters):
        t1 = time.time()
        should_log_iter_mem = args.log_memory and (it % max(args.memory_log_every, 1) == 0)
        if should_log_iter_mem:
            log_memory(f"iter_{it}:start")

        # qPOTS proposes up to q points in normalized space.
        x_new = acq.qpots(
            bounds=bounds,
            iteration=it,
            nystrom=args.nystrom,
            iters=args.iters,
            nychoice=args.nychoice,
            dim=args.dim,
            ngen=args.ngen,
            q=args.q,
        )
        x_new = x_new.reshape(-1, args.dim)
        if should_log_iter_mem:
            log_memory(f"iter_{it}:after_candidate_selection", extra=f"x_new_shape={tuple(x_new.shape)}")
        # Evaluate proposed points in physical space.
        y_new = f(unnormalize(x_new, bounds))
        if should_log_iter_mem:
            log_memory(f"iter_{it}:after_objective_eval", extra=f"y_new_shape={tuple(y_new.shape)}")

        # Append newly observed data and refit surrogate models.
        train_x_full = torch.row_stack([train_x_full, x_new])
        train_y_full = torch.row_stack([train_y_full, y_new])

        # Keep fully observed values for diagnostic "true" HV tracking.
        coupled_new_y = f(unnormalize(x_new, bounds))
        coupled_train_y = torch.row_stack([coupled_train_y, coupled_new_y])

        true_hv = compute_true_hypervolume(
            coupled_train_y,
            ref_point,
            args.nobj,
            args.ncons,
            debug_mem=args.log_memory,
        )
        true_hvs.append(true_hv)

        gps = fit_gp_models(train_x_full, train_y_full, bounds, args, device)
        acq = Acquisition(tf, gps, device=device, q=args.q)
        hvs.append(float(expected_hypervolume(gps, ref_point=ref_point)[0]))
        if should_log_iter_mem:
            log_memory(f"iter_{it}:after_gp_refit")

        t2 = time.time()
        times.append(t2 - t1)
        print(
            f"[REP {args.rep}] iter={it} elapsed={t2-t1:.2f}s true_hv={true_hv} "
            f"x_new_shape={tuple(x_new.shape)}",
            flush=True,
        )

    tag = "partial" if args.use_partial else "joint"
    stem = f"{args.rep}_{args.func}_{tag}"
    save_outputs(
        outdir=outdir,
        stem=stem,
        train_x=train_x_full,
        train_y=train_y_full,
        coupled_y=coupled_train_y,
        hvs=hvs,
        true_hvs=true_hvs,
        times=times,
    )
    if args.log_memory:
        log_memory("rep:finished", extra=f"iters={args.iters}")
    print(f"[REP {args.rep}] finished. Outputs written to: {outdir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
