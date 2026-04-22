#!/usr/bin/env python3
"""
Merge per-repetition result files produced by run_one_rep.py.

This script is optional but convenient after all array tasks finish.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge job-array result files.")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--func", type=str, default="dtlz2")
    parser.add_argument("--tag", type=str, default="joint", choices=["joint", "partial"])
    parser.add_argument("--num_reps", type=int, required=True)
    parser.add_argument("--out_prefix", type=str, default="all")
    args = parser.parse_args()
    if args.num_reps <= 0:
        parser.error("--num_reps must be a positive integer.")
    return args


def load_required(path: Path) -> Any:
    """Load a required `.npy` file, raising a clear error when missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    return np.load(path, allow_pickle=True)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Keep field naming centralized to avoid duplicated load/save blocks.
    fields = {
        "train_x": [],
        "train_y": [],
        "coupled_y": [],
        "hv": [],
        "true_hv": [],
        "times": [],
        "nsga_expansions": [],
        "non_invertible_count": [],
    }

    for rep in range(args.num_reps):
        stem = f"{rep}_{args.func}_{args.tag}"
        for field_name, bucket in fields.items():
            bucket.append(load_required(results_dir / f"{stem}_{field_name}.npy"))

    prefix = f"{args.out_prefix}_{args.func}_{args.tag}"
    for field_name, bucket in fields.items():
        np.save(results_dir / f"{prefix}_{field_name}.npy", np.array(bucket, dtype=object))

    print(f"Merged {args.num_reps} repetitions into {results_dir}/{prefix}_*.npy")


if __name__ == "__main__":
    main()
