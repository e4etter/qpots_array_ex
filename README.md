# qPOTS Job Array Example

This repository runs one optimization repetition per Slurm array task using `run_one_rep.py` and pip-installed `qpots` ([repo](https://github.com/csdlpsu/qpots)).


## Layout

| Path | Role |
|------|------|
| `bo_array.slurm` | Slurm submit script. Edit `#SBATCH` resources, array range, and CLI args for your site. |
| `run_one_rep.py` | Single repetition runner (init data, GP fit, BO loop, outputs). |
| `merge_results.py` | Merges per-repetition outputs into `all_<func>_<tag>_*.npy`. |
| `environment.yml` | Conda environment specification for local `./env`. |
| `scaling_memory.md` | Memory behavior and tuning guidance mapped to `run_one_rep.py` options. |
| `scaling_runtime.md` | Runtime behavior and tuning guidance mapped to `run_one_rep.py` options. |


## Quick Start (On RC Compute Node)

```bash
git clone https://github.com/e4etter/qpots_array_ex.git
cd qpots_array_ex
```

From this directory:
```bash
module purge
module load anaconda/2023.09

conda env create -p ./env -f environment.yml
conda activate ./env
```

Install `botorch`:
```bash
python -m pip install --no-input botorch==0.12.0 gpytorch==1.13
python -c "import torch, botorch, gpytorch, pymoo; print('env ok')"
```

Install [qPOTS](https://github.com/csdlpsu/qpots)
```bash
git clone https://github.com/csdlpsu/qpots
cd qpots
pip install .
cd ..
python -c "import qpots; print('qpots is available')"
```


Submit (defaults inside `bo_array.slurm` are **examples**; change `#SBATCH` and CLI flags before production use):

```bash
sbatch bo_array.slurm
```

Override the number of repetitions at submit time:

```bash
REPS=10
sbatch --array=0-$((REPS-1)) bo_array.slurm
```

## Outputs

Per repetition `rep`, function `func`, and mode `joint` or `partial`:

- `results_*/<rep>_<func>_<tag>_train_x.npy`, `_train_y.npy`, `_coupled_y.npy`
- `results_*/<rep>_<func>_<tag>_hv.npy`, `_true_hv.npy`, `_times.npy`
- `results_*/<rep>_<func>_<tag>_nsga_expansions.npy`, `_non_invertible_count.npy`

Merge after all tasks succeed:

```bash
python merge_results.py --results_dir results_<jobid> --func dtlz2 --tag joint --num_reps <num reps>
```

## Core `run_one_rep.py` options

Most impactful knobs:

- **Problem setup**: `--func`, `--dim`, `--nobj`, `--ncons`, `--ref_point`
- **Budget / loop size**: `--ntrain`, `--iters`, `--q`
- **Inner qPOTS search**: `--ngen`, `--nystrom`, `--nychoice`
- **Execution context**: `--rep`, `--start_seed`, `--outdir`
- **Diagnostics**: `--log_memory` (enables hypervolume memory hints), `--memory_log_every` (accepted for compatibility)

Compatibility notes:

- `--use_partial` is accepted but not supported in the current pip-only flow.
- `--max_nsga_multiplier`, `--threshold`, and `--memory_log_every` are accepted for compatibility with older scripts.

## Practical tuning workflow

1. Start with modest values (`ntrain`, `iters`, `q`) and confirm logs/results.
2. Scale quality by increasing `iters` and/or `q`.
3. Increase `nobj` / `dim` only as needed for experiment fidelity.
4. Tune runtime with `ngen`, thread count (`SLURM_CPUS_PER_TASK`), and array parallelism.
5. Tune memory with `ntrain`, `iters`, `q`, and objective count.

For detailed parameter-level guidance:

- See `scaling_memory.md` for RAM behavior and OOM mitigation.
- See `scaling_runtime.md` for wall-time behavior and throughput tuning.
