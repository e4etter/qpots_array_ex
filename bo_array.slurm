#!/bin/bash 
# Simple Slurm job-array launcher for one REP per array element.
#
# Submit with:
#   sbatch bo_array.slurm
#
# Notes:
# - This script intentionally avoids MPI.
# - Each array task runs one independent repetition.

#SBATCH -J qpots_dtlz2_array
#SBATCH -A akr6198_sc_default
#SBATCH -p sla-prio
# Default array range is 10 reps (0-9). To set programmatically at submit time:
#   REPS=10; sbatch --array=0-$((REPS-1)) bo_array.slurm
# The submit-time --array value overrides this directive.
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH -t 24:00:00
#SBATCH --output=logs/%A/%x_%A_%a.out
#### #SBATCH --error=logs/%A/%x_%A_%a.err

set -euo pipefail

SECONDS=0
echo "Starting array task on $(hostname) at $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

# Use the directory where `sbatch` was invoked.
# This avoids Slurm spool paths (e.g., /var/spool/slurm/...) that may be unwritable.
BASE_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
echo "BASE_DIR = ${BASE_DIR}"
cd "${BASE_DIR}"

# Create results and log directories.
RESULTS_DIR="${BASE_DIR}/results_${SLURM_ARRAY_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

# Set thread controls to avoid oversubscription.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Set thread affinity controls (often helpful on HPC).
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

module purge
module load anaconda/2023.09

# Activate the pre-built local environment.
# Note: conda activation hooks may read optional vars (e.g., MKL_INTERFACE_LAYER).
# With `set -u`, unset optional vars can abort activation; disable nounset only here.
set +u
conda activate ./env
set -u

# Check if the environment is ready.
python -c "import torch, botorch, gpytorch, pymoo" >/dev/null 2>&1 || {
  echo "ERROR: Required Python packages are missing in the environment." >&2
  echo "Build/repair the environment first (see README.md)." >&2
  exit 1
}

# Example run (edit flags to match your experiment).
# - design dim, nobj, ntrain, iters, q below must stay self-consistent with `--ref_point`.
# - NSGA population is 100*dim*multiplier but capped so pop_size*(nobj+ncons) <= 21201 (SobolQMC)
# - one repetition per array index
NOBJ=3
REF_POINT_VALUE=-100
REF_POINT_ARGS=()
for ((i=0; i<NOBJ; i++)); do
  REF_POINT_ARGS+=("${REF_POINT_VALUE}")
done


# Run the optimization.
set -x

python run_one_rep.py \
  --rep "${SLURM_ARRAY_TASK_ID}" \
  --outdir "${RESULTS_DIR}" \
  --start_seed 1023 \
  --func dtlz2 \
  --dim 30 \
  --nobj "${NOBJ}" \
  --ncons 0 \
  --ntrain 50 \
  --iters 50 \
  --q 4 \
  --ngen 50 \
  --max_nsga_multiplier 2 \
  --ref_point "${REF_POINT_ARGS[@]}" \
  "--log_memory" \
  --memory_log_every 1

set +x

# Calculate duration in hours, minutes, seconds
duration=$SECONDS
hrs=$((duration / 3600))
mins=$(( (duration % 3600) / 60 ))
secs=$((duration % 60))

printf "Time elapsed: %02d:%02d:%02d\n" ${hrs} ${mins} ${secs}

echo ""
echo "Finished array task ${SLURM_ARRAY_TASK_ID} at $(date)"
echo ""
