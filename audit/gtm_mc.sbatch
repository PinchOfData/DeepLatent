#!/bin/bash
# Submit from the root of an isolated, synchronized GTM source snapshot.
# Example (only after reviewing the laptop pilot):
# sbatch --array=0-2%3 --export=ALL,GTM_N=8000,GTM_RUN_ID=gtm_calibration audit/gtm_mc.sbatch
#SBATCH --job-name=gtm_mc
#SBATCH --partition=defq
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=gtm_mc.%A_%a.log

set -euo pipefail
ulimit -t unlimited

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

GTM_WORKDIR="${SLURM_TMPDIR:-/tmp}/gtm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$GTM_WORKDIR"
export NUMBA_CACHE_DIR="$GTM_WORKDIR/numba"
export MPLCONFIGDIR="$GTM_WORKDIR/mpl"

GTM_RUN_ID="${GTM_RUN_ID:?Set GTM_RUN_ID to a new campaign name}"
GTM_N="${GTM_N:-4000}"
GTM_WORDS="${GTM_WORDS:-25}"
GTM_BASE_SEED="${GTM_BASE_SEED:-12000}"
GTM_REP="${SLURM_ARRAY_TASK_ID:-0}"
GTM_SEED=$((GTM_BASE_SEED + GTM_REP))
GTM_RESULT_DIR="/scratch/Gauthier/deeplatent/results/$GTM_RUN_ID"
mkdir -p "$GTM_RESULT_DIR"
GTM_RESULT="$GTM_RESULT_DIR/gtm_n${GTM_N}_w${GTM_WORDS}_rep${GTM_REP}.json"
if [[ -e "$GTM_RESULT" ]]; then
    echo "Refusing to overwrite existing result: $GTM_RESULT" >&2
    exit 1
fi

cd "$SLURM_SUBMIT_DIR"
"$HOME/.conda/envs/deeplatent/bin/python" -u audit/experiment_gtm_pilot.py \
    --n "$GTM_N" --words "$GTM_WORDS" --seed "$GTM_SEED" \
    --checkpoints 4000 8000 16000 32000 --two-step-prior learned \
    --device cpu --threads 1 --out "$GTM_RESULT"

echo "WORKER_DONE n=$GTM_N words=$GTM_WORDS rep=$GTM_REP"
