#!/bin/bash
# Generic SLURM runner for the GTM fast-iteration scripts.
#   sbatch --export=ALL,GTM_CMD="audit/fast/gtm_ridge_scan.py --words 25 --anchors 5" \
#          -J scan_a5 audit/fast/hpc_fast.sbatch
# GTM_CMD is a python command line relative to the repo root (script + args).
# Results/logs: /scratch/Gauthier/deeplatent/results/fast/ and logs_slurm/.
#SBATCH --job-name=gtm_fast
#SBATCH --partition=defq
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/Gauthier/deeplatent/logs_slurm/%x.%j.log

set -euo pipefail
ulimit -t unlimited
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
WORK="${SLURM_TMPDIR:-/tmp}/gtm_fast_${SLURM_JOB_ID}"
mkdir -p "$WORK" /scratch/Gauthier/deeplatent/results/fast
export NUMBA_CACHE_DIR="$WORK/numba" MPLCONFIGDIR="$WORK/mpl"
REPO=/scratch/Gauthier/deeplatent/DeepLatent
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
ENVPY=$HOME/.conda/envs/deeplatent/bin/python
echo "JOB $SLURM_JOB_ID on $(hostname): $GTM_CMD"
$ENVPY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'threads', torch.get_num_threads())"
# shellcheck disable=SC2086
$ENVPY -u $GTM_CMD
echo "WORKER_DONE $SLURM_JOB_ID"
