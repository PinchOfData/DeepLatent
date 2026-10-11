#!/bin/bash
# Step 05 (stage 2) on CPU, for when the GPU partitions are full. Same script and package
# checks as 05_fit_votes_text.sbatch; 12 compute threads plus the data-loading workers.
#   FIT_ARGS="--congresses 103-118 --steps 5000,10000" sbatch --export=ALL,TAG=votes_text_103_118_cpu slurm/05_fit_votes_text_cpu.sbatch
# (pass FIT_ARGS through the environment: sbatch --export splits its value on commas)
#SBATCH --job-name=app2_05_text_cpu
#SBATCH --partition=medium_cpu
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/home/Gauthier/app2_house_ideal_points/logs/%x.%j.log
set -euo pipefail
ulimit -t unlimited
APP2=/home/Gauthier/app2_house_ideal_points
ENVPY=/home/Gauthier/.conda/envs/deeplatent/bin/python
export PYTHONPATH=$APP2/pkg:$APP2/code PYTHONUNBUFFERED=1 MPLCONFIGDIR=$APP2/tmp OMP_NUM_THREADS=12
cd $APP2/code
echo "JOB $SLURM_JOB_ID on $(hostname): TAG=$TAG FIT_ARGS=${FIT_ARGS:-}"
$ENVPY - <<'PY'
import hashlib, pathlib
import deeplatent, torch
pkg = pathlib.Path(deeplatent.__file__).resolve().parent
want = pathlib.Path("/home/Gauthier/app2_house_ideal_points/pkg/deeplatent").resolve()
assert pkg == want, f"wrong deeplatent: {pkg} (expected {want})"
print("deeplatent", pkg, "| torch", torch.__version__, "| threads", torch.get_num_threads())
PY
# shellcheck disable=SC2086
$ENVPY -u 05_fit_votes_text.py --tag "$TAG" --device cpu ${FIT_ARGS:-}
echo WORKER_DONE
