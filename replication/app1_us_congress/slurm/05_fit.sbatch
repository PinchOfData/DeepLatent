#!/bin/bash
# Steps 05-06: fit the GTM on a GPU, then write the exhibits. deeplatent is imported from the copy
# synced to ~/app1_us_congress/pkg (the job aborts otherwise and logs the md5 of every module).
#   sbatch --export=ALL,TAG=full_k20,FIT_ARGS="--per-cell 0 --kl-anneal 5000 30000" slurm/05_fit.sbatch
#SBATCH --job-name=app1_05_fit
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/home/Gauthier/app1_us_congress/logs/%x.%j.log
set -euo pipefail
ulimit -t unlimited
APP1=/home/Gauthier/app1_us_congress
ENVPY=/home/Gauthier/.conda/envs/deeplatent/bin/python
export PYTHONPATH=$APP1/pkg:$APP1/code PYTHONUNBUFFERED=1 MPLCONFIGDIR=$APP1/tmp/mpl
export APP1_CHECKPOINTS=/scratch/Gauthier/app1_us_congress/checkpoints  # ~0.5 GB per checkpoint
cd $APP1/code
echo "JOB $SLURM_JOB_ID on $(hostname): TAG=$TAG FIT_ARGS=${FIT_ARGS:-} EXHIBIT_ARGS=${EXHIBIT_ARGS:-}"
$ENVPY - <<'PY'
import hashlib, pathlib
import deeplatent, torch
pkg = pathlib.Path(deeplatent.__file__).resolve().parent
want = pathlib.Path("/home/Gauthier/app1_us_congress/pkg/deeplatent").resolve()
assert pkg == want, f"wrong deeplatent: {pkg} (expected {want})"
print("deeplatent", pkg, "| torch", torch.__version__, "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
for f in sorted(pkg.glob("*.py")):
    print("md5", hashlib.md5(f.read_bytes()).hexdigest(), f.name)
PY
# shellcheck disable=SC2086
$ENVPY -u 05_fit_gtm.py --tag "$TAG" ${FIT_ARGS:-}
if [ "${REVIEW_CHECKPOINTS:-0}" = 1 ]; then  # REVIEW_CHECKPOINTS=1: topics of every checkpoint (08)
  # shellcheck disable=SC2086
  $ENVPY -u 08_checkpoint_topics.py --tag "$TAG" ${REVIEW_ARGS:-}
fi
if [ "${SKIP_EXHIBITS:-0}" != 1 ]; then  # SKIP_EXHIBITS=1: fit only (smoke tests, checkpoint runs)
  # shellcheck disable=SC2086
  $ENVPY -u 06_exhibits.py --tag "$TAG" ${EXHIBIT_ARGS:-}
fi
echo WORKER_DONE
