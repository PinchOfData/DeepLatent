#!/bin/bash
# Run one GPU step of application 3:   sbatch --export=ALL,STEP=03_fit_joint.py,ARGS="--tag t --lam 1e4" slurm/gpu.sbatch
#SBATCH --job-name=app3_gpu
#SBATCH --partition=gpuh200,medium_gpuh200,medium_gpunew
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/scratch/Gauthier/app3_wsj/run/logs/%x.%j.log
set -euo pipefail
ulimit -t unlimited
APP3=/scratch/Gauthier/app3_wsj/run  # not /home: GPU and CPU nodes see different homes since 2026-10-10
export PYTHONPATH=$APP3/code PYTHONUNBUFFERED=1 MPLCONFIGDIR=$APP3/tmp/mpl
cd $APP3/code
PY=${PY:-/home/Gauthier/.conda/envs/deeplatent/bin/python}
echo "JOB $SLURM_JOB_ID on $(hostname): STEP=$STEP ARGS=${ARGS:-}"
$PY -c "import torch; print('torch', torch.__version__, '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
md5sum ./*.py
# shellcheck disable=SC2086
$PY -u "$STEP" ${ARGS:-}
echo WORKER_DONE
