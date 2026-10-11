#!/bin/bash
# Run several fits of one step in one GPU job, one line of arguments each (a file of lines in ARGS_FILE):
#   sbatch --export=ALL,STEP=03_fit_joint.py,ARGS_FILE=/scratch/.../args.txt slurm/gpu_batch.sbatch
#SBATCH --job-name=app3_gpub
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
echo "JOB $SLURM_JOB_ID on $(hostname): STEP=$STEP ARGS_FILE=$ARGS_FILE"
md5sum ./*.py
while IFS= read -r line; do
  [ -z "$line" ] && continue
  echo "=== $line"
  # shellcheck disable=SC2086
  $PY -u "$STEP" $line
done < "$ARGS_FILE"
echo WORKER_DONE
