#!/bin/bash
# Run one CPU step of application 3:   sbatch --export=ALL,STEP=01_flag_records.py,ARGS="" slurm/cpu.sbatch
#SBATCH --job-name=app3_cpu
#SBATCH --partition=short_cpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/Gauthier/app3_wsj/run/logs/%x.%j.log
set -euo pipefail
ulimit -t unlimited  # every partition starts jobs with a 600 s CPU-time soft limit (HPC.md)
APP3=/scratch/Gauthier/app3_wsj/run  # not /home: GPU and CPU nodes see different homes since 2026-10-10
export PYTHONPATH=$APP3/code PYTHONUNBUFFERED=1 MPLCONFIGDIR=$APP3/tmp/mpl
cd $APP3/code
echo "JOB $SLURM_JOB_ID on $(hostname): STEP=$STEP ARGS=${ARGS:-}"
# shellcheck disable=SC2086
${PY:-/home/Gauthier/.conda/envs/deeplatent/bin/python} -u "$STEP" ${ARGS:-}
echo WORKER_DONE
