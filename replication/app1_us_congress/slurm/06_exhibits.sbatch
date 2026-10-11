#!/bin/bash
# Step 06 alone: the exhibits of a fitted model in results/$TAG (needs gtm.ckpt, theta.npy, docs.csv.gz).
#   sbatch --export=ALL,TAG=sage_congress_party_l1-0_fixed_kmeans_step20000[,EXHIBIT_ARGS="--mc-reps 100"] slurm/06_exhibits.sbatch
#SBATCH --job-name=app1_06_exhibits
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/home/Gauthier/app1_us_congress/logs/%x.%j.log
set -euo pipefail
ulimit -t unlimited
APP1=/home/Gauthier/app1_us_congress
export PYTHONPATH=$APP1/pkg:$APP1/code PYTHONUNBUFFERED=1 MPLCONFIGDIR=$APP1/tmp/mpl
cd $APP1/code
echo "JOB $SLURM_JOB_ID on $(hostname): TAG=$TAG EXHIBIT_ARGS=${EXHIBIT_ARGS:-}"
# shellcheck disable=SC2086
/home/Gauthier/.conda/envs/deeplatent/bin/python -u 06_exhibits.py --tag "$TAG" ${EXHIBIT_ARGS:-}
echo WORKER_DONE
