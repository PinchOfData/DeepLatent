#!/bin/bash
# Step 08 alone: review the checkpoints of an earlier fit (topics, shares, probes, t-SNE at every checkpoint).
#   sbatch --export=ALL,TAG=vsupp_anneal5k_plr1e-4,REVIEW_ARGS="--max-step 20000 --out-name checkpoint_review_probes" slurm/08_review.sbatch
#SBATCH --job-name=app1_08_review
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/home/Gauthier/app1_us_congress/logs/%x.%j.log
set -euo pipefail
ulimit -t unlimited
APP1=/home/Gauthier/app1_us_congress
export PYTHONPATH=$APP1/pkg:$APP1/code PYTHONUNBUFFERED=1 MPLCONFIGDIR=$APP1/tmp/mpl
export APP1_CHECKPOINTS=/scratch/Gauthier/app1_us_congress/checkpoints
cd $APP1/code
echo "JOB $SLURM_JOB_ID on $(hostname): TAG=$TAG REVIEW_ARGS=${REVIEW_ARGS:-}"
# shellcheck disable=SC2086
/home/Gauthier/.conda/envs/deeplatent/bin/python -u 08_checkpoint_topics.py --tag "$TAG" ${REVIEW_ARGS:-}
echo WORKER_DONE
