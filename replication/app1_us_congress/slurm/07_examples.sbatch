#!/bin/bash
# Step 07: example speeches with the largest share of each topic, for a fitted model in results/$TAG.
# Reads the raw speech text (parquet), so it runs in the P&P environment (pyarrow).
#   sbatch --export=ALL,TAG=k20_plr1e-4_step6000[,EXAMPLE_ARGS="--per-topic 8"] slurm/07_examples.sbatch
#SBATCH --job-name=app1_07_examples
#SBATCH --partition=defq
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=/home/Gauthier/app1_us_congress/logs/%x.%j.log
set -euo pipefail
ulimit -t unlimited
APP1=/home/Gauthier/app1_us_congress
export PYTHONPATH=$APP1/code PYTHONUNBUFFERED=1
cd $APP1/code
# shellcheck disable=SC2086
/home/Gauthier/envs/platforms_policies/bin/python -u 07_topic_examples.py --tag "$TAG" ${EXAMPLE_ARGS:-}
echo WORKER_DONE
