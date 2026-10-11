#!/bin/bash
# Step 09: party-specific language within topics by era, for a fitted model in results/$TAG (needs gtm.ckpt).
#   sbatch --export=ALL,TAG=sage_congress_party_l1-0_fixed_kmeans_step20000 slurm/09_party_language.sbatch
#SBATCH --job-name=app1_09_party_era
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/home/Gauthier/app1_us_congress/logs/%x.%j.log
set -euo pipefail
ulimit -t unlimited
APP1=/home/Gauthier/app1_us_congress
export PYTHONPATH=$APP1/pkg:$APP1/code PYTHONUNBUFFERED=1 MPLCONFIGDIR=$APP1/tmp/mpl
cd $APP1/code
echo "JOB $SLURM_JOB_ID on $(hostname): TAG=$TAG"
/home/Gauthier/.conda/envs/deeplatent/bin/python -u 09_party_language_by_era.py --tag "$TAG" ${ERA_ARGS:-}
echo WORKER_DONE
