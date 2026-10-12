#!/bin/bash
# Step 08: words of a saved step-05 model (loadings, encoder effects; CPU, minutes).
#   WORDS_ARGS="--tag votes_text_103_118_baselines_v2 --step 80000" sbatch --export=ALL slurm/08_words_from_model.sbatch
#SBATCH --job-name=app2_08_words_model
#SBATCH --partition=medium_cpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/Gauthier/app2_house_ideal_points/logs/%x.%j.log
set -euo pipefail
ulimit -t unlimited
APP2=/home/Gauthier/app2_house_ideal_points
export PYTHONPATH=$APP2/pkg:$APP2/code PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
cd $APP2/code
# shellcheck disable=SC2086
/home/Gauthier/.conda/envs/deeplatent/bin/python -u 08_words_from_model.py ${WORDS_ARGS:-}
echo WORKER_DONE
