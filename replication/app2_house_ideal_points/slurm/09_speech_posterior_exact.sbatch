#!/bin/bash
# Step 09: exact speeches-only posterior of a saved step-05 model vs its speech encoder (CPU, minutes).
#   POST_ARGS="--tag votes_text_103_118_baselines_v2 --step 80000" sbatch --export=ALL slurm/09_speech_posterior_exact.sbatch
#SBATCH --job-name=app2_09_speech_post
#SBATCH --partition=medium_cpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/Gauthier/app2_house_ideal_points/logs/%x.%j.log
set -euo pipefail
ulimit -t unlimited
APP2=/home/Gauthier/app2_house_ideal_points
export PYTHONPATH=$APP2/pkg:$APP2/code PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
cd $APP2/code
# shellcheck disable=SC2086
/home/Gauthier/.conda/envs/deeplatent/bin/python -u 09_speech_posterior_exact.py ${POST_ARGS:-}
echo WORKER_DONE
