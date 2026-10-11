#!/bin/bash
# Step 06: words at each end of a fit's ideal-point spectrum (CPU, minutes).
#   WORDS_ARGS="--tag votes_text_103_118_cpu --step 40000 --estimate theta_text" sbatch --export=ALL slurm/06_words.sbatch
#SBATCH --job-name=app2_06_words
#SBATCH --partition=medium_cpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=/home/Gauthier/app2_house_ideal_points/logs/%x.%j.log
set -euo pipefail
ulimit -t unlimited
APP2=/home/Gauthier/app2_house_ideal_points
export PYTHONPATH=$APP2/code PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
cd $APP2/code
for est in theta_text theta_mean; do
  # shellcheck disable=SC2086
  /home/Gauthier/.conda/envs/deeplatent/bin/python -u 06_words_by_ideal_point.py ${WORDS_ARGS:-} --estimate $est
done
echo WORKER_DONE
