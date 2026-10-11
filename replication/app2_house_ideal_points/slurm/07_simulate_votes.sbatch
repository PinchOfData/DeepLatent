#!/bin/bash
# Step 07 (diagnostic): semi-synthetic votes with Nokken-Poole positions as the truth, on the real
# voting pattern; writes ~/app2_house_ideal_points/data_sim/house (CPU, minutes).
#   sbatch slurm/07_simulate_votes.sbatch
#SBATCH --job-name=app2_07_sim
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
/home/Gauthier/.conda/envs/deeplatent/bin/python -u 07_simulate_votes.py --out $APP2/data_sim --seed 0
echo WORKER_DONE
