#!/bin/bash
# Step 02: House units x (votes, speech term counts), 43rd-118th Congress (CPU, minutes).
#SBATCH --job-name=app2_02_build
#SBATCH --partition=medium_cpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --output=/home/Gauthier/app2_house_ideal_points/logs/%x.%j.log
set -euo pipefail
ulimit -t unlimited
APP2=/home/Gauthier/app2_house_ideal_points
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
cd $APP2/code
/home/Gauthier/.conda/envs/deeplatent/bin/python -u 02_build_house.py
echo WORKER_DONE
