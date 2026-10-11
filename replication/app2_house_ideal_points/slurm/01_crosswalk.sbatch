#!/bin/bash
# Step 01: GST speaker ids -> Voteview members (minutes, CPU). Uses the Application 1 data env
# (pandas + pyarrow, needed to read the govinfo speeches for the check).
#SBATCH --job-name=app2_01_xwalk
#SBATCH --partition=medium_cpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=/home/Gauthier/app2_house_ideal_points/logs/%x.%j.log
set -euo pipefail
ulimit -t unlimited
APP2=/home/Gauthier/app2_house_ideal_points
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
cd $APP2/code
/home/Gauthier/envs/platforms_policies/bin/python -u 01_crosswalk.py
echo WORKER_DONE
