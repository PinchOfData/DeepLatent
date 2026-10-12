#!/bin/bash
# Step 10: House roles of the 103rd-118th (chairs, ranking members, leaders, floor-manager phrases).
# Runs in Application 1's environment (pandas + pyarrow + pyyaml; the deeplatent env has no pyarrow).
#   sbatch slurm/10_house_roles.sbatch
#SBATCH --job-name=app2_10_roles
#SBATCH --partition=medium_cpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --output=/home/Gauthier/app2_house_ideal_points/logs/%x.%j.log
set -euo pipefail
ulimit -t unlimited
APP2=/home/Gauthier/app2_house_ideal_points
export PYTHONPATH=$APP2/code PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
cd $APP2/code
/home/Gauthier/envs/platforms_policies/bin/python -u 10_house_roles.py
echo WORKER_DONE
