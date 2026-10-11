#!/bin/bash
# Step 01: every raw speech -> data/speeches/<source>/<congress>.parquet (4 tasks).
#SBATCH --job-name=app1_01_build
#SBATCH --partition=defq
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --array=0-3
#SBATCH --output=/home/Gauthier/app1_us_congress/logs/%x.%A_%a.log
source /home/Gauthier/app1_us_congress/slurm/env.sh
case $SLURM_ARRAY_TASK_ID in
  0) python -u 01_build_speeches.py --hein bound --congresses 43-77 ;;
  1) python -u 01_build_speeches.py --hein bound --congresses 78-111 ;;
  2) python -u 01_build_speeches.py --hein daily --congresses 97-114 ;;
  3) python -u 01_build_speeches.py --govinfo --congresses 103-118 ;;
esac
echo WORKER_DONE
