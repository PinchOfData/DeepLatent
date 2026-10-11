#!/bin/bash
# Figure 1: one fit per likelihood (task 0-3 = embedding, survey, bow, vote).
#SBATCH --job-name=sims_01_lik
#SBATCH --partition=medium_cpu
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --array=0-3
#SBATCH --output=/home/Gauthier/simulations/logs/%x.%A_%a.log
source /home/Gauthier/simulations/slurm/env.sh
LIKS=(embedding survey bow vote)
$ENVPY -u 01_likelihoods.py --likelihood "${LIKS[$SLURM_ARRAY_TASK_ID]}"
echo WORKER_DONE
