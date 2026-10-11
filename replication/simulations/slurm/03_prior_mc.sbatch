#!/bin/bash
# Figure 3: 1,000 Monte Carlo replications, 100 per task (resumable: rerun to finish a task).
#SBATCH --job-name=sims_03_mc
#SBATCH --partition=medium_cpu
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-9
#SBATCH --output=/home/Gauthier/simulations/logs/%x.%A_%a.log
source /home/Gauthier/simulations/slurm/env.sh
$ENVPY -u 03_prior_mc.py --task "$SLURM_ARRAY_TASK_ID" --per-task 100 --prior-lr 1e-3 --steps 10000,15000,20000
echo WORKER_DONE
