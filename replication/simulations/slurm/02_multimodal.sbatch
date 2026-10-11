#!/bin/bash
# Figure 2: votes, survey and text fitted jointly; joint and single-modality readouts.
#SBATCH --job-name=sims_02_mm
#SBATCH --partition=medium_cpu
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/home/Gauthier/simulations/logs/%x.%j.log
source /home/Gauthier/simulations/slurm/env.sh
$ENVPY -u 02_multimodal.py
echo WORKER_DONE
