#!/bin/bash
# Step 03 alone (e.g. another vocabulary): sbatch --export=ALL,VOCAB=/path/vocab.txt 03_count.sbatch
#SBATCH --job-name=app1_03_count
#SBATCH --partition=defq
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-9
#SBATCH --output=/home/Gauthier/app1_us_congress/logs/%x.%A_%a.log
source /home/Gauthier/app1_us_congress/slurm/env.sh
TASKS=$(python -c "import common; print(' '.join(map(str, common.worker_tasks($SLURM_ARRAY_TASK_ID, 10))))")
for t in $TASKS; do python -u 03_count_terms.py --task "$t" ${VOCAB:+--vocab "$VOCAB"}; done
echo WORKER_DONE
