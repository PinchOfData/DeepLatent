#!/bin/bash
# Steps 02-03 over all 103 (source, congress) tasks of common.ALL_TASKS (bound 43-111, daily 97-114,
# govinfo 103-118), split across 10 workers of 8 cores (at most 10 jobs run per user) by
# common.worker_tasks. Cleaning runs once: finished tasks are skipped on a rerun. Step 03 can be
# rerun alone with another vocabulary (03_count.sbatch).
#SBATCH --job-name=app1_02_clean
#SBATCH --partition=defq
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-9
#SBATCH --output=/home/Gauthier/app1_us_congress/logs/%x.%A_%a.log
source /home/Gauthier/app1_us_congress/slurm/env.sh
TASKS=$(python -c "import common; print(' '.join(map(str, common.worker_tasks($SLURM_ARRAY_TASK_ID, 10))))")
echo "worker $SLURM_ARRAY_TASK_ID tasks: $TASKS"
for t in $TASKS; do
  python -u 02_clean_speeches.py --task "$t" --n-process "$SLURM_CPUS_PER_TASK"
  python -u 03_count_terms.py --task "$t"
done
echo WORKER_DONE
