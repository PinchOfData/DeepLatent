#!/bin/bash
# Step 04: GST leave-out estimator, random-party series, subsampling intervals. Reads only npz and
# csv.gz files, so it runs in the deeplatent environment (which has matplotlib for the figure).
#   sbatch [--export=ALL,VOCAB=/path/vocab.txt] slurm/04_gst.sbatch
#SBATCH --job-name=app1_04_gst
#SBATCH --partition=defq
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --output=/home/Gauthier/app1_us_congress/logs/%x.%j.log
set -euo pipefail
ulimit -t unlimited
APP1=/home/Gauthier/app1_us_congress
export PYTHONPATH=$APP1/code PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MPLCONFIGDIR=$APP1/tmp/mpl
cd $APP1/code
/home/Gauthier/.conda/envs/deeplatent/bin/python -u 04_gst_leave_out.py ${VOCAB:+--vocab "$VOCAB"}
echo WORKER_DONE
