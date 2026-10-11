#!/bin/bash
# Wave 2: the expanding-window evaluation. For every window (w2_common.windows: ends FIRST_END, FIRST_END+STEP, ...,
# training to end-5, validation the last 5 years, forecasting the next STEP years): one GPU job fitting the joint
# model at every lam and seed in sequence (w2_fit_joint.py; lam 0 = our topic model alone; time limit scaled with the
# window's training months, H200 partitions of 1 and 3 days) and one CPU job fitting LDA for every seed in sequence
# (w2_fit_lda.py, gtm env, at most LDA_DOCS training records); then the forecasts and the evaluation (w2_forecast.py)
# once all have ended (afterany: it skips missing seeds with a warning). 2 jobs per window + 1: the cluster caps
# submitted jobs at ~30 and running ones at 10 per user.
# Run on the cluster from /scratch/Gauthier/app3_wsj/run:
#   DTM=/scratch/Gauthier/app3_wsj/data/dtm_w2 [GRID=k50 K=50 LAMS="0 1e2 1e3 1e4 1e5" SEEDS="0 1 2"
#    FIRST_END=1979 STEP=3 LAST_YEAR=2011 COMMON=""] bash slurm/w2_windows.sh
set -euo pipefail
cd /scratch/Gauthier/app3_wsj/run
DTM=${DTM:?set DTM to the wave-2 document-term matrix directory}
GRID=${GRID:-k50}
K=${K:-50}
LAMS=${LAMS:-"0 1e2 1e3 1e4 1e5"}
SEEDS=${SEEDS:-"0 1 2"}
FIRST_END=${FIRST_END:-1979}
STEP=${STEP:-3}
LAST_YEAR=${LAST_YEAR:-2011}
COMMON=${COMMON:-""}
LDA_DOCS=${LDA_DOCS:-1000000}
START_YEAR=${START_YEAR:-1947}
mkdir -p tmp/w2_$GRID
ids=()
for ((end = FIRST_END; end < LAST_YEAR; end += STEP)); do
  args=tmp/w2_$GRID/joint_e$end.txt
  : > $args
  for seed in $SEEDS; do
    for lam in $LAMS; do
      echo "--dtm-dir $DTM --grid $GRID --window-end $end --lam $lam --n-topics $K --seed $seed $COMMON" >> $args
    done
  done
  n=$(wc -l < $args)
  train_months=$(( (end - 5 - START_YEAR + 1) * 12 ))
  per_line=$(( 15 + 20 * train_months / 700 ))  # minutes, with margin (wave 1: ~5 min at 192 months)
  ids+=($(sbatch --parsable --partition=gpuh200,long_gpuh200 --exclude=gnode06 --time=$(( n * per_line + 30 )):00 \
    --job-name=w2_joint_e$end --export=ALL,STEP=w2_fit_joint.py,ARGS_FILE=/scratch/Gauthier/app3_wsj/run/$args \
    slurm/gpu_batch.sbatch))
  ids+=($(sbatch --parsable --partition=defq --time=24:00:00 --cpus-per-task=16 --mem=128G --job-name=w2_lda_e$end \
    --export=ALL,STEP=w2_fit_lda.py,ARGS="--dtm-dir $DTM --grid $GRID --window-end $end --n-topics $K --seeds $SEEDS --train-docs $LDA_DOCS --workers 16",PY=/home/Gauthier/.conda/envs/gtm/bin/python \
    slurm/cpu.sbatch))
  echo "window ending $end: joint ${ids[-2]} (${n} fits, ${per_line} min each), LDA ${ids[-1]}"
done
dep=$(IFS=:; echo "${ids[*]}")
ev=$(sbatch --parsable --dependency=afterany:$dep --partition=defq --time=10:00:00 --job-name=w2_forecast \
  --export=ALL,STEP=w2_forecast.py,ARGS="--grid $GRID --dtm-dir $DTM --lams $LAMS --seeds $SEEDS --first-end $FIRST_END --step $STEP --last-year $LAST_YEAR" \
  slurm/cpu.sbatch)
echo "$ev w2_forecast"
