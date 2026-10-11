#!/bin/bash
# The trade-off grid: the joint model for every prediction weight lam (0 = topics alone, the two-step model's first
# stage) and seed; the same model with the topics cut off from the prediction loss (--two-step-head: the two-step
# estimator with the joint model's own head, step selection and training); LDA (the classical two-step first stage);
# then the comparison (04), on the selected and on the final steps: once with the joint fits only (as soon as they are
# done) and once with LDA too. The joint fits run one after the other in one GPU job (about 5 minutes each).
# Run on the cluster from /scratch/Gauthier/app3_wsj/run:
#   [K=50 SEEDS="0" LAMS="..." COMMON="" GRID=k50 LDA_TAGS="lda_k50_s0"] bash slurm/lam_grid.sh
# LDA_TAGS: compare with LDA fits that already exist instead of fitting LDA again (its attention does not depend on
# the joint runs' settings; 04 applies their head inputs to it).
set -euo pipefail
cd /scratch/Gauthier/app3_wsj/run
K=${K:-50}
SEEDS=${SEEDS:-0}
LAMS=${LAMS:-"0 1 1e2 1e3 3e3 1e4 3e4 1e5 1e6 1e7"}
COMMON=${COMMON:-""}
GRID=${GRID:-k${K}}
args=tmp/grid_${GRID}.txt
: > $args
joint=() lda=() ldatags=()
for seed in $SEEDS; do
  for lam in $LAMS; do
    tag="joint_${GRID}_lam${lam}_s${seed}"
    echo "--tag $tag --lam $lam --n-topics $K --seed $seed $COMMON" >> $args
    joint+=($tag)
  done
  tag="twostephead_${GRID}_s${seed}"
  echo "--tag $tag --lam 1e4 --two-step-head --n-topics $K --seed $seed $COMMON" >> $args
  joint+=($tag)
  [ -n "${LDA_TAGS:-}" ] && continue
  tag="lda_k${K}_s${seed}"
  lda+=($(sbatch --parsable --partition=defq --time=12:00:00 --cpus-per-task=16 --mem=96G \
    --export=ALL,STEP=03_fit_lda.py,ARGS="--tag $tag --n-topics $K --seed $seed --workers 16",PY=/home/Gauthier/.conda/envs/gtm/bin/python \
    slurm/cpu.sbatch))
  ldatags+=($tag)
done
[ -n "${LDA_TAGS:-}" ] && ldatags=($LDA_TAGS)
g=$(sbatch --parsable --exclude=gnode06 --time=$(( ${#joint[@]} * 10 + 30 )):00 \
  --export=ALL,STEP=03_fit_joint.py,ARGS_FILE=/scratch/Gauthier/app3_wsj/run/$args slurm/gpu_batch.sbatch)
echo "$g joint fits: ${joint[*]}"
echo "${lda[*]} LDA: ${ldatags[*]}"
for which in selected final; do
  e1=$(sbatch --parsable --dependency=afterok:$g --time=01:00:00 \
    --export=ALL,STEP=04_evaluate.py,ARGS="--tags ${joint[*]} --which $which" slurm/cpu.sbatch)
  dep=$(IFS=:; echo "$g${lda[*]:+:${lda[*]}}")
  e2=$(sbatch --parsable --dependency=afterok:$dep --time=01:00:00 \
    --export=ALL,STEP=04_evaluate.py,ARGS="--tags ${joint[*]} ${ldatags[*]} --which $which" slurm/cpu.sbatch)
  echo "04 ($which): joint only $e1, with LDA $e2"
done
