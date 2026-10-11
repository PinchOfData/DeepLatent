#!/bin/bash
# The SAGE-decoder comparison of 2026-10-08: K = 20, the supplemented vocabulary, the speeches and settings
# of the linear-decoder run vsupp_anneal5k_plr1e-4 (KL annealing over the first 5,000 steps, prior lr 1e-4,
# 20,000 steps, a checkpoint every 1,000), for every combination of
#   --interactions none | party | congress_party   (topic-specific party deviations)
#   --l1 0 | 100                                     (L1 penalty on the deviations, in word counts)
#   --background fixed | learned                     (background log frequencies)
# Each job fits (checkpoints in scratch) and reviews every checkpoint from step 5,000 (08_checkpoint_topics.py).
# INIT=kmeans starts the topics from k-means clusters of the speeches (tags end in _kmeans; reviewed from step
# 1,000): from the default random start every combination collapsed to 9-16 used topics.
# Run on the cluster from ~/app1_us_congress:
#   [INTERS="none party"] [L1S="0 100"] [BGS="fixed learned"] [INIT=kmeans] bash slurm/sage_grid.sh [sbatch options]
set -euo pipefail
cd /home/Gauthier/app1_us_congress
V=/home/Gauthier/app1_us_congress/inputs/vocab_us_supplemented.txt
P=gpuh200,medium_gpuh200,gpunew,medium_gpunew
for inter in ${INTERS:-none party congress_party}; do
  for l1 in ${L1S:-0 100}; do
    for bg in ${BGS:-fixed learned}; do
      tag="sage_${inter}_l1-${l1}_${bg}"
      args="--n-topics 20 --per-cell 0 --val-frac 0 --kl-anneal 0 5000 --num-steps 20000 --checkpoint-every 1000"
      args="$args --log-every 1000 --posterior-samples 10 --prior-lr 1e-4 --vocab $V"
      args="$args --decoder sage --interactions $inter --l1 $l1 --background $bg --init ${INIT:-random}"
      review=""
      if [ "${INIT:-random}" = kmeans ]; then tag="${tag}_kmeans"; review="--min-step 1000"; fi
      id=$(sbatch --parsable --partition=$P --time=04:00:00 "$@" \
        --export=ALL,TAG=$tag,REVIEW_CHECKPOINTS=1,SKIP_EXHIBITS=1,REVIEW_ARGS="$review",FIT_ARGS="$args" slurm/05_fit.sbatch)
      echo "$id $tag"
    done
  done
done
