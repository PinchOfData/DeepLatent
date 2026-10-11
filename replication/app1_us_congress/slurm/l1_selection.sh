#!/bin/bash
# Choosing the L1 penalty of the SAGE decoder (topic x party by congress, k-means start, fixed background) by how
# well the fitted party language predicts new speakers: 10% of the speakers of every congress x party are held out
# (--val-by speaker) and the held-out per-speech loss (-ELBO, whole held-out set) is recorded every 5,000 steps.
# One full 20,000-step fit per K and L1; the penalty is chosen separately for each K (lowest held-out loss at step
# 20,000), then refit on all speeches with exhibits.
# Run on the cluster from ~/app1_us_congress:   [KS="20 50"] [L1S="0 3 10 30 100"] bash slurm/l1_selection.sh
set -euo pipefail
cd /home/Gauthier/app1_us_congress
V=/home/Gauthier/app1_us_congress/inputs/vocab_us_supplemented.txt
P=gpuh200,medium_gpuh200,gpunew,medium_gpunew
for k in ${KS:-20 50}; do
  for l1 in ${L1S:-0 3 10 30 100}; do
    tag="sage_cp_k${k}_l1-${l1}_heldout"
    args="--n-topics $k --per-cell 0 --val-frac 0.1 --val-by speaker --validate-every 5000 --patience 1000"
    args="$args --kl-anneal 0 5000 --num-steps 20000 --log-every 1000 --posterior-samples 10 --prior-lr 1e-4 --vocab $V"
    args="$args --decoder sage --interactions congress_party --l1 $l1 --background fixed --init kmeans"
    id=$(sbatch --parsable --partition=$P --time=06:00:00 "$@" \
      --export=ALL,TAG=$tag,SKIP_EXHIBITS=1,FIT_ARGS="$args" slurm/05_fit.sbatch)
    echo "$id $tag"
  done
done
