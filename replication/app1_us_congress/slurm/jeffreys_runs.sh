#!/bin/bash
# SAGE decoder with SAGE's Normal-Jeffreys prior on every deviation, estimated SCHOLAR-style, trained long enough for
# Adam to bring the deviations near their unpenalized values before the precisions are refreshed: precision 1 (a
# variance-1 ridge) until step 20,000, then 1/deviation^2 refreshed every 100 steps until step 40,000. Topic x party
# by congress, k-means start, fixed background, KL annealing over the first 5,000 steps.
#   1. final fits on all speeches, K = 20 and K = 50 (checkpoints every 1,000 steps + review, exhibits, then party
#      language by era (09) and example speeches (07));
#   2. held-out-speaker checks (10% of the speakers of every congress x party held out; held-out loss at steps 20,000
#      and 40,000, the final-step one is the comparison), with and without the prior (the unpenalized fits
#      sage_cp_k*_l1-0_heldout stopped at 20,000 steps, so they are rerun to 40,000 for a like-for-like comparison).
# Run on the cluster from ~/app1_us_congress:   [KS="20 50"] bash slurm/jeffreys_runs.sh
set -euo pipefail
cd /home/Gauthier/app1_us_congress
V=/home/Gauthier/app1_us_congress/inputs/vocab_us_supplemented.txt
P=gpuh200,medium_gpuh200,gpunew,medium_gpunew
for k in ${KS:-20 50}; do
  base="--n-topics $k --per-cell 0 --kl-anneal 0 5000 --num-steps 40000 --log-every 1000 --prior-lr 1e-4 --vocab $V"
  base="$base --decoder sage --interactions congress_party --background fixed --init kmeans"
  jeff="--jeffreys --jeffreys-start 20000 --jeffreys-every 100"
  heldout="--val-frac 0.1 --val-by speaker --validate-every 20000 --patience 1000 --posterior-samples 10"

  tag="sage_cp_k${k}_jeffreys"
  fit=$(sbatch --parsable --partition=$P --time=06:00:00 \
    --export=ALL,TAG=$tag,REVIEW_CHECKPOINTS=1,REVIEW_ARGS="--min-step 5000",FIT_ARGS="$base $jeff --val-frac 0 --checkpoint-every 1000 --posterior-samples 30" \
    slurm/05_fit.sbatch)
  era=$(sbatch --parsable --dependency=afterok:$fit --partition=$P,short_gpuh200 --export=ALL,TAG=$tag slurm/09_party_language.sbatch)
  ex=$(sbatch --parsable --dependency=afterok:$fit --export=ALL,TAG=$tag slurm/07_examples.sbatch)
  echo "$fit $tag (09: $era, 07: $ex)"

  for variant in jeffreys none; do
    tag="sage_cp_k${k}_${variant}_40k_heldout"
    extra=$([ "$variant" = jeffreys ] && echo "$jeff" || echo "")
    id=$(sbatch --parsable --partition=$P --time=06:00:00 \
      --export=ALL,TAG=$tag,SKIP_EXHIBITS=1,FIT_ARGS="$base $extra $heldout" slurm/05_fit.sbatch)
    echo "$id $tag"
  done
done
