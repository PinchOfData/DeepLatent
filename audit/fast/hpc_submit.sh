#!/bin/bash
# Sync the repo to the HPC and submit the fast-iteration jobs. Usage:
#   bash audit/fast/hpc_submit.sh scans      # ridge scans for candidate DGPs + harness smoke test
#   bash audit/fast/hpc_submit.sh run <tag> <extra harness args...>   # one joint + two-step run
#   bash audit/fast/hpc_submit.sh fetch      # pull results/fast back into audit/fast/hpc_results/
set -euo pipefail
HOST=${HPC_HOST:-bocconi-hpc2}
LOCAL=/mnt/c/Users/Gauthier/Desktop/DeepLatent
REMOTE=/scratch/Gauthier/deeplatent/DeepLatent
RES=/scratch/Gauthier/deeplatent/results/fast
SB=audit/fast/hpc_fast.sbatch

sync_repo() {
  rsync -az --exclude '.git' --exclude 'old' --exclude 'src2' --exclude 'dist' --exclude '.conda' \
    --exclude 'papers' --exclude '__pycache__' --exclude '*.ckpt' --exclude 'logs' --exclude 'figures' \
    --exclude 'tables' --exclude 'notebooks' --exclude '*.mem' --exclude '*.log' --exclude '*.npz' \
    --exclude 'audit/hpc_results*' --exclude 'audit/fast/hpc_results' \
    "$LOCAL/" "$HOST:$REMOTE/"
  ssh "$HOST" "mkdir -p /scratch/Gauthier/deeplatent/logs_slurm $RES"
}

submit() {  # name, partition, cpus, time, command
  ssh "$HOST" "cd $REMOTE && sbatch -J $1 -p $2 -c $3 -t $4 --export=ALL,GTM_CMD=\"$5\" $SB"
}

case "${1:-}" in
  scans)
    sync_repo
    submit smoke_plugin defq 4 00:30:00 "audit/experiment_gtm_pilot.py --n 600 --anchors 5 --prior-sd-scale 2 --hidden 32 --batch 64 --checkpoints 40 --posterior-samples 64 --diagnostic-docs 32 --threads 4 --out $RES/smoke_plugin.json"
    submit scan_a5_sd1 defq 4 02:00:00 "audit/fast/gtm_ridge_scan.py --words 25 --anchors 5 --anchor-logit 4 --prior-sd-scale 1.0 --scales 0.85 1.15 1.3"
    submit scan_a5_sd2 defq 4 02:00:00 "audit/fast/gtm_ridge_scan.py --words 25 --anchors 5 --anchor-logit 4 --prior-sd-scale 2.0 --scales 0.85 1.15 1.3"
    submit scan_a5_sd25 defq 4 02:00:00 "audit/fast/gtm_ridge_scan.py --words 25 --anchors 5 --anchor-logit 4 --prior-sd-scale 2.5 --scales 0.85 1.15 1.3"
    submit oracle_a5_sd2 defq 4 02:00:00 "audit/fast/gtm_oracle_truthonly.py --n 10000 --words 25 --anchors 5 --anchor-logit 4 --prior-sd-scale 2.0 --out $RES/oracle_a5_sd2_w25.json"
    ;;
  run)
    tag=$2; shift 2
    sync_repo
    submit "run_$tag" defq 8 12:00:00 "audit/experiment_gtm_pilot.py --n 10000 --words 25 --hidden 128 --lr 5e-3 --prior-lr 5e-4 --batch 1024 --checkpoints 2000 4000 8000 16000 24000 --threads 8 --device cpu $* --out $RES/run_$tag.json"
    ;;
  fetch)
    mkdir -p "$LOCAL/audit/fast/hpc_results"
    rsync -az --exclude '*.ckpt' "$HOST:$RES/" "$LOCAL/audit/fast/hpc_results/"
    rsync -az "$HOST:/scratch/Gauthier/deeplatent/logs_slurm/" "$LOCAL/audit/fast/hpc_results/logs/" 2>/dev/null || true
    ;;
  *) echo "usage: $0 {scans|run <tag> [harness args]|fetch}"; exit 1;;
esac
