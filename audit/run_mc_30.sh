#!/bin/bash
# Overnight 30-rep Monte Carlo for the topic->outcome coefficient. Prepared to launch as-is:
#     bash audit/run_mc_30.sh
#
# Setup (all validated on a single rep):
#   JOINT    : GTM, labels_in_encoder, covariate prior mean + LEARNED Sigma, learned sigma2_y, linear head.
#   TWO-STEP : unsupervised GTM under a fixed logistic STANDARD normal prior N(0, I) (update_prior=False,
#              learn_prior_cov=False -> no learned-Sigma scale drift), then OLS of y on theta_hat.
#   Topics use ANCHOR WORDS (10/topic) so the 5 topics separate cleanly at N=10k (kills the topic-3/4 bleed).
#   sigma_y = 1.0 (MC_SIGMA now actually wired). num_workers=0 keeps WSL RAM flat (num_workers=4 worker/shm
#   churn was crashing the VM). Per-rep del/empty_cache frees GPU between reps.
#
# Read the JOINT at its ~8k bias-min sweet spot (E[c] crosses 1 ~8-9k, then overshoots -- U-shape); the
# TWO-STEP is stably attenuated (~c=0.86) at every checkpoint. Single-rep preview @8k: joint mab=0.046
# (=oracle floor) vs two-step mab=0.259 (~5.6x worse).
#
# Runtime: ~15 min/rep (joint+two-step to 10k steps each) x 30 ~= 7-8 hours.
# Outputs: results -> audit/results_mc_30.json (saved after every rep) | run log + mem log timestamped.

cd /mnt/c/Users/Gauthier/Desktop/DeepLatent || exit 1
PY=/home/gauthier/miniconda3/envs/deeplatent/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=audit/mc30_${STAMP}.log
MEMLOG=audit/mc30_${STAMP}.mem

if pgrep -f "experiment_mc_calibrate" >/dev/null; then
  echo "WARNING: an experiment_mc_calibrate process is already running (GPU allows one at a time). Aborting."
  exit 1
fi

# lightweight safety monitor (RAM / #python procs / shm) -> MEMLOG, ~12h then self-stops
( for _ in $(seq 1 720); do
    echo "$(date +%H:%M:%S) used=$(free -m | awk '/Mem:/{print $3}')MB py=$(pgrep -c python) shm=$(df -m /dev/shm | awk 'NR==2{print $3}')MB" >> "$MEMLOG"
    sleep 60
  done ) &
MON=$!

echo "launched 30-rep MC at $STAMP -> log:$LOG  results:audit/results_mc_30.json  mem:$MEMLOG"
MC_REPS=30 MC_N=10000 MC_HIDDEN="[128]" MC_COMP=10 MC_SIGMA=1.0 \
MC_LR=5e-3 MC_PRIOR_LR=5e-4 MC_BATCH=1024 MC_ANCHOR=10 \
MC_CKPTS="[6000,7000,8000,9000,10000]" \
MC_OUT="audit/results_mc_30.json" MC_BASE_SEED=1000 \
"$PY" audit/experiment_mc_calibrate.py 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}

kill "$MON" 2>/dev/null
echo "DONE (exit $RC). results -> audit/results_mc_30.json | log -> $LOG | mem -> $MEMLOG"
exit "$RC"
