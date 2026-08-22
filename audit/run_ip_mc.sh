#!/bin/bash
# Overnight 3x30-rep Monte Carlo: IdealPointNN consistency & CI coverage.
# Design: IDEALPOINT_CONSISTENCY.md | harness: audit/experiment_ip_pilot.py
#
# Calibration (single-rep trajectories, results_ip_cal_*.json + results_ip_pilot.json):
#   JOINT plateaus AT truth with no U-overshoot at every n (n=1000 flat 16k-32k;
#   n=2000 flat 16k-24k; n=16000 converged by 24k) -> train to 24k, read @16k & 24k.
#   TWO-STEP is flat at every checkpoint at every n -> read @8k only.
# J=25 fixed (kappa>0 regime), n in {1000, 4000, 16000}, 30 reps each.
# ~17h total on the MX550; JSONs update after EVERY rep -> partial results usable anytime.
# Kill safely with: pkill -f experiment_ip_pilot; rerun a cell by rerunning its line.
cd /mnt/c/Users/Gauthier/Desktop/DeepLatent || exit 1
PY=/home/gauthier/miniconda3/envs/deeplatent/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
MEMLOG=audit/ip_mc_${STAMP}.mem

# lightweight WSL-safety monitor (RAM / #python procs), ~18h then self-stops
( for i in $(seq 1 4320); do
    echo "$(date +%H:%M:%S) used=$(free -m | awk '/Mem:/{print $3}')MB py=$(pgrep -c python)"
    sleep 15
  done >> "$MEMLOG" ) &
MONPID=$!

run_cell () {  # n seed out log
  IP_J=25 IP_CKPTS='[16000,24000]' IP_CKPTS_2STEP='[8000]' IP_NS_POST=100 IP_REPS=30 \
  IP_N="$1" IP_SEED="$2" IP_OUT="$3" "$PY" audit/experiment_ip_pilot.py > "$4" 2>&1
}

run_cell 1000  4000 audit/results_ip_mc_n1000.json  audit/ip_mc_n1000.log
run_cell 4000  5000 audit/results_ip_mc_n4000.json  audit/ip_mc_n4000.log
run_cell 16000 6000 audit/results_ip_mc_n16000.json audit/ip_mc_n16000.log

kill $MONPID 2>/dev/null
echo ALL_DONE > audit/ip_mc.done
