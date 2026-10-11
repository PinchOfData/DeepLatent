# Sourced by every job: CPU settings, the deeplatent env, and a check that deeplatent is imported
# from the copy synced to ~/simulations/pkg (aborts otherwise; logs the md5 of every module).
set -euo pipefail
ulimit -t unlimited  # the cluster's default 600 s CPU-time limit kills long jobs
SIMS=/home/Gauthier/simulations
ENVPY=/home/Gauthier/.conda/envs/deeplatent/bin/python
export PYTHONPATH=$SIMS/pkg:$SIMS/code PYTHONUNBUFFERED=1 MPLCONFIGDIR=$SIMS/tmp/mpl
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
mkdir -p "$SIMS/logs" "$SIMS/results" "$SIMS/tmp/mpl"
cd "$SIMS/code"
echo "JOB ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-none} on $(hostname)"
$ENVPY - <<'PY'
import hashlib, pathlib
import deeplatent, torch
pkg = pathlib.Path(deeplatent.__file__).resolve().parent
want = pathlib.Path("/home/Gauthier/simulations/pkg/deeplatent").resolve()
assert pkg == want, f"wrong deeplatent: {pkg} (expected {want})"
print("deeplatent", pkg, "| torch", torch.__version__)
for f in sorted(pkg.glob("*.py")):
    print("md5", hashlib.md5(f.read_bytes()).hexdigest(), f.name)
PY
