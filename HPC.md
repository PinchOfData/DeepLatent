# HPC Instructions (Bocconi) — DeepLatent

Adapted from `Desktop/alfred/instructions/hpc/workflows.md`. This file is the
DeepLatent-specific version: same cluster, plus the paths, env, and SLURM patterns
used by this project. **Always use SLURM** — never run Python on the login node.

## Connection

**SSH aliases** (in WSL `~/.ssh/config`):
- `ssh bocconi-hpc` → `lnode01-da.hpc.unibocconi.it` (user `gauthier`)
- `ssh bocconi-hpc2` → `lnode02-da.hpc.unibocconi.it` — **fallback**: added 2026-08-20
  when lnode01 started rejecting the (valid) ed25519 key while lnode02 accepted it.
  If one node refuses the key, try the other before debugging anything else.

## Project paths

| Path | Purpose |
|---|---|
| `/home/Gauthier` | home (quota ~280G soft / 300G hard; check with `quota -s`) |
| `~/.conda/envs/deeplatent` | project conda env (Python 3.11 + torch; created in HOME by request) |
| `/scratch/Gauthier/deeplatent/` | **dedicated project workdir** (BeeGFS) — repo, sbatch scripts, results, job logs |
| `/scratch/Gauthier/deeplatent/DeepLatent/` | rsync'd copy of this repo (no `.git` — the WSL copy stays canonical) |
| `/scratch/Gauthier/pip_cache`, `/scratch/Gauthier/conda_pkgs` | package caches (keep heavy caches off home) |

Sync repo changes from WSL (run locally):
```bash
rsync -az --exclude '.git' --exclude 'old' --exclude 'src2' --exclude 'dist' \
  --exclude '.conda' --exclude 'papers' --exclude '__pycache__' --exclude '*.ckpt' \
  --exclude 'logs' --exclude 'figures' --exclude 'tables' --exclude 'notebooks' \
  --exclude '*.mem' --exclude '*.log' \
  /mnt/c/Users/Gauthier/Desktop/DeepLatent/ bocconi-hpc2:/scratch/Gauthier/deeplatent/DeepLatent/
```
Fetch results back (run locally):
```bash
rsync -az bocconi-hpc2:/scratch/Gauthier/deeplatent/results/ audit/hpc_results/
```

## Job scheduler: SLURM

### Key partitions (verified 2026-08-20 — alfred's table is outdated; cluster was reorganized and now includes H200 nodes)

| Partition | Time limit | Hardware | Use case |
|-----------|------------|----------|----------|
| `defq` (default) | 1 day | CPU nodes | MC reps of small models (our nets are tiny — CPU is fine and plentiful) |
| `short_cpu` / `medium_cpu` | 1h / 6h | CPU | short CPU jobs |
| `compute` | 3 days | CPU | long CPU jobs |
| `gpua100` / `medium_gpua100` | 1d / 6h | A100 | GPU jobs (A100) |
| `gpunew` / `medium_gpunew` / `long_gpunew` | 1d / 6h / 3d | H100 | GPU jobs (H100) |
| `gpuh200` / `medium_gpuh200` / `long_gpuh200` | 1d / 6h / 3d | H200 | GPU jobs (H200) |
| `debug_cpu` / `debug_gpua100` / `debug_gpunew` / `debug_gpuh200` | 15 min | — | quick tests |

**QOS:** the account has `debug,normal`. Debug partitions require
`#SBATCH --qos=debug` — without it submission fails with
`Invalid qos specification`. Normal partitions need no explicit QOS.
There is no partition named `debug_gpu` anymore.

**QOS job caps:** ~30 submitted jobs and 10 concurrently running per user —
large job arrays queue-flow 10 at a time, and a second 30-task array cannot be
submitted until the first drains (chain them).

**⚠ 600s CPU-time SOFT limit (every partition, verified 2026-08-20):** all jobs
start with `ulimit -St` = 600 seconds of *CPU time* (not wall time). A
multi-threaded torch process burns it in ~2.5 min wall and dies with
`CPU time limit exceeded (core dumped)` — this silently killed a full 30-task
array. The hard limit is `unlimited`, so the fix is one line at the top of every
job script:
```bash
ulimit -t unlimited
```
Also pin BLAS threads for these tiny models (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1`) — single-thread is just as fast (Python-overhead-bound)
and keeps CPU-time ≈ wall-time.

Check availability before submitting:
```bash
ssh bocconi-hpc2 "sinfo -p defq,gpu,gpunew,medium_gpu,medium_gpunew --format='%P %l %a %D %t %C'"
```

### Common commands (via ssh from WSL)

```bash
ssh bocconi-hpc2 "squeue -u gauthier"                 # job queue
ssh bocconi-hpc2 "sbatch /scratch/Gauthier/deeplatent/<script>.sbatch"
ssh bocconi-hpc2 "scancel <jobid>"                    # cancel
ssh bocconi-hpc2 "sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS"
ssh bocconi-hpc2 "tail -20 /scratch/Gauthier/deeplatent/logs_slurm/<name>.<jobid>.log"
```

## Login node rules

The login node is shared. Allowed: file ops, `squeue`/`sbatch`/`scancel`/`sinfo`,
light editing, `module` commands, rsync/scp. **Everything else — any `python`,
even `import torch` — goes through `sbatch`** (or `debug_cpu`/`debug_gpu` for
15-min interactive tests).

## Conda env

Env `deeplatent` (Python 3.11) lives in `~/.conda/envs`. Recreate with
`/scratch/Gauthier/deeplatent/env_setup.sbatch`.

**IMPORTANT — no `conda activate` in batch scripts.** In SLURM batch shells
`conda activate` fails with `CondaError: Run 'conda init' before 'conda activate'`,
and a subsequent bare `pip install` silently falls back to the BASE python and
pollutes `~/.local` (this happened on 2026-08-20; job 631949 had to be cancelled
and `~/.local/lib/python3.13` deleted). Always call the env's binaries by
**absolute path** instead:

```bash
ENVPY=$HOME/.conda/envs/deeplatent/bin/python
$ENVPY -m pip install --cache-dir /scratch/Gauthier/pip_cache <pkg>
$ENVPY my_script.py
```

(`conda create -y -n deeplatent python=3.11` itself works fine in batch after
`module load miniconda3`; only `activate` is broken. Compute nodes DO have
internet access — pip installs can run inside SLURM jobs.)

## SLURM patterns for this project

Batch scripts live in `/scratch/Gauthier/deeplatent/*.sbatch`; job logs go to
`/scratch/Gauthier/deeplatent/logs_slurm/`; experiment outputs to
`/scratch/Gauthier/deeplatent/results/`.

**Monte Carlo = job arrays, one rep per task.** The experiment harnesses take env
knobs (`IP_REPS`, `IP_N`, `IP_SEED`, `IP_OUT`, ...), so a 30-rep cell maps to
`--array=0-29` with `IP_REPS=1`-per-task semantics via `IP_SEED=$((BASE+SLURM_ARRAY_TASK_ID))`
and one JSON per task, merged afterwards. 90 tasks (3 n-cells × 30 reps) run
concurrently instead of 17h sequentially on the local MX550.

Template:
```bash
#!/bin/bash
#SBATCH -J ip_mc
#SBATCH -p defq
#SBATCH -t 02:00:00
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --array=0-29
#SBATCH -o /scratch/Gauthier/deeplatent/logs_slurm/%x.%A_%a.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /scratch/Gauthier/deeplatent/DeepLatent
IP_REPS=1 IP_SEED=$((4000 + SLURM_ARRAY_TASK_ID)) \
IP_OUT=/scratch/Gauthier/deeplatent/results/ip_n1000_rep${SLURM_ARRAY_TASK_ID}.json \
$HOME/.conda/envs/deeplatent/bin/python audit/experiment_ip_pilot.py
```

Notes:
- Our models are tiny (encoders [64,64]); CPU tasks on `defq` are usually the right
  call — GPUs only pay off for the big-N or many-modality runs. Benchmark one rep
  on `debug_cpu` vs `debug_gpu` before committing an array.
- `IP_REPS=1` prints the full calibration trajectory; results JSON per task, merged
  with a small local script after `rsync` back.
- Set `num_workers=0` (harness default) — same DataLoader discipline as WSL.
