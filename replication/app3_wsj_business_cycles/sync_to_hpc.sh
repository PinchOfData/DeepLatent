#!/bin/bash
# Copy this module (code, slurm) to the HPC (/scratch/Gauthier/app3_wsj/run: since the 2026-10-10 maintenance the GPU
# nodes see a different /home than the login and CPU nodes), then check that every file arrived unchanged (md5). Run
# from the laptop (WSL), with the Bocconi VPN on:   bash replication/app3_wsj_business_cycles/sync_to_hpc.sh
# Data and results stay on the HPC: results/, logs/, tmp/ and the data next to run/ are never touched.
# The application's code is self-contained: it does not import the deeplatent package.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
HOST=${HPC_HOST:-bocconi-hpc2}  # HPC_HOST=bocconi-hpc to go through the other login node
DEST=/scratch/Gauthier/app3_wsj/run
ssh -o BatchMode=yes $HOST "mkdir -p $DEST/results $DEST/logs $DEST/tmp/mpl /scratch/Gauthier/app3_wsj/data /scratch/Gauthier/app3_wsj/checkpoints"
rsync -a --delete --exclude results --exclude logs --exclude tmp --exclude '__pycache__' "$HERE/" "$HOST:$DEST/"
local_md5=$(cd "$HERE" && md5sum code/*.py slurm/* sync_to_hpc.sh | LC_ALL=C sort -k2)
remote_md5=$(ssh -o BatchMode=yes $HOST "cd $DEST && md5sum code/*.py slurm/* sync_to_hpc.sh" | LC_ALL=C sort -k2)
if [ "$local_md5" = "$remote_md5" ]; then
    echo "synced: $(echo "$local_md5" | wc -l) files identical on $HOST:$DEST"
else
    diff <(echo "$local_md5") <(echo "$remote_md5") || true
    echo "MISMATCH after sync" >&2
    exit 1
fi
