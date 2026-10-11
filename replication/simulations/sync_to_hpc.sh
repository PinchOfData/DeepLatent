#!/bin/bash
# Copy this module (code, slurm) and the deeplatent package to the HPC home, then check that every
# file arrived unchanged (md5). Run from the laptop (WSL), with the Bocconi VPN on:
#   bash replication/simulations/sync_to_hpc.sh
# results/, logs/ and tmp/ on the HPC are never touched (fetch results with fetch_results.sh).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
HOST=bocconi-hpc2
DEST=/home/Gauthier/simulations
ssh -o BatchMode=yes $HOST "mkdir -p $DEST/pkg $DEST/results $DEST/logs $DEST/tmp/mpl"
rsync -a --delete --exclude results --exclude logs --exclude tmp --exclude pkg --exclude figures \
    --exclude '__pycache__' "$HERE/" "$HOST:$DEST/"
rsync -a --delete --exclude '__pycache__' --exclude '.ipynb_checkpoints' "$REPO/deeplatent/" "$HOST:$DEST/pkg/deeplatent/"
local_md5=$( (cd "$HERE" && md5sum code/*.py slurm/* sync_to_hpc.sh fetch_results.sh; cd "$REPO" && md5sum deeplatent/*.py) | LC_ALL=C sort -k2)
remote_md5=$(ssh -o BatchMode=yes $HOST "cd $DEST && md5sum code/*.py slurm/* sync_to_hpc.sh fetch_results.sh && cd pkg && md5sum deeplatent/*.py" | LC_ALL=C sort -k2)
if [ "$local_md5" = "$remote_md5" ]; then
    echo "synced: $(echo "$local_md5" | wc -l) files identical on $HOST:$DEST"
else
    diff <(echo "$local_md5") <(echo "$remote_md5") || true
    echo "MISMATCH after sync" >&2
    exit 1
fi
