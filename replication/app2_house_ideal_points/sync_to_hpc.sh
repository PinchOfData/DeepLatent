#!/bin/bash
# Copy this module (code, slurm) and the deeplatent package to the HPC home, link the raw inputs,
# and check that every file arrived unchanged (md5). Run from the laptop (WSL), Bocconi VPN on:
#   bash replication/app2_house_ideal_points/sync_to_hpc.sh
# data/, results/, logs/ on the HPC are never touched except for the raw-input links; nor is the
# HPC's git repository (.git, .gitignore), which versions code/, slurm/ and pkg/ there.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
HOST=bocconi-hpc
DEST=/home/Gauthier/app2_house_ideal_points
APP1RAW=/home/Gauthier/app1_us_congress/data/raw
ssh -o BatchMode=yes $HOST "mkdir -p $DEST/pkg $DEST/data/raw $DEST/results $DEST/logs $DEST/tmp \
  && ln -sfn $APP1RAW/hein-bound $DEST/data/raw/hein-bound \
  && ln -sfn $APP1RAW/hein-daily $DEST/data/raw/hein-daily \
  && ln -sfn $APP1RAW/HSall_members.csv $DEST/data/raw/HSall_members.csv \
  && ln -sfn /home/Gauthier/deeplatent/data/HSall_votes.csv $DEST/data/raw/HSall_votes.csv"
rsync -a --delete --exclude data --exclude data_sim --exclude results --exclude logs --exclude tmp --exclude pkg \
    --exclude .git --exclude .gitignore \
    --exclude '__pycache__' "$HERE/" "$HOST:$DEST/"
rsync -a --delete --exclude '__pycache__' --exclude '.ipynb_checkpoints' "$REPO/deeplatent/" "$HOST:$DEST/pkg/deeplatent/"
local_md5=$( (cd "$HERE" && md5sum code/*.py slurm/* sync_to_hpc.sh; cd "$REPO" && md5sum deeplatent/*.py) | LC_ALL=C sort -k2)
remote_md5=$(ssh -o BatchMode=yes $HOST "cd $DEST && md5sum code/*.py slurm/* sync_to_hpc.sh && cd pkg && md5sum deeplatent/*.py" | LC_ALL=C sort -k2)
if [ "$local_md5" = "$remote_md5" ]; then
    echo "synced: $(echo "$local_md5" | wc -l) files identical on $HOST:$DEST"
else
    diff <(echo "$local_md5") <(echo "$remote_md5") || true
    echo "MISMATCH after sync" >&2
    exit 1
fi
