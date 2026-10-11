#!/bin/bash
# Copy the estimates written on the HPC (results/) and the job logs back to the laptop:
#   bash replication/simulations/fetch_results.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
HOST=bocconi-hpc2
SRC=/home/Gauthier/simulations
mkdir -p "$HERE/results" "$HERE/logs"
rsync -a "$HOST:$SRC/results/" "$HERE/results/"
rsync -a "$HOST:$SRC/logs/" "$HERE/logs/"
echo "fetched: $(find "$HERE/results" -type f | wc -l) result files"
