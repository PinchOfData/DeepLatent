# Sourced by the data-side jobs (steps 01-04): the Platforms & Policies Python environment
# (spaCy 3.8 + en_core_web_lg 3.8.0, fasttext), which is what P&P's own cleaning runs in.
set -euo pipefail
ulimit -t unlimited  # the cluster's default 600 s CPU-time limit kills long jobs
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
APP1=/home/Gauthier/app1_us_congress
export TMPDIR=$APP1/tmp
mkdir -p "$TMPDIR" "$APP1/logs"
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/Gauthier/envs/platforms_policies
cd "$APP1"
md5sum --quiet -c code/vendor_pp/MD5SUMS  # the vendored P&P code is unmodified
cd "$APP1/code"
echo "host $(hostname) | python $(python -c 'import sys; print(sys.version.split()[0])') | spacy $(python -c 'import spacy; print(spacy.__version__)')"
