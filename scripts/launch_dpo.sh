#!/usr/bin/env bash
# Launch both from-scratch DPO runs from the SFT best checkpoint.
# Each run keeps policy and frozen reference on the SAME GPU: direct GPU-to-GPU
# copies are broken on our server (they return zeros), and one run fits in ~12 GB.
# Usage: bash scripts/launch_dpo.sh [hh|uf|both]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../.venv/bin/python}
WHICH=${1:-both}
export PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { # name data device
  mkdir -p "outputs/dpo_$1"
  tmux new-session -d -s "pa2_dpo_$1" "$PY scripts/train_dpo.py --train_path data/processed/$2_train.jsonl \
    --val_path data/processed/$2_val.jsonl --output_dir outputs/dpo_$1 --policy_device $3 --ref_device $3 \
    --gradient_checkpointing --eval_batch_size 2 2>&1 | grep --line-buffered -v -i 'warn\|deprecated\|regex' > outputs/dpo_$1/train.log"
  echo "launched pa2_dpo_$1 on $3"
}
if [ "$WHICH" = hh ] || [ "$WHICH" = both ]; then run hh hh cuda:0; fi
if [ "$WHICH" = uf ] || [ "$WHICH" = both ]; then run uf uf cuda:1; fi
