#!/usr/bin/env bash
# Judge every pair file through the Kaggle model proxy from outside a notebook.
# Usage: bash scripts/run_judging_local.sh <judge_model> <python_with_kaggle_benchmarks> <refresh_script> <env_file> [pairs_dir] [out_suffix]
# refresh_script must re-run `kaggle benchmarks init` (proxy keys are short-lived).
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL=$1; PY=$2; REFRESH=$3; ENVF=$4; PAIRS_DIR=${5:-outputs/eval}; SUFFIX=${6:-}
SLUG=${MODEL##*/}
OUT=outputs/eval/judgments/$SLUG$SUFFIX
mkdir -p "$OUT"
for pairs in "$PAIRS_DIR"/*_pairs.jsonl; do
  key=$(basename "$pairs" _pairs.jsonl)
  bash "$REFRESH" > /dev/null
  echo "=== $key $(date -u +%H:%M:%SZ)"
  "$PY" scripts/judge_pairwise_kaggle.py --pairs_path "$pairs" --output_path "$OUT/${key}_judged.jsonl" \
    --judge_model "$MODEL" --env_file "$ENVF" --batch_size 5 --sleep_seconds 5 --max_retries 3 \
    --reuse_judgments > "$OUT/${key}.log" 2>&1
  grep -E '"(wins|losses|ties|unknown|failed)"' "$OUT/${key}_judged_summary.json" | tr -d ' \n'; echo
done
echo "ALL DONE $(date -u +%H:%M:%SZ)"
