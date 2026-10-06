#!/usr/bin/env bash
# Small LoRA-DPO learning-rate / epoch sweep; each config is trained, then
# generates on the 40-prompt persona eval set for style diagnostics.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../.venv/bin/python}
mkdir -p outputs/lora_sweep
for cfg in "5e-5 3" "1e-4 3" "2e-4 3" "2e-4 5"; do
  set -- $cfg; lr=$1; ep=$2; name="lr${lr}_ep${ep}"
  [ -f "outputs/lora_sweep/$name/gen_persona.json" ] && continue
  $PY scripts/train_lora_dpo.py --lr "$lr" --epochs "$ep" --output_dir "outputs/lora_sweep/$name" > "outputs/lora_sweep/$name.log" 2>&1
  $PY scripts/generate.py --model_path HuggingFaceTB/SmolLM2-360M-Instruct --adapter_path "outputs/lora_sweep/$name/adapter" \
    --eval_set eval_sets/persona_eval_40.json --output_path "outputs/lora_sweep/$name/gen_persona.json" \
    --generator "smollm2-360m-lora-dpo-$name" --prompt_format chat_template --max_new_tokens 256 --batch_size 40 --device cuda:0 \
    >> "outputs/lora_sweep/$name.log" 2>&1
done
