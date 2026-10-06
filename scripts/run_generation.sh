#!/usr/bin/env bash
# Task 6: all generations (same greedy decoding for every Qwen model) and blind pair files.
# Idempotent: generate.py skips any output that already exists.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../.venv/bin/python}
DEVICE=${DEVICE:-cuda:0}
REP=${REP:-1.0}           # repetition_penalty, identical for all Qwen models
NGRAM=${NGRAM:-0}         # no_repeat_ngram_size, identical for all Qwen models
MAXNEW=${MAXNEW:-512}
G=outputs/generations
E=outputs/eval
mkdir -p "$G" "$E"

ALPACA=eval_sets/alpaca_eval_300_seed42.json
HH=eval_sets/hh_rlhf_eval_300_seed42.json
PERSONA=eval_sets/persona_eval_40.json

gen() { # model_path generator eval_set dataset out
  $PY scripts/generate.py --model_path "$1" --generator "$2" --eval_set "$3" --dataset "$4" --output_path "$5" \
    --max_new_tokens "$MAXNEW" --repetition_penalty "$REP" --no_repeat_ngram_size "$NGRAM" --device "$DEVICE"
}

for spec in "Qwen/Qwen2.5-0.5B qwen2.5-0.5b-base base" \
            "outputs/sft/best qwen2.5-0.5b-sft sft" \
            "outputs/dpo_uf/best qwen2.5-0.5b-dpo-ultrafeedback dpo_uf" \
            "outputs/dpo_hh/best qwen2.5-0.5b-dpo-hh dpo_hh"; do
  set -- $spec
  gen "$1" "$2" "$ALPACA" alpaca_eval "$G/alpaca_$3.json"
  gen "$1" "$2" "$HH" hh_rlhf_style_eval "$G/hh_$3.json"
done

# Persona track uses SmolLM2's own chat template and a shorter budget (persona answers are ~30 words).
for spec in "smollm2-360m-instruct-base persona_smollm2_base" "smollm2-360m-instruct-lora-dpo persona_lora_dpo"; do
  set -- $spec
  adapter=()
  [ "$2" = persona_lora_dpo ] && adapter=(--adapter_path outputs/lora_dpo/adapter)
  $PY scripts/generate.py --model_path HuggingFaceTB/SmolLM2-360M-Instruct "${adapter[@]}" --generator "$1" \
    --eval_set "$PERSONA" --dataset persona_style_eval --output_path "$G/$2.json" --prompt_format chat_template \
    --max_new_tokens 256 --batch_size 40 --device "$DEVICE"
done

pairs() { $PY scripts/build_pairs.py --candidate_path "$1" --baseline_path "$2" --output_path "$E/$3_pairs.jsonl" --eval_kind "$4" "${@:5}"; }
# Required comparisons
pairs "$G/alpaca_dpo_uf.json" "$G/alpaca_sft.json" uf_dpo_vs_sft_alpaca general
pairs "$G/hh_dpo_hh.json" "$G/hh_sft.json" hh_dpo_vs_sft_hh safety --eval_set "$HH"
pairs "$G/persona_lora_dpo.json" "$G/persona_smollm2_base.json" persona_lora_vs_base_style persona_style
pairs "$G/persona_lora_dpo.json" "$G/persona_smollm2_base.json" persona_lora_vs_base_quality persona_quality
# Cross-set checks (alignment tax / transfer)
pairs "$G/alpaca_dpo_hh.json" "$G/alpaca_sft.json" hh_dpo_vs_sft_alpaca general
pairs "$G/hh_dpo_uf.json" "$G/hh_sft.json" uf_dpo_vs_sft_hh safety --eval_set "$HH"
