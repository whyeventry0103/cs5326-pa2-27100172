# Small-Model Alignment: SFT + DPO on Qwen2.5-0.5B

From-scratch supervised fine-tuning and Direct Preference Optimization (no HF
`Trainer`, no TRL) on `Qwen/Qwen2.5-0.5B`, plus a TRL `DPOTrainer` + LoRA persona
experiment on `HuggingFaceTB/SmolLM2-360M-Instruct`. Built on the
[course starter](https://github.com/ahmad903501/small-model-alignment-starter).
The final report is in `report/report.pdf`.

## Where things are

| Path | What |
|---|---|
| `src/alignment/batching.py` | prompt+response tokenization, EOS before truncation, response-only `-100` labels, padding |
| `src/alignment/losses.py` | response-only log-probs, token-mean SFT loss, DPO objective, reward accuracy and margin |
| `src/alignment/prompting.py` | the single ChatML prompt template used for SFT, DPO and generation |
| `submission_adapter.py` | grader interface: thin wrappers around `src/alignment` |
| `scripts/prepare_data.py` | builds every fixed train/val/eval subset (seed 42) |
| `scripts/train_sft.py` | hand-written SFT loop |
| `scripts/train_dpo.py` | hand-written DPO loop (used for both HH-RLHF and UltraFeedback) |
| `scripts/train_lora_dpo.py` | TRL `DPOTrainer` + PEFT LoRA persona run |
| `scripts/generate.py` | greedy batched generation, one JSON per model |
| `scripts/build_pairs.py` | blind, order-randomized pairwise JSONL (identity kept in `hidden`) |
| `scripts/judge_pairwise_kaggle.py` | resumable batched judge (Kaggle model proxy) |
| `notebooks/judge_kaggle.ipynb` | Kaggle notebook that runs the judge on the saved pair files |
| `scripts/plot_metrics.py`, `scripts/style_metrics.py`, `scripts/summarize_results.py` | figures and tables for the report |
| `eval_sets/` | fixed eval sets: HH synthetic (300, provided), AlpacaEval (300, seed 42), persona (40) |
| `outputs/<run>/metrics.jsonl`, `config.json` | per-run logs and the exact arguments used |
| `outputs/generations/`, `outputs/eval/` | generations, pair files, judgments, summaries |

Model weights are not committed (they are several GB). Every checkpoint is reproducible
from the commands below.

## Environment

Runs were done on a Linux server with 2x RTX 6000 Ada (48 GB, shared with other jobs),
Python 3.10, in an isolated venv:

```bash
uv venv .venv && uv pip install --python .venv/bin/python "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python -r requirements.txt && uv pip install --python .venv/bin/python -e . --no-deps
```

Verified environment for the reported runs (`nvidia-smi` + version print on the server):
2x NVIDIA RTX 6000 Ada (48 GB), torch 2.6.0+cu124, transformers 4.57.6, datasets 4.8.5,
trl 1.14.1, peft 0.21.2. On Kaggle, use the starter's install pattern (keep the preinstalled torch).

## Tests

```bash
PYTHONPATH=. python -m pytest -q tests/
```

`tests/test_public_adapter_smoke.py` is the public suite. `tests/test_alignment_core.py` adds
edge cases: truncation boundaries, pad==eos tokenizers, missing pad/eos, row order,
padding invariance, exact numerics, and DPO optimisation steps.

## Reproducing the runs

```bash
python scripts/prepare_data.py                        # data/processed/*.jsonl + eval_sets/alpaca_eval_300_seed42.json
python scripts/train_sft.py --output_dir outputs/sft   # ~40 min
python scripts/train_dpo.py --train_path data/processed/hh_train.jsonl --val_path data/processed/hh_val.jsonl \
  --output_dir outputs/dpo_hh --policy_device cuda:0 --ref_device cuda:0 --gradient_checkpointing --eval_batch_size 2
python scripts/train_dpo.py --train_path data/processed/uf_train.jsonl --val_path data/processed/uf_val.jsonl \
  --output_dir outputs/dpo_uf --policy_device cuda:1 --ref_device cuda:1 --gradient_checkpointing --eval_batch_size 2
python scripts/train_lora_dpo.py --output_dir outputs/lora_dpo
bash scripts/run_generation.sh                        # all generations + pair files
```

### Judging (Kaggle model proxy)

All judging goes through Kaggle's model proxy via `kaggle_benchmarks`, in blind batches of 5,
≥5 s apart, with retries, per-batch checkpoints and resume. Every pair is judged in both A/B orders
(`python scripts/swap_pairs.py` writes `outputs/eval/swapped/`).

```bash
# Primary judge gemini-3.6-flash: a private Kaggle Benchmarks task with the pair files as a private dataset
kaggle b t push pa2-pairwise-judge -f scripts/kaggle_judge_task.py -d <owner>/pa2-judge-pairs --wait
kaggle b t run pa2-pairwise-judge -m gemini-3.6-flash --wait
kaggle b t download pa2-pairwise-judge -m gemini-3.6-flash   # -> outputs/eval/judgments/gemini-3.6-flash/
# (same with scripts/kaggle_judge_task_swapped.py and the swapped pair files)

# Secondary judge from any machine: `kaggle b init` writes short-lived MODEL_PROXY_* credentials
bash scripts/run_judging_local.sh google/gemini-3-flash-preview <python-with-kaggle-benchmarks> <refresh.sh> <.env> [pairs_dir] [suffix]

python scripts/summarize_results.py   # report tables, agreement, position bias, examples
```

`notebooks/judge_kaggle.ipynb` is an equivalent single-notebook route for a Kaggle utility notebook.
