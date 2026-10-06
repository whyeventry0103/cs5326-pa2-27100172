"""Build the fixed train/validation/eval subsets used by every run.

Outputs (all deterministic given --seed):
  data/processed/sft_{train,val}.jsonl          UltraChat-200k first turn
  data/processed/hh_{train,val}.jsonl           Anthropic HH-RLHF, single-turn only
  data/processed/uf_{train,val}.jsonl           UltraFeedback binarized, strict preferences
  eval_sets/alpaca_eval_300_seed42.json         AlpacaEval subset (300 prompts, seed 42)

Filtering (documented in the report):
  SFT: prompt+response+EOS must fit in --max_length tokens so EOS is always learned.
  DPO: single-turn only; formatted prompt must fit in --max_prompt_length tokens;
       chosen != rejected; both non-empty; UltraFeedback requires score_chosen > score_rejected.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from alignment.io_utils import write_json, write_jsonl  # noqa: E402
from alignment.prompting import format_prompt  # noqa: E402

ALPACA_EVAL_URL = "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json"


def n_tokens(tok, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))


def take_filtered(rows, keep, n: int, seed: int) -> list[dict]:
    """Shuffle row indices with ``seed`` and keep the first ``n`` rows passing ``keep``."""
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    out = []
    for i in order:
        ex = keep(rows[i])
        if ex is not None:
            out.append(ex)
            if len(out) == n:
                return out
    raise RuntimeError(f"only {len(out)} examples passed the filter, needed {n}")


def build_sft(tok, args, split: str, n: int, tag: str) -> list[dict]:
    rows = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)

    def keep(r):
        msgs = r["messages"]
        if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
            return None
        instruction, response = msgs[0]["content"].strip(), msgs[1]["content"].strip()
        if not instruction or not response:
            return None
        total = n_tokens(tok, format_prompt(instruction)) + n_tokens(tok, response) + 1
        if total > args.max_length:
            return None
        return {"id": f"{tag}-{r['prompt_id']}", "instruction": instruction, "response": response}

    return take_filtered(rows, keep, n, args.seed)


def parse_hh_single_turn(text: str):
    """Return (prompt, response) for a single-turn HH transcript, else None."""
    if text.count("\n\nHuman:") != 1 or text.count("\n\nAssistant:") != 1:
        return None
    head, _, response = text.partition("\n\nAssistant:")
    prompt = head.split("\n\nHuman:", 1)[1]
    return prompt.strip(), response.strip()


def build_hh(tok, args, split: str, n: int, tag: str) -> list[dict]:
    rows = load_dataset("Anthropic/hh-rlhf", split=split)

    def keep(r):
        c, rj = parse_hh_single_turn(r["chosen"]), parse_hh_single_turn(r["rejected"])
        if c is None or rj is None or c[0] != rj[0]:
            return None
        prompt, chosen, rejected = c[0], c[1], rj[1]
        if not prompt or not chosen or not rejected or chosen == rejected:
            return None
        if n_tokens(tok, format_prompt(prompt)) > args.max_prompt_length:
            return None
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    out = take_filtered(rows, keep, n, args.seed)
    return [{"id": f"{tag}-{i:05d}", **ex} for i, ex in enumerate(out)]


def build_uf(tok, args, split: str, n: int, tag: str) -> list[dict]:
    rows = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)

    def keep(r):
        if len(r["chosen"]) != 2 or len(r["rejected"]) != 2:
            return None
        if r["score_chosen"] <= r["score_rejected"]:
            return None
        prompt = r["prompt"].strip()
        chosen, rejected = r["chosen"][1]["content"].strip(), r["rejected"][1]["content"].strip()
        if not prompt or not chosen or not rejected or chosen == rejected:
            return None
        if n_tokens(tok, format_prompt(prompt)) > args.max_prompt_length:
            return None
        return {
            "id": f"{tag}-{r['prompt_id'][:16]}",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "score_chosen": r["score_chosen"],
            "score_rejected": r["score_rejected"],
        }

    return take_filtered(rows, keep, n, args.seed)


def build_alpaca_eval(n: int, seed: int) -> list[dict]:
    with urllib.request.urlopen(ALPACA_EVAL_URL, timeout=120) as resp:
        rows = json.load(resp)
    idx = sorted(random.Random(seed).sample(range(len(rows)), n))
    return [
        {
            "id": f"alpaca-eval-{seed}-{i:04d}",
            "source_index": i,
            "instruction": rows[i]["instruction"],
            "dataset": "alpaca_eval",
            "source_subset": rows[i]["dataset"],
        }
        for i in idx
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--out_dir", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_prompt_length", type=int, default=256)
    ap.add_argument("--sft_train", type=int, default=30000)
    ap.add_argument("--sft_val", type=int, default=1000)
    ap.add_argument("--dpo_train", type=int, default=15000)
    ap.add_argument("--dpo_val", type=int, default=1000)
    ap.add_argument("--alpaca_n", type=int, default=300)
    ap.add_argument("--only", nargs="*", default=["sft", "hh", "uf", "alpaca"])
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    out = Path(args.out_dir)
    stats = {}
    if "sft" in args.only:
        for split, n, name in (("train_sft", args.sft_train, "train"), ("test_sft", args.sft_val, "val")):
            rows = build_sft(tok, args, split, n, f"sft-{name}")
            write_jsonl(out / f"sft_{name}.jsonl", rows)
            stats[f"sft_{name}"] = len(rows)
    if "hh" in args.only:
        for split, n, name in (("train", args.dpo_train, "train"), ("test", args.dpo_val, "val")):
            rows = build_hh(tok, args, split, n, f"hh-{name}")
            write_jsonl(out / f"hh_{name}.jsonl", rows)
            stats[f"hh_{name}"] = len(rows)
    if "uf" in args.only:
        for split, n, name in (("train_prefs", args.dpo_train, "train"), ("test_prefs", args.dpo_val, "val")):
            rows = build_uf(tok, args, split, n, f"uf-{name}")
            write_jsonl(out / f"uf_{name}.jsonl", rows)
            stats[f"uf_{name}"] = len(rows)
    if "alpaca" in args.only:
        rows = build_alpaca_eval(args.alpaca_n, args.seed)
        write_json(ROOT / "eval_sets" / f"alpaca_eval_{args.alpaca_n}_seed{args.seed}.json", rows)
        stats["alpaca_eval"] = len(rows)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
