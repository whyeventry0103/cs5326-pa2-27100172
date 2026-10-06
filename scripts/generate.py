"""Greedy batched generation on a fixed eval set; writes one JSON file per model.

Prompt formats:
  chatml        the explicit ChatML template used for SFT/DPO training (Qwen runs)
  chat_template the tokenizer's own chat template (SmolLM2-Instruct persona runs)

Entries contain id, instruction, output, generator, dataset (+ category when the
eval set has one). Decoding settings are written to <output>.meta.json.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from alignment.io_utils import read_json, write_json  # noqa: E402
from alignment.prompting import format_prompt  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--adapter_path", default=None, help="optional PEFT/LoRA adapter")
    ap.add_argument("--eval_set", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--generator", required=True)
    ap.add_argument("--dataset", default=None, help="defaults to each row's dataset field")
    ap.add_argument("--prompt_format", choices=["chatml", "chat_template"], default="chatml")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="sanity checks on the first N prompts")
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_path)
    if out.exists() and not args.overwrite and args.limit is None:
        print(f"{out} exists; skipping (use --overwrite)")
        return

    rows = read_json(args.eval_set)[: args.limit]
    tok = AutoTokenizer.from_pretrained(args.model_path)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=getattr(torch, args.dtype))
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path).merge_and_unload()
    model.to(args.device).eval()

    if args.prompt_format == "chatml":
        prompts = [format_prompt(r["instruction"]) for r in rows]
    else:
        prompts = [
            tok.apply_chat_template([{"role": "user", "content": r["instruction"]}], tokenize=False, add_generation_prompt=True)
            for r in rows
        ]
    stop_ids = sorted({i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if i is not None and i >= 0})

    gen_kwargs = dict(
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        eos_token_id=stop_ids,
        pad_token_id=tok.pad_token_id,
    )
    # Longest prompts first so padding waste stays low; outputs are restored to input order.
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    outputs: dict[int, tuple[str, int]] = {}
    t0 = time.time()
    for s in range(0, len(order), args.batch_size):
        idx = order[s : s + args.batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(args.device)
        with torch.no_grad():
            gen = model.generate(**enc, **gen_kwargs)
        for row_i, seq in zip(idx, gen[:, enc["input_ids"].shape[1] :]):
            ids = seq.tolist()
            n_new = next((k for k, t in enumerate(ids) if t in stop_ids), len(ids))
            outputs[row_i] = (tok.decode(ids[:n_new], skip_special_tokens=True).strip(), n_new)
        print(f"{min(s + args.batch_size, len(order))}/{len(order)} done ({time.time() - t0:.0f}s)", flush=True)

    results = []
    for i, r in enumerate(rows):
        entry = {
            "id": r["id"],
            "instruction": r["instruction"],
            "output": outputs[i][0],
            "generator": args.generator,
            "dataset": args.dataset or r.get("dataset", "unknown"),
        }
        if "category" in r:
            entry["category"] = r["category"]
        entry["n_new_tokens"] = outputs[i][1]
        entry["hit_max_new_tokens"] = outputs[i][1] >= args.max_new_tokens
        results.append(entry)

    if args.limit is not None:
        for e in results:
            print("=" * 80, f"\n[{e['id']}] {e['instruction'][:200]}\n---\n{e['output'][:1200]}\n(tokens={e['n_new_tokens']})")
        return
    write_json(out, results)
    meta = {k: v for k, v in vars(args).items() if k not in ("overwrite", "limit")}
    meta.update(stop_token_ids=stop_ids, seconds=round(time.time() - t0, 1), n=len(results),
                frac_hit_max_new_tokens=sum(e["hit_max_new_tokens"] for e in results) / len(results))
    write_json(out.with_suffix(".meta.json"), meta)
    print(f"wrote {len(results)} generations to {out}")


if __name__ == "__main__":
    main()
