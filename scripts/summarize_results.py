"""Collect judged comparisons and generation statistics into report tables.

Writes outputs/eval/results_summary.json and prints LaTeX table rows.
Also dumps outputs/eval/examples_<comparison>.json with wins/losses for error analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alignment.io_utils import read_json, read_jsonl, write_json  # noqa: E402

COMPARISONS = [
    ("uf_dpo_vs_sft_alpaca", "UltraFeedback-DPO vs SFT", "AlpacaEval-300"),
    ("hh_dpo_vs_sft_hh", "HH-DPO vs SFT", "HH synthetic-300"),
    ("hh_dpo_vs_sft_alpaca", "HH-DPO vs SFT (cross)", "AlpacaEval-300"),
    ("uf_dpo_vs_sft_hh", "UltraFeedback-DPO vs SFT (cross)", "HH synthetic-300"),
    ("persona_lora_vs_base_style", "LoRA-DPO vs SmolLM2 (style)", "Persona-40"),
    ("persona_lora_vs_base_quality", "LoRA-DPO vs SmolLM2 (substance)", "Persona-40"),
    ("persona_lora_lr1e-4_vs_base_style", "LoRA-DPO lr=1e-4 vs SmolLM2 (style)", "Persona-40"),
    ("persona_lora_lr1e-4_vs_base_quality", "LoRA-DPO lr=1e-4 vs SmolLM2 (substance)", "Persona-40"),
]


def outcome_counts(rows):
    c = Counter(r["outcome"] for r in rows)
    w, l, t, u = (c.get(k, 0) for k in ("win", "loss", "tie", "unknown"))
    return {"wins": w, "losses": l, "ties": t, "unknown": u,
            "win_rate_excl_ties": w / (w + l) if w + l else None,
            "win_rate_ties_half": (w + 0.5 * t) / (w + l + t) if w + l + t else None}


def gen_stats(path: Path) -> dict:
    rows = read_json(path)
    n = len(rows)
    return {"n": n, "mean_words": sum(len(r["output"].split()) for r in rows) / n,
            "hit_max_new_tokens": sum(r.get("hit_max_new_tokens", False) for r in rows) / n,
            "empty": sum(not r["output"].strip() for r in rows) / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", default=str(ROOT / "outputs/eval"))
    ap.add_argument("--gen_dir", default=str(ROOT / "outputs/generations"))
    args = ap.parse_args()
    eval_dir, gen_dir = Path(args.eval_dir), Path(args.gen_dir)

    results = {"comparisons": {}, "generations": {}}
    for key, label, dataset in COMPARISONS:
        judged_path = eval_dir / f"{key}_judged.jsonl"
        if not judged_path.exists():
            continue
        judged = read_jsonl(judged_path)
        pairs = {p["pair_id"]: p for p in read_jsonl(eval_dir / f"{key}_pairs.jsonl")}
        summary = read_json(eval_dir / f"{key}_judged_summary.json")
        by_cat = defaultdict(list)
        for r in judged:
            by_cat[r.get("category") or "all"].append(r)
        results["comparisons"][key] = {
            "label": label, "dataset": dataset, "pairs": len(pairs), "judge_model": summary["judge_model"],
            "failed": summary["failed"], "total_attempts": summary["total_attempts"],
            **outcome_counts(judged),
            "by_category": {c: outcome_counts(v) for c, v in sorted(by_cat.items())} if len(by_cat) > 1 else {},
        }
        examples = []
        for r in judged:
            p = pairs[r["pair_id"]]
            cand_pos = "A" if p["hidden"]["A"] == p["hidden"]["candidate"] else "B"
            base_pos = "B" if cand_pos == "A" else "A"
            examples.append({"outcome": r["outcome"], "category": r.get("category"), "instruction": p["instruction"],
                             "candidate": p[f"response_{cand_pos.lower()}"], "baseline": p[f"response_{base_pos.lower()}"]})
        write_json(eval_dir / f"examples_{key}.json", examples)

    for path in sorted(gen_dir.glob("*.json")):
        if not path.name.endswith(".meta.json"):
            results["generations"][path.stem] = gen_stats(path)
    write_json(eval_dir / "results_summary.json", results)

    print("% comparison & dataset & n & W & T & L & unk/fail & win% (excl. ties) & win% (ties=1/2)")
    for key, r in results["comparisons"].items():
        fmt = lambda x: "--" if x is None else f"{100 * x:.1f}"  # noqa: E731
        print(f"{r['label']} & {r['dataset']} & {r['pairs']} & {r['wins']} & {r['ties']} & {r['losses']} & "
              f"{r['unknown']}/{r['failed']} & {fmt(r['win_rate_excl_ties'])} & {fmt(r['win_rate_ties_half'])} \\\\")
    print(json.dumps(results["generations"], indent=1))


if __name__ == "__main__":
    main()
