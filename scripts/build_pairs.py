"""Build a blind pairwise JSONL file (candidate vs baseline) from two generation files.

Response order is randomized per pair with a fixed seed. The A/B -> model mapping
is stored only under ``hidden``; the judge script never puts it in the prompt.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alignment.io_utils import read_json, write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate_path", required=True, help="e.g. DPO generations")
    ap.add_argument("--baseline_path", required=True, help="e.g. SFT generations")
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--eval_kind", choices=["general", "safety", "persona_style", "persona_quality"], default="general")
    ap.add_argument("--eval_set", default=None, help="optional eval set to pull expected_behavior guidance from")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cand = {r["id"]: r for r in read_json(args.candidate_path)}
    base_rows = read_json(args.baseline_path)
    guidance = {}
    if args.eval_set:
        guidance = {r["id"]: r.get("expected_behavior") for r in read_json(args.eval_set)}
    missing = [r["id"] for r in base_rows if r["id"] not in cand]
    if missing:
        raise SystemExit(f"{len(missing)} baseline ids missing from candidate file, e.g. {missing[:3]}")

    rng = random.Random(args.seed)
    pairs = []
    for b in base_rows:
        c = cand[b["id"]]
        assert c["instruction"] == b["instruction"], f"instruction mismatch for {b['id']}"
        cand_first = rng.random() < 0.5
        a, bb = (c, b) if cand_first else (b, c)
        pairs.append({
            "pair_id": f"{args.eval_kind}:{b['id']}",
            "id": b["id"],
            "eval_kind": args.eval_kind,
            "category": b.get("category"),
            "instruction": b["instruction"],
            "guidance": guidance.get(b["id"]),
            "response_a": a["output"],
            "response_b": bb["output"],
            "hidden": {"A": a["generator"], "B": bb["generator"], "candidate": c["generator"], "baseline": b["generator"]},
        })
    write_jsonl(args.output_path, pairs)
    n_a = sum(p["hidden"]["A"] == p["hidden"]["candidate"] for p in pairs)
    print(f"wrote {len(pairs)} pairs to {args.output_path} (candidate in position A: {n_a})")


if __name__ == "__main__":
    main()
