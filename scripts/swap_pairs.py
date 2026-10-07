"""Write order-swapped copies of every pairwise file (Response A <-> Response B).

Judging both orders and keeping only verdicts that agree removes the judge's
position bias (swap-consistency). pair_ids are unchanged so the two passes
line up; swapped files go to a separate directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from judge_pairwise_kaggle import read_jsonl  # noqa: E402

import json  # noqa: E402


def swap(pair: dict) -> dict:
    hidden = {**pair["hidden"], "A": pair["hidden"]["B"], "B": pair["hidden"]["A"]}
    return {**pair, "response_a": pair["response_b"], "response_b": pair["response_a"], "hidden": hidden}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_dir", default=str(ROOT / "outputs/eval"))
    ap.add_argument("--out_dir", default=str(ROOT / "outputs/eval/swapped"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(args.pairs_dir).glob("*_pairs.jsonl")):
        rows = [swap(p) for p in read_jsonl(path)]
        with open(out / path.name, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        print(f"{path.name}: {len(rows)} swapped pairs")


if __name__ == "__main__":
    main()
