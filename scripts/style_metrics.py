"""Cheap lexical persona-style diagnostics for persona generations.

The lexicon is the set of signature insults in the chosen responses of
data/persona_lora/ramsay_style_preferences.json (e.g. "you donkey!"). This is
a proxy for style, not a quality metric; the pairwise judge covers quality.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alignment.io_utils import read_json, write_json  # noqa: E402

INSULTS = r"donut|doughnut|donkey|weapon|clown|lunatic|muppet|maniac|barbarian|disaster|butcher|idiot|fool"
INSULT_RE = re.compile(rf"\byou (?:\w+ )?(?:{INSULTS})\b", re.I)


def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n": n,
        "insult_rate": sum(bool(INSULT_RE.search(r["output"])) for r in rows) / n,
        "exclamations_per_response": sum(r["output"].count("!") for r in rows) / n,
        "mean_words": sum(len(r["output"].split()) for r in rows) / n,
        "hit_max_new_tokens": sum(r.get("hit_max_new_tokens", False) for r in rows) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("generation_files", nargs="+")
    ap.add_argument("--output_path", default=None)
    args = ap.parse_args()
    report = {}
    for path in args.generation_files:
        rows = read_json(path)
        by_cat = defaultdict(list)
        for r in rows:
            by_cat[r.get("category", "all")].append(r)
        report[Path(path).stem] = {"overall": metrics(rows), **{c: metrics(v) for c, v in sorted(by_cat.items())}}
    print(json.dumps(report, indent=1))
    if args.output_path:
        write_json(args.output_path, report)


if __name__ == "__main__":
    main()
