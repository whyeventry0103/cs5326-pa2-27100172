"""Collect judged comparisons (one or more judges) and generation statistics into report tables.

Judgments live in outputs/eval/judgments/<judge_slug>/<comparison>_judged.jsonl.
Writes outputs/eval/results_summary.json, report/results_tables.tex, and
outputs/eval/examples_<comparison>.json (primary judge) for error analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from judge_pairwise_kaggle import read_jsonl, write_json  # noqa: E402

COMPARISONS = [
    ("uf_dpo_vs_sft_alpaca", "UF-DPO vs SFT", "AlpacaEval-300", "main"),
    ("hh_dpo_vs_sft_hh", "HH-DPO vs SFT", "HH synthetic-300", "main"),
    ("hh_dpo_vs_sft_alpaca", "HH-DPO vs SFT", "AlpacaEval-300", "cross"),
    ("uf_dpo_vs_sft_hh", "UF-DPO vs SFT", "HH synthetic-300", "cross"),
    ("persona_lora_vs_base_style", "LoRA-DPO vs base (style)", "Persona-40", "persona"),
    ("persona_lora_vs_base_quality", "LoRA-DPO vs base (substance)", "Persona-40", "persona"),
    ("persona_lora_lr1e-4_vs_base_style", "LoRA lr=1e-4 vs base (style)", "Persona-40", "persona"),
    ("persona_lora_lr1e-4_vs_base_quality", "LoRA lr=1e-4 vs base (substance)", "Persona-40", "persona"),
]


def outcome_counts(rows: list[dict]) -> dict:
    c = Counter(r["outcome"] for r in rows)
    w, l, t, u = (c.get(k, 0) for k in ("win", "loss", "tie", "unknown"))
    decided = w + l
    return {
        "n": len(rows), "wins": w, "losses": l, "ties": t, "unknown": u,
        "win_rate_excl_ties": w / decided if decided else None,
        "win_rate_ties_half": (w + 0.5 * t) / (w + l + t) if w + l + t else None,
    }


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else None


def gen_stats(path: Path) -> dict:
    rows = json.loads(path.read_text())
    n = len(rows)
    return {"n": n, "mean_words": sum(len(r["output"].split()) for r in rows) / n,
            "hit_max_new_tokens": sum(r.get("hit_max_new_tokens", False) for r in rows) / n}


def pct(x: float | None) -> str:
    return "--" if x is None else f"{100 * x:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments_dir", default=str(ROOT / "outputs/eval/judgments"))
    ap.add_argument("--pairs_dir", default=str(ROOT / "outputs/eval"))
    ap.add_argument("--gen_dir", default=str(ROOT / "outputs/generations"))
    ap.add_argument("--primary", default="gemini-3.6-flash")
    args = ap.parse_args()
    jdir, pdir = Path(args.judgments_dir), Path(args.pairs_dir)
    judges = sorted(d.name for d in jdir.iterdir() if d.is_dir())
    primary = args.primary if args.primary in judges else judges[0]

    results: dict = {"primary_judge": primary, "judges": {}, "agreement": {}, "generations": {}}
    judged_by: dict[str, dict[str, dict[str, dict]]] = defaultdict(dict)
    for judge in judges:
        results["judges"][judge] = {}
        for key, label, dataset, group in COMPARISONS:
            path = jdir / judge / f"{key}_judged.jsonl"
            if not path.exists():
                continue
            rows = read_jsonl(path)
            judged_by[judge][key] = {r["pair_id"]: r for r in rows}
            summ_path = jdir / judge / f"{key}_judged_summary.json"
            summ = json.loads(summ_path.read_text()) if summ_path.exists() else {}
            by_cat = defaultdict(list)
            for r in rows:
                by_cat[r.get("category") or "all"].append(r)
            decided_ab = [r["verdict"] for r in rows if r["verdict"] in ("A", "B")]
            results["judges"][judge][key] = {
                "label": label, "dataset": dataset, "group": group, "pairs": summ.get("pairs", len(rows)),
                "failed": summ.get("failed", 0), "total_attempts": summ.get("total_attempts"),
                "position_A_share": decided_ab.count("A") / len(decided_ab) if decided_ab else None,
                **outcome_counts(rows),
                "by_category": {c: outcome_counts(v) for c, v in sorted(by_cat.items())} if len(by_cat) > 1 else {},
            }

    if len(judges) >= 2:
        j1, j2 = judges[0], judges[1]
        for key, *_ in COMPARISONS:
            a, b = judged_by[j1].get(key, {}), judged_by[j2].get(key, {})
            common = sorted(set(a) & set(b))
            if not common:
                continue
            oa, ob = [a[p]["outcome"] for p in common], [b[p]["outcome"] for p in common]
            results["agreement"][key] = {
                "judges": [j1, j2], "n": len(common),
                "raw_agreement": sum(x == y for x, y in zip(oa, ob)) / len(common), "cohen_kappa": cohen_kappa(oa, ob),
            }

    pairs_cache = {key: {p["pair_id"]: p for p in read_jsonl(pdir / f"{key}_pairs.jsonl")} for key, *_ in COMPARISONS
                   if (pdir / f"{key}_pairs.jsonl").exists()}
    for key, rows in judged_by[primary].items():
        examples = []
        for r in rows.values():
            p = pairs_cache[key][r["pair_id"]]
            cand = "a" if p["hidden"]["A"] == p["hidden"]["candidate"] else "b"
            base = "b" if cand == "a" else "a"
            other = {j: judged_by[j][key][r["pair_id"]]["outcome"] for j in judges
                     if j != primary and r["pair_id"] in judged_by[j].get(key, {})}
            examples.append({"pair_id": r["pair_id"], "outcome": r["outcome"], "other_judges": other,
                             "category": p.get("category"), "instruction": p["instruction"],
                             "candidate": p[f"response_{cand}"], "baseline": p[f"response_{base}"]})
        write_json(pdir / f"examples_{key}.json", examples)

    for path in sorted(Path(args.gen_dir).glob("*.json")):
        if not path.name.endswith(".meta.json"):
            results["generations"][path.stem] = gen_stats(path)
    write_json(pdir / "results_summary.json", results)

    lines = []
    for key, label, dataset, group in COMPARISONS:
        cells = []
        for judge in judges:
            r = results["judges"][judge].get(key)
            cells.append("-- & -- & -- & --" if r is None else
                         f"{r['wins']}/{r['ties']}/{r['losses']} & {r['unknown'] + r['failed']} & "
                         f"{pct(r['win_rate_excl_ties'])} & {pct(r['win_rate_ties_half'])}")
        ag = results["agreement"].get(key)
        agree = "--" if ag is None else f"{100 * ag['raw_agreement']:.0f}\\% / {ag['cohen_kappa']:.2f}"
        n = next((results["judges"][j][key]["pairs"] for j in judges if key in results["judges"][j]), "--")
        lines.append(f"{label} & {dataset} & {n} & " + " & ".join(cells) + f" & {agree} \\\\")
    (ROOT / "report").mkdir(exist_ok=True)
    (ROOT / "report" / "results_rows.tex").write_text("\n".join(lines) + "\n")
    print("judges:", judges, "primary:", primary)
    print("\n".join(lines))
    print(json.dumps(results["agreement"], indent=1))


if __name__ == "__main__":
    main()
