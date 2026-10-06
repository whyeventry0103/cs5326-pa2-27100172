"""LaTeX table of the LoRA-DPO sweep (report/sweep_table.tex) from outputs/lora_sweep/*."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from style_metrics import metrics  # noqa: E402

from alignment.io_utils import read_json  # noqa: E402

SELECTED = "lr2e-4_ep3_b0.1_sft1.0"


def row(label: str, gen_path: Path, summary_path: Path | None) -> str:
    gens = read_json(gen_path)
    m = metrics(gens)
    s = metrics([g for g in gens if g.get("category") == "safety_sensitive"])
    acc = margin = "--"
    if summary_path is not None:
        ev = read_json(summary_path)["final_eval"]
        acc, margin = f"{ev['eval_rewards/accuracies']:.2f}", f"{ev['eval_rewards/margins']:.2f}"
    return (f"{label} & {acc} & {margin} & {100 * m['insult_rate']:.0f}\\% & {m['exclamations_per_response']:.1f} & "
            f"{m['mean_words']:.0f} & {100 * m['hit_max_new_tokens']:.0f}\\% & {100 * s['insult_rate']:.0f}\\% \\\\")


def main() -> None:
    sweep = ROOT / "outputs" / "lora_sweep"
    lines = [
        r"\begin{table}[H]\centering\small",
        r"\caption{LoRA-DPO sweep (3 epochs, r=16). Pref.\ acc./margin: TRL metrics on the 20 held-out pairs. "
        r"Style proxies on the 40 persona prompts: share of responses with a signature insult from the training data "
        r"(``you donkey/donut/\ldots''), exclamation marks per response, mean words, share hitting the 256-token cap "
        r"(repetition loops), and insult share on the 10 safety-sensitive prompts. \textbf{Bold}: submitted.}",
        r"\label{tab:sweep}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Config & Pref.\ acc. & Margin & Insult & ! / resp & Words & Hit cap & Insult (safety) \\",
        r"\midrule",
        row("SmolLM2-360M-Instruct (no LoRA)", ROOT / "outputs/generations/persona_smollm2_base.json", None),
    ]
    def order(p: Path):
        parts = p.name.split("_")
        return ("sft1.0" in p.name, float(parts[0][2:]), float(parts[2][1:])) if len(parts) == 4 else (True, 0.0, 0.0)

    for d in sorted((p for p in sweep.iterdir() if p.is_dir()), key=order):
        gen = d / "gen_persona.json"
        if not d.is_dir() or not gen.exists():
            continue
        lr, _, beta, sft = d.name.split("_")
        label = f"lr={lr[2:]}, $\\beta$={beta[1:]}, " + ("+NLL" if sft != "sft0" else "plain DPO")
        if d.name == SELECTED:
            label = f"\\textbf{{{label}}}"
        lines.append(row(label, gen, d / "summary.json"))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out = ROOT / "report" / "sweep_table.tex"
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
