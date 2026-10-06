"""Training-curve figures for the report (SFT, both DPO runs, LoRA-DPO).

Raw per-log-step values are drawn faintly; an exponential moving average
(alpha given by --ema) is overlaid and labelled. Checkpoint selections are marked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from alignment.io_utils import read_json, read_jsonl  # noqa: E402

C_TRAIN, C_VAL, C_HH, C_UF = "#2a6fdb", "#d1495b", "#2a9d8f", "#e9a03b"


def ema(values, alpha):
    out, s = [], None
    for v in values:
        s = v if s is None else alpha * v + (1 - alpha) * s
        out.append(s)
    return out


def series(rows, event, key):
    pts = [(r["step"], r[key]) for r in rows if r.get("event") == event and key in r]
    return [p[0] for p in pts], [p[1] for p in pts]


def plot_sft(run_dir: Path, out: Path, alpha: float) -> None:
    rows = read_jsonl(run_dir / "metrics.jsonl")
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x, y = series(rows, "train", "train_loss")
    ax.plot(x, y, color=C_TRAIN, alpha=0.25, lw=0.8)
    ax.plot(x, ema(y, alpha), color=C_TRAIN, lw=1.6, label=f"train loss (EMA α={alpha})")
    vx, vy = series(rows, "eval", "val_loss")
    ax.plot(vx, vy, "o-", color=C_VAL, ms=3, lw=1.4, label="validation loss (1k ex.)")
    best = read_json(run_dir / "best" / "checkpoint_meta.json")
    ax.axvline(best["step"], color="grey", ls="--", lw=1)
    ax.annotate(f"best ckpt (step {best['step']})", (best["step"], max(vy)), fontsize=8, color="grey",
                xytext=(4, -10), textcoords="offset points")
    ax.set_xlabel("optimizer step (effective batch 32)")
    ax.set_ylabel("response-token NLL")
    ax.set_title("SFT on UltraChat (Qwen2.5-0.5B)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_dpo(runs: dict[str, Path], out: Path, alpha: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    colors = [C_HH, C_UF]
    for (name, run_dir), color in zip(runs.items(), colors):
        rows = read_jsonl(run_dir / "metrics.jsonl")
        best = read_json(run_dir / "best" / "checkpoint_meta.json")["step"]
        for ax, tkey, vkey, label in (
            (axes[0], "train_loss", "val_loss", "DPO loss"),
            (axes[1], "train_acc", "val_reward_acc", "reward accuracy"),
            (axes[2], "train_margin", "val_reward_margin", "reward margin  $r_w - r_l$"),
        ):
            x, y = series(rows, "train", tkey)
            ax.plot(x, ema(y, alpha), color=color, lw=1.0, alpha=0.5, label=f"{name} train (EMA α={alpha})")
            vx, vy = series(rows, "eval", vkey)
            ax.plot(vx, vy, "o-", color=color, ms=3, lw=1.6, label=f"{name} val (1k pairs)")
            ax.axvline(best, color=color, ls="--", lw=0.9)
            ax.set_title(label)
            ax.set_xlabel("optimizer step (effective batch 32)")
            ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("From-scratch DPO from the SFT checkpoint (β=0.1, lr=1e-6); dashed = selected best checkpoint", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_lora(run_dir: Path, out: Path) -> None:
    hist = read_json(run_dir / "log_history.json")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    tr = [h for h in hist if "loss" in h and "eval_loss" not in h]
    ev = [h for h in hist if "eval_loss" in h]
    axes[0].plot([h["step"] for h in tr], [h["loss"] for h in tr], color=C_TRAIN, label="train")
    axes[0].plot([h["step"] for h in ev], [h["eval_loss"] for h in ev], "o-", color=C_VAL, ms=3, label="held-out (20 pairs)")
    axes[0].set_title("LoRA-DPO loss")
    axes[1].plot([h["step"] for h in tr], [h["rewards/margins"] for h in tr], color=C_TRAIN, label="train")
    axes[1].plot([h["step"] for h in ev], [h["eval_rewards/margins"] for h in ev], "o-", color=C_VAL, ms=3, label="held-out")
    axes[1].set_title("reward margin")
    for ax in axes:
        ax.set_xlabel("optimizer step (effective batch 8)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("TRL DPOTrainer + LoRA, SmolLM2-360M-Instruct (unsmoothed)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(ROOT / "outputs"))
    ap.add_argument("--fig_dir", default=str(ROOT / "report" / "figures"))
    ap.add_argument("--lora_run", default="lora_dpo")
    ap.add_argument("--ema", type=float, default=0.2)
    args = ap.parse_args()
    o, figs = Path(args.outputs), Path(args.fig_dir)
    figs.mkdir(parents=True, exist_ok=True)
    plot_sft(o / "sft", figs / "sft_loss.png", args.ema)
    plot_dpo({"HH-RLHF": o / "dpo_hh", "UltraFeedback": o / "dpo_uf"}, figs / "dpo_curves.png", args.ema)
    plot_lora(o / args.lora_run, figs / "lora_dpo.png")
    print(f"figures written to {figs}")


if __name__ == "__main__":
    main()
