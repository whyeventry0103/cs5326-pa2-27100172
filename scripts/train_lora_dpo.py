"""Task 5: TRL DPOTrainer + PEFT/LoRA persona tuning of SmolLM2-360M-Instruct.

Rows {prompt, chosen, rejected} are converted to TRL's conversational format
(role/content message lists) so DPOTrainer applies the model's own chat
template. With a peft_config and ref_model=None, TRL uses the frozen base
weights (adapter disabled) as the DPO reference.

LoRA choices (justified in the report):
  r=16, alpha=32, dropout=0.05 on all attention and MLP projections. Persona
  style is a broad lexical/tonal shift, so adapting the MLPs as well as
  attention gives the adapter enough capacity on a 360M model, while r=16
  keeps it to ~1% of the parameters on only ~200 training pairs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import Dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import DPOConfig, DPOTrainer  # noqa: E402

from alignment.io_utils import read_json, write_json  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--data_path", default=str(ROOT / "data/persona_lora/ramsay_style_preferences.json"))
    ap.add_argument("--output_dir", default=str(ROOT / "outputs/lora_dpo"))
    ap.add_argument("--val_size", type=int, default=20)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--sft_weight", type=float, default=0.0,
                    help="weight of an added NLL term on chosen responses (RPO-style); 0 = plain DPO")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--per_device_batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def to_conversational(rows):
    return Dataset.from_list([
        {
            "prompt": [{"role": "user", "content": r["prompt"]}],
            "chosen": [{"role": "assistant", "content": r["chosen"]}],
            "rejected": [{"role": "assistant", "content": r["rejected"]}],
        }
        for r in rows
    ])


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run_args.json", vars(args))

    rows = read_json(args.data_path)
    random.Random(args.seed).shuffle(rows)
    val_rows, train_rows = rows[: args.val_size], rows[args.val_size :]
    write_json(out / "split.json", {"train": len(train_rows), "val": len(val_rows),
                                    "val_prompts": [r["prompt"] for r in val_rows]})

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="bfloat16")
    peft_config = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
        target_modules=args.target_modules, bias="none", task_type="CAUSAL_LM",
    )
    config = DPOConfig(
        output_dir=str(out / "trainer"),
        beta=args.beta,
        loss_type=["sigmoid", "sft"] if args.sft_weight > 0 else ["sigmoid"],
        loss_weights=[1.0, args.sft_weight] if args.sft_weight > 0 else None,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        bf16=True,
        gradient_checkpointing=False,
        logging_steps=2,
        eval_strategy="steps",
        eval_steps=5,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
    )
    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=to_conversational(train_rows),
        eval_dataset=to_conversational(val_rows),
        processing_class=tok,
        peft_config=peft_config,
    )
    trainable, total = trainer.model.get_nb_trainable_parameters()
    print(f"trainable params: {trainable} / {total} ({100 * trainable / total:.2f}%)", flush=True)
    trainer.train()
    trainer.model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    write_json(out / "log_history.json", trainer.state.log_history)
    write_json(out / "summary.json", {"trainable_params": trainable, "total_params": total,
                                      "final_eval": trainer.evaluate()})
    print(json.dumps(trainer.state.log_history[-3:], indent=1))


if __name__ == "__main__":
    main()
