"""Hand-written SFT loop (no HF Trainer): response-only loss, fp16 autocast with
fp32 master weights + GradScaler, gradient accumulation, clipping, warmup+cosine,
periodic validation, and best/final checkpoints.

Loss reduction: mean NLL over all unmasked response tokens in each effective
(accumulated) batch. Each micro-batch contributes sum(NLL) / tokens_in_window.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from alignment.batching import collate, tokenize_example  # noqa: E402
from alignment.io_utils import read_jsonl, seed_everything, write_json  # noqa: E402
from alignment.losses import token_logprobs  # noqa: E402
from alignment.prompting import format_prompt  # noqa: E402
from alignment.training import (  # noqa: E402
    MetricsLogger,
    adamw_param_groups,
    load_model,
    load_tokenizer,
    save_checkpoint,
    warmup_cosine,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--train_path", default=str(ROOT / "data/processed/sft_train.jsonl"))
    ap.add_argument("--val_path", default=str(ROOT / "data/processed/sft_val.jsonl"))
    ap.add_argument("--output_dir", default=str(ROOT / "outputs/sft"))
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--per_device_batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=50, help="optimizer steps")
    ap.add_argument("--log_every", type=int, default=5, help="optimizer steps")
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--max_train_examples", type=int, default=None, help="pilot runs only")
    ap.add_argument("--max_val_examples", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def tokenize_rows(tok, rows, max_length):
    return [tokenize_example(tok, format_prompt(r["instruction"]), r["response"], max_length) for r in rows]


@torch.no_grad()
def evaluate(model, examples, pad_id, batch_size, device) -> float:
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for i in range(0, len(examples), batch_size):
        batch = collate(examples[i : i + batch_size], pad_id)
        with torch.autocast("cuda", dtype=torch.float16):
            logps, mask = token_logprobs(model, batch)
        total_nll -= logps.sum().item()
        total_tokens += mask.sum().item()
    model.train()
    return total_nll / max(1, total_tokens)


def main():
    args = parse_args()
    seed_everything(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", vars(args))
    log = MetricsLogger(out / "metrics.jsonl")

    tok = load_tokenizer(args.model)
    pad_id = tok.pad_token_id
    train_rows = read_jsonl(args.train_path)[: args.max_train_examples]
    val_rows = read_jsonl(args.val_path)[: args.max_val_examples]
    train = tokenize_rows(tok, train_rows, args.max_length)
    val = tokenize_rows(tok, val_rows, args.max_length)

    model = load_model(args.model, args.device, torch.float32, trainable=True)
    optim = torch.optim.AdamW(adamw_param_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    micro_per_epoch = len(train) // args.per_device_batch_size
    total_steps = (micro_per_epoch // args.grad_accum) * args.epochs
    warmup = max(1, round(args.warmup_ratio * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, warmup_cosine(total_steps, warmup))
    scaler = torch.amp.GradScaler("cuda")
    log.log(event="start", train_examples=len(train), val_examples=len(val), total_steps=total_steps, warmup_steps=warmup)

    val_loss = evaluate(model, val, pad_id, args.eval_batch_size, args.device)
    log.log(event="eval", step=0, val_loss=val_loss)
    best_val = float("inf")
    step = 0
    rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        micro_batches = [
            [train[j] for j in order[i : i + args.per_device_batch_size]]
            for i in range(0, micro_per_epoch * args.per_device_batch_size, args.per_device_batch_size)
        ]
        for w in range(0, len(micro_batches) - args.grad_accum + 1, args.grad_accum):
            window = [collate(mb, pad_id) for mb in micro_batches[w : w + args.grad_accum]]
            window_tokens = sum(int(b["labels"][:, 1:].ne(-100).sum()) for b in window)
            window_nll = 0.0
            for batch in window:
                with torch.autocast("cuda", dtype=torch.float16):
                    logps, _ = token_logprobs(model, batch)
                loss = -logps.sum() / max(1, window_tokens)
                scaler.scale(loss).backward()
                window_nll += loss.item()
            scaler.unscale_(optim)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm).item()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            sched.step()
            step += 1

            if step % args.log_every == 0 or step == 1:
                log.log(
                    event="train", step=step, epoch=epoch, train_loss=window_nll, lr=sched.get_last_lr()[0],
                    grad_norm=grad_norm, loss_scale=scaler.get_scale(), tokens=window_tokens,
                    peak_mem_gb=torch.cuda.max_memory_allocated(args.device) / 1e9,
                )
            if step % args.eval_every == 0 or step == total_steps:
                val_loss = evaluate(model, val, pad_id, args.eval_batch_size, args.device)
                improved = val_loss < best_val
                log.log(event="eval", step=step, val_loss=val_loss, best=improved)
                if improved:
                    best_val = val_loss
                    save_checkpoint(model, tok, out / "best", {"step": step, "val_loss": val_loss})

    save_checkpoint(model, tok, out / "final", {"step": step, "val_loss": val_loss})
    log.log(event="done", step=step, best_val_loss=best_val, final_val_loss=val_loss)


if __name__ == "__main__":
    main()
