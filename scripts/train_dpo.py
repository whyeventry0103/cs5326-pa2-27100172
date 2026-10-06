"""Hand-written DPO loop (no TRL): trainable policy and frozen reference both
initialized from the SFT best checkpoint, response-only summed log-probs,
-log sigmoid(beta * (delta_policy - delta_ref)), gradient accumulation, fp16
autocast with fp32 master weights + GradScaler, and checkpoint selection by
validation reward accuracy.

The reference is kept in fp32 and run under the same autocast as the policy so
that at step 0 both produce identical log-probs (all implicit rewards tie at 0).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from alignment.batching import build_preference_batch  # noqa: E402
from alignment.io_utils import read_jsonl, seed_everything, write_json  # noqa: E402
from alignment.losses import concatenated_logprobs, dpo_objective  # noqa: E402
from alignment.prompting import format_prompt  # noqa: E402
from alignment.training import MetricsLogger, load_model, load_tokenizer, save_checkpoint  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_path", default=str(ROOT / "outputs/sft/best"))
    ap.add_argument("--train_path", required=True)
    ap.add_argument("--val_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--optimizer", choices=["adamw", "rmsprop"], default="adamw")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_prompt_length", type=int, default=256)
    ap.add_argument("--per_device_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=32)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=30, help="optimizer steps")
    ap.add_argument("--log_every", type=int, default=5, help="optimizer steps")
    ap.add_argument("--eval_batch_size", type=int, default=4, help="pairs per eval forward")
    ap.add_argument("--max_train_examples", type=int, default=None)
    ap.add_argument("--max_val_examples", type=int, default=None)
    ap.add_argument("--policy_device", default="cuda:0")
    ap.add_argument("--ref_device", default="cuda:1")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def to_batch(tok, rows, args):
    return build_preference_batch(
        tok,
        [format_prompt(r["prompt"]) for r in rows],
        [r["chosen"] for r in rows],
        [r["rejected"] for r in rows],
        max_length=args.max_length,
        pad_to_max_length=False,
        max_prompt_length=args.max_prompt_length,
    )


def logps_pair(model, batch, pad_id, grad: bool):
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.float16):
        return concatenated_logprobs(model, batch, pad_id)


@torch.no_grad()
def eval_logps(model, val_batches, pad_id, device):
    was_training = model.training
    model.eval()
    chosen, rejected = [], []
    for batch in val_batches:
        c, r = logps_pair(model, batch, pad_id, grad=False)
        chosen.append(c.to(device))
        rejected.append(r.to(device))
    model.train(was_training)
    return torch.cat(chosen), torch.cat(rejected)


def main():
    args = parse_args()
    seed_everything(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", vars(args))
    log = MetricsLogger(out / "metrics.jsonl")

    tok = load_tokenizer(args.sft_path)
    pad_id = tok.pad_token_id
    train_rows = read_jsonl(args.train_path)[: args.max_train_examples]
    val_rows = read_jsonl(args.val_path)[: args.max_val_examples]
    val_batches = [to_batch(tok, val_rows[i : i + args.eval_batch_size], args) for i in range(0, len(val_rows), args.eval_batch_size)]

    policy = load_model(args.sft_path, args.policy_device, torch.float32, trainable=True)
    if args.gradient_checkpointing:
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reference = load_model(args.sft_path, args.ref_device, torch.float32, trainable=False)

    params = [p for p in policy.parameters() if p.requires_grad]
    if args.optimizer == "adamw":
        optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    else:
        optim = torch.optim.RMSprop(params, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")
    micro_per_epoch = len(train_rows) // args.per_device_batch_size
    total_steps = (micro_per_epoch // args.grad_accum) * args.epochs
    log.log(event="start", train_pairs=len(train_rows), val_pairs=len(val_rows), total_steps=total_steps)

    ref_val_c, ref_val_r = eval_logps(reference, val_batches, pad_id, args.policy_device)

    def run_eval(step: int) -> tuple[float, float]:
        pol_c, pol_r = eval_logps(policy, val_batches, pad_id, args.policy_device)
        o = dpo_objective(pol_c, pol_r, ref_val_c, ref_val_r, args.beta)
        acc, margin = o.reward_accuracy.item(), o.reward_margin.item()
        log.log(
            event="eval", step=step, val_loss=o.loss.item(), val_reward_acc=acc, val_reward_margin=margin,
            val_chosen_reward=o.chosen_rewards.mean().item(), val_rejected_reward=o.rejected_rewards.mean().item(),
            val_policy_chosen_logp=pol_c.mean().item(), val_policy_rejected_logp=pol_r.mean().item(),
        )
        return acc, margin

    run_eval(0)
    best = (float("-inf"), float("-inf"))
    step = 0
    rng = random.Random(args.seed)
    acc_buf: dict[str, list[float]] = {}
    for epoch in range(args.epochs):
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        for m in range(micro_per_epoch):
            rows = [train_rows[j] for j in order[m * args.per_device_batch_size : (m + 1) * args.per_device_batch_size]]
            batch = to_batch(tok, rows, args)
            ref_c, ref_r = logps_pair(reference, batch, pad_id, grad=False)
            pol_c, pol_r = logps_pair(policy, batch, pad_id, grad=True)
            o = dpo_objective(pol_c, pol_r, ref_c, ref_r, args.beta)
            scaler.scale(o.loss / args.grad_accum).backward()
            for k, v in (
                ("loss", o.loss.item()), ("acc", o.reward_accuracy.item()), ("margin", o.reward_margin.item()),
                ("chosen_reward", o.chosen_rewards.mean().item()), ("rejected_reward", o.rejected_rewards.mean().item()),
                ("policy_chosen_logp", pol_c.detach().mean().item()), ("policy_rejected_logp", pol_r.detach().mean().item()),
            ):
                acc_buf.setdefault(k, []).append(v)
            if (m + 1) % args.grad_accum != 0:
                continue

            scaler.unscale_(optim)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm).item()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0 or step == 1:
                means = {f"train_{k}": sum(v) / len(v) for k, v in acc_buf.items()}
                log.log(event="train", step=step, epoch=epoch, grad_norm=grad_norm, loss_scale=scaler.get_scale(),
                        peak_mem_gb=torch.cuda.max_memory_allocated(args.policy_device) / 1e9, **means)
                acc_buf = {}
            if step % args.eval_every == 0 or step == total_steps:
                acc, margin = run_eval(step)
                if (acc, margin) > best:
                    best = (acc, margin)
                    save_checkpoint(policy, tok, out / "best", {"step": step, "val_reward_acc": acc, "val_reward_margin": margin,
                                                                "reference": args.sft_path})
                    log.log(event="saved_best", step=step, val_reward_acc=acc)
            if step == total_steps:
                break

    save_checkpoint(policy, tok, out / "final", {"step": step, "reference": args.sft_path})
    log.log(event="done", step=step, best_val_reward_acc=best[0], best_val_reward_margin=best[1])


if __name__ == "__main__":
    main()
