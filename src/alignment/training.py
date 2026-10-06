"""Shared pieces of the hand-written SFT and DPO training loops."""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .io_utils import append_jsonl, write_json


def load_tokenizer(path: str):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(path: str, device: str, dtype: torch.dtype = torch.float32, trainable: bool = True):
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, attn_implementation="sdpa")
    model.to(device)
    model.config.use_cache = False
    if trainable:
        model.train()
    else:
        model.eval()
        model.requires_grad_(False)
    return model


def adamw_param_groups(model, weight_decay: float) -> list[dict]:
    """Apply weight decay to matrices only (not biases or norm weights)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "norm" in name else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def warmup_cosine(total_steps: int, warmup_steps: int):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return fn


def save_checkpoint(model, tokenizer, out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    write_json(out_dir / "checkpoint_meta.json", meta)


class MetricsLogger:
    """Appends one JSON line per event to metrics.jsonl so partial runs survive."""

    def __init__(self, path: Path):
        self.path = path
        self.start = time.time()

    def log(self, **row) -> None:
        row = {"elapsed_s": round(time.time() - self.start, 1), **row}
        append_jsonl(self.path, row)
        print(" ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
