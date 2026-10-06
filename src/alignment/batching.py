"""Tokenization and batch construction for response-only language modeling.

Every sequence is built as ``prompt_ids + response_ids + [eos]`` where the
prompt and response are tokenized independently. EOS is appended before
truncation, so a response cut off at ``max_length`` loses its EOS. Prompt and
padding positions get label ``-100``; masks are derived from token positions,
never from token ids, because some tokenizers (Qwen2.5 base) use the same id
for padding and EOS.
"""

from __future__ import annotations

from typing import Sequence

import torch

IGNORE_INDEX = -100


def _encode(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _pad_id(tokenizer) -> int:
    for candidate in (getattr(tokenizer, "pad_token_id", None), getattr(tokenizer, "eos_token_id", None)):
        if candidate is not None:
            return int(candidate)
    return 0


def tokenize_example(
    tokenizer,
    prompt: str,
    response: str,
    max_length: int,
    max_prompt_length: int | None = None,
) -> tuple[list[int], list[int]]:
    """Return (input_ids, labels) for one example, unpadded.

    ``max_prompt_length`` optionally left-truncates the prompt (keeping its
    end, which holds the assistant header) before the right truncation of the
    full sequence to ``max_length``.
    """
    prompt_ids = _encode(tokenizer, prompt)
    if max_prompt_length is not None and len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[-max_prompt_length:]
    response_ids = _encode(tokenizer, response)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        response_ids = response_ids + [int(eos_id)]

    input_ids = (prompt_ids + response_ids)[:max_length]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + response_ids)[:max_length]
    return input_ids, labels


def collate(
    examples: Sequence[tuple[list[int], list[int]]],
    pad_id: int,
    pad_to: int | None = None,
) -> dict[str, torch.Tensor]:
    """Right-pad tokenized examples into tensors, preserving row order."""
    width = max([len(ids) for ids, _ in examples] + [1])
    if pad_to is not None:
        width = max(width, pad_to)
    input_ids, attention_mask, labels = [], [], []
    for ids, labs in examples:
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
        labels.append(labs + [IGNORE_INDEX] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def build_lm_batch(
    tokenizer,
    prompts: Sequence[str],
    responses: Sequence[str],
    max_length: int,
    pad_to_max_length: bool = True,
    max_prompt_length: int | None = None,
) -> dict[str, torch.Tensor]:
    """Build input_ids / attention_mask / response-only labels for a batch."""
    if len(prompts) != len(responses):
        raise ValueError(f"got {len(prompts)} prompts but {len(responses)} responses")
    examples = [
        tokenize_example(tokenizer, p, r, max_length, max_prompt_length)
        for p, r in zip(prompts, responses)
    ]
    return collate(examples, _pad_id(tokenizer), max_length if pad_to_max_length else None)


def build_preference_batch(
    tokenizer,
    prompts: Sequence[str],
    chosen: Sequence[str],
    rejected: Sequence[str],
    max_length: int,
    pad_to_max_length: bool = True,
    max_prompt_length: int | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Build separate chosen / rejected LM batches sharing the same prompts."""
    if not (len(prompts) == len(chosen) == len(rejected)):
        raise ValueError("prompts, chosen and rejected must have equal length")
    kwargs = dict(max_length=max_length, pad_to_max_length=pad_to_max_length, max_prompt_length=max_prompt_length)
    return {
        "chosen": build_lm_batch(tokenizer, prompts, chosen, **kwargs),
        "rejected": build_lm_batch(tokenizer, prompts, rejected, **kwargs),
    }
