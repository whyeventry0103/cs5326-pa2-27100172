"""Stable grading interface for the SFT and DPO implementation.

Keep the function signatures below unchanged. Each function is a thin wrapper
around the implementation in ``src/alignment``.

Conventions (see ``src/alignment/batching.py``):
- sequence = prompt tokens + response tokens + EOS (EOS appended before
  right truncation to ``max_length``), right-padded to ``max_length``;
- labels are -100 on prompt and padding positions;
- ``compute_sft_loss`` uses the mean over all unmasked response tokens.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from alignment.batching import build_lm_batch, build_preference_batch  # noqa: E402
from alignment.losses import (  # noqa: E402
    dpo_forward,
    dpo_objective,
    sequence_logprobs,
    sft_loss_from_batch,
)


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return input_ids, attention_mask, and response-only labels."""

    return build_lm_batch(tokenizer, list(prompts), list(responses), max_length)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""

    batch = build_lm_batch(tokenizer, list(prompts), list(responses), max_length)
    return sequence_logprobs(model, batch)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Return a scalar response-only SFT loss using either documented reduction."""

    batch = build_lm_batch(tokenizer, list(prompts), list(responses), max_length)
    loss, _ = sft_loss_from_batch(model, batch)
    return loss


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return separate response-masked LM batches for chosen and rejected rows."""

    return build_preference_batch(
        tokenizer, list(prompts), list(chosen_responses), list(rejected_responses), max_length
    )


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return the scalar mean DPO objective from per-example log probabilities."""

    return dpo_objective(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta,
    ).loss


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Return a scalar mean DPO loss with a frozen reference computation."""

    batch = build_preference_batch(
        tokenizer, list(prompts), list(chosen_responses), list(rejected_responses), max_length
    )
    return dpo_forward(policy_model, reference_model, batch, beta).loss
