"""From-scratch SFT and DPO building blocks for the small-model alignment assignment."""

from .batching import IGNORE_INDEX, build_lm_batch, build_preference_batch, tokenize_example
from .losses import (
    DPOOutput,
    dpo_forward,
    dpo_objective,
    reference_logprobs,
    sequence_logprobs,
    sft_loss_from_batch,
)
from .prompting import SYSTEM_PROMPT, format_prompt

__all__ = [
    "IGNORE_INDEX",
    "DPOOutput",
    "SYSTEM_PROMPT",
    "build_lm_batch",
    "build_preference_batch",
    "dpo_forward",
    "dpo_objective",
    "format_prompt",
    "reference_logprobs",
    "sequence_logprobs",
    "sft_loss_from_batch",
    "tokenize_example",
]
