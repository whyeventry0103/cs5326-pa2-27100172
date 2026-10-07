"""From-scratch SFT and DPO building blocks for the small-model alignment assignment.

Re-exports are resolved lazily so that torch-free utilities (e.g. ``alignment.io_utils``
read/write helpers used by evaluation scripts) import without pulling in torch.
"""

from importlib import import_module

_EXPORTS = {
    "IGNORE_INDEX": "batching",
    "build_lm_batch": "batching",
    "build_preference_batch": "batching",
    "tokenize_example": "batching",
    "DPOOutput": "losses",
    "dpo_forward": "losses",
    "dpo_objective": "losses",
    "reference_logprobs": "losses",
    "sequence_logprobs": "losses",
    "sft_loss_from_batch": "losses",
    "SYSTEM_PROMPT": "prompting",
    "format_prompt": "prompting",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        return getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(f"module 'alignment' has no attribute {name!r}")
