"""Single prompt format shared by SFT, DPO, and evaluation generation.

We use an explicit ChatML string (the format Qwen2.5 tokenizers ship special
tokens for) instead of ``tokenizer.apply_chat_template`` so the exact template
is visible in code and cannot drift with tokenizer versions. The response is
followed by ``tokenizer.eos_token`` (``<|endoftext|>`` for Qwen2.5 base), which
is appended by the batch builder, not by this module.
"""

from __future__ import annotations

SYSTEM_PROMPT = "You are a helpful assistant."

CHATML_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{instruction}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def format_prompt(instruction: str, system: str = SYSTEM_PROMPT) -> str:
    """Return the model-facing prompt for one single-turn instruction."""
    return CHATML_TEMPLATE.format(system=system, instruction=instruction.strip())
