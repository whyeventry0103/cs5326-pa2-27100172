"""Response-only log probabilities, SFT loss, and the DPO objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .batching import IGNORE_INDEX


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def move(t: torch.Tensor, device) -> torch.Tensor:
    """Move a tensor between devices, staging GPU->GPU copies through host memory.

    Direct peer-to-peer copies silently returned zeros on our training server
    (driver reports peer access but the copy is broken), which zeroed the
    reference log-probs when policy and reference sat on different GPUs.
    """
    device = torch.device(device)
    if t.device == device:
        return t
    if t.device.type == "cuda" and device.type == "cuda":
        return t.cpu().to(device)
    return t.to(device)


def _to(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: move(v, device) for k, v in batch.items()}


def token_logprobs(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-token log p(label) and a response mask, both shape [B, T-1].

    Logits at position t predict the token at t+1, so labels are shifted left
    by one. Log-softmax is done in fp32 even under fp16 autocast.
    """
    batch = _to(batch, model_device(model))
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    labels = batch["labels"][:, 1:]
    mask = labels.ne(IGNORE_INDEX)
    # Only response positions are upcast and normalized: [N, V] instead of [B, T, V].
    selected = logits[:, :-1, :][mask].float()
    picked = selected.gather(-1, labels[mask].unsqueeze(-1)).squeeze(-1) - torch.logsumexp(selected, dim=-1)
    logps = torch.zeros(labels.shape, dtype=picked.dtype, device=picked.device).masked_scatter(mask, picked)
    return logps, mask


def sequence_logprobs(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Summed response-token log probability per row, shape [B]."""
    logps, _ = token_logprobs(model, batch)
    return logps.sum(dim=-1)


def sft_loss_from_batch(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-mean response NLL over the batch, plus the response-token count.

    Reduction: mean over all unmasked response tokens in the batch.
    """
    logps, mask = token_logprobs(model, batch)
    n_tokens = mask.sum()
    return -logps.sum() / n_tokens.clamp_min(1), n_tokens


@dataclass(frozen=True)
class DPOOutput:
    loss: torch.Tensor  # scalar batch mean
    chosen_rewards: torch.Tensor  # beta * (pi - ref) on chosen, detached, [B]
    rejected_rewards: torch.Tensor  # same on rejected, [B]

    @property
    def reward_accuracy(self) -> torch.Tensor:
        return (self.chosen_rewards > self.rejected_rewards).float().mean()

    @property
    def reward_margin(self) -> torch.Tensor:
        return (self.chosen_rewards - self.rejected_rewards).mean()


def dpo_objective(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
) -> DPOOutput:
    """-log sigmoid(beta * [(pi_w - pi_l) - (ref_w - ref_l)]), averaged over the batch."""
    device = policy_chosen_logps.device
    ref_w = move(reference_chosen_logps, device)
    ref_l = move(reference_rejected_logps, device)
    chosen_logratio = policy_chosen_logps - ref_w
    rejected_logratio = policy_rejected_logps - ref_l
    losses = -F.logsigmoid(beta * (chosen_logratio - rejected_logratio))
    return DPOOutput(
        loss=losses.mean(),
        chosen_rewards=(beta * chosen_logratio).detach(),
        rejected_rewards=(beta * rejected_logratio).detach(),
    )


def reference_logprobs(reference_model, preference_batch: dict[str, dict[str, torch.Tensor]]):
    """Summed chosen / rejected logps from the frozen reference, no autograd."""
    with torch.no_grad():
        chosen = sequence_logprobs(reference_model, preference_batch["chosen"])
        rejected = sequence_logprobs(reference_model, preference_batch["rejected"])
    return chosen.detach(), rejected.detach()


def concat_preference_batch(preference_batch, pad_id: int) -> dict[str, torch.Tensor]:
    """Stack chosen rows then rejected rows into one right-padded [2B, T] batch."""
    chosen, rejected = preference_batch["chosen"], preference_batch["rejected"]
    width = max(chosen["input_ids"].shape[1], rejected["input_ids"].shape[1])
    fill = {"input_ids": pad_id, "attention_mask": 0, "labels": IGNORE_INDEX}

    def pad(t: torch.Tensor, value: int) -> torch.Tensor:
        return F.pad(t, (0, width - t.shape[1]), value=value)

    return {k: torch.cat([pad(chosen[k], v), pad(rejected[k], v)], dim=0) for k, v in fill.items()}


def concatenated_logprobs(model, preference_batch, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One forward pass over chosen+rejected; returns (chosen_logps, rejected_logps)."""
    n = preference_batch["chosen"]["input_ids"].shape[0]
    logps = sequence_logprobs(model, concat_preference_batch(preference_batch, pad_id))
    return logps[:n], logps[n:]


def dpo_forward(policy_model, reference_model, preference_batch, beta: float) -> DPOOutput:
    """Full DPO step for one preference batch (policy with grad, reference without)."""
    ref_chosen, ref_rejected = reference_logprobs(reference_model, preference_batch)
    policy_chosen = sequence_logprobs(policy_model, preference_batch["chosen"])
    policy_rejected = sequence_logprobs(policy_model, preference_batch["rejected"])
    return dpo_objective(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
