"""Edge-case tests for the SFT/DPO core beyond the public smoke suite."""

from __future__ import annotations

import math

import pytest
import torch

import submission_adapter as adapter
from tests.toy_lm import ToyCausalLM, ToyTokenizer


class PadIsEosTokenizer(ToyTokenizer):
    """Mimics Qwen2.5 base, where pad_token_id == eos_token_id."""

    pad_token_id = 1


class NoPadTokenizer(ToyTokenizer):
    pad_token_id = None


class NoEosTokenizer(ToyTokenizer):
    eos_token_id = None


def _manual_logp(model, tokenizer, prompt, response, max_length):
    ids = tokenizer.encode(prompt) + tokenizer.encode(response)
    n_prompt = len(tokenizer.encode(prompt))
    if tokenizer.eos_token_id is not None:
        ids = ids + [tokenizer.eos_token_id]
    ids = ids[:max_length]
    log_probs = torch.log_softmax(model.logit_bias.detach().float(), dim=-1)
    return sum(log_probs[tok].item() for pos, tok in enumerate(ids) if pos >= max(n_prompt, 1))


def test_eos_kept_when_sequence_exactly_fits():
    tok = ToyTokenizer()
    # 3 prompt + 2 response + 1 eos = 6
    batch = adapter.build_sft_batch(tok, ["ab:"], ["xy"], max_length=6)
    assert batch["labels"][0, -1].item() == tok.eos_token_id
    assert batch["attention_mask"][0].tolist() == [1] * 6


def test_eos_dropped_when_response_truncated():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["ab:"], ["xyz"], max_length=5)
    assert batch["input_ids"].shape == (1, 5)
    assert tok.eos_token_id not in batch["labels"][0].tolist()
    assert batch["labels"][0].tolist() == [-100, -100, -100] + tok.encode("xy")


def test_prompt_longer_than_max_length_has_no_response_tokens():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.5)
    batch = adapter.build_sft_batch(tok, ["abcdefgh"], ["xy"], max_length=4)
    assert batch["labels"].eq(-100).all()
    logps = adapter.compute_response_logprobs(model, tok, ["abcdefgh"], ["xy"], max_length=4)
    assert logps.item() == 0.0
    loss = adapter.compute_sft_loss(model, tok, ["abcdefgh"], ["xy"], max_length=4)
    assert torch.isfinite(loss)


def test_pad_equal_to_eos_still_masks_padding_but_keeps_eos_label():
    tok = PadIsEosTokenizer()
    batch = adapter.build_sft_batch(tok, ["Q: ", "Q: "], ["a", "abcd"], max_length=12)
    row = batch["labels"][0].tolist()
    # prompt(3) + a + eos, then padding with the same id must be -100
    assert row[:3] == [-100] * 3
    assert row[3:5] == [tok.encode("a")[0], tok.eos_token_id]
    assert all(v == -100 for v in row[5:])
    assert batch["input_ids"][0, 5:].eq(tok.eos_token_id).all()
    assert batch["attention_mask"][0].tolist() == [1] * 5 + [0] * 7


def test_missing_pad_token_falls_back_to_eos():
    tok = NoPadTokenizer()
    batch = adapter.build_sft_batch(tok, ["Q: "], ["a"], max_length=8)
    assert batch["input_ids"][0, -1].item() == tok.eos_token_id


def test_missing_eos_token_appends_nothing():
    tok = NoEosTokenizer()
    batch = adapter.build_sft_batch(tok, ["Q: "], ["ab"], max_length=8)
    assert batch["labels"][0].ne(-100).sum().item() == 2


def test_row_order_preserved_and_padding_invariant():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.9)
    prompts = ["short: ", "a much longer prompt: ", "mid prompt: "]
    responses = ["x", "longer answer here", "mid"]
    batched = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length=48)
    for i in range(3):
        alone = adapter.compute_response_logprobs(model, tok, [prompts[i]], [responses[i]], max_length=48)
        torch.testing.assert_close(batched[i], alone[0])
        expected = _manual_logp(model, tok, prompts[i], responses[i], 48)
        assert batched[i].item() == pytest.approx(expected, abs=1e-4)
    reversed_logps = adapter.compute_response_logprobs(
        model, tok, prompts[::-1], responses[::-1], max_length=48
    )
    torch.testing.assert_close(reversed_logps, batched.flip(0))


def test_sft_loss_is_token_mean():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.3)
    prompts, responses = ["P: ", "Prompt: "], ["a", "abcdef"]
    loss = adapter.compute_sft_loss(model, tok, prompts, responses, max_length=32)
    logps = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length=32)
    n_tokens = (1 + 1) + (6 + 1)
    torch.testing.assert_close(loss, -logps.sum() / n_tokens)


def test_dpo_loss_from_logps_matches_formula():
    pw = torch.tensor([-1.0, -3.0, -2.5])
    pl = torch.tensor([-2.0, -1.0, -2.5])
    rw = torch.tensor([-1.5, -2.0, -2.0])
    rl = torch.tensor([-1.5, -2.0, -3.0])
    beta = 0.5
    expected = torch.stack(
        [
            -torch.log(torch.sigmoid(beta * ((a - b) - (c - d))))
            for a, b, c, d in zip(pw, pl, rw, rl)
        ]
    ).mean()
    loss = adapter.compute_dpo_loss_from_logps(pw, pl, rw, rl, beta)
    assert loss.ndim == 0
    torch.testing.assert_close(loss, expected)


def test_dpo_loss_is_stable_for_large_margins():
    big = torch.tensor([1e4])
    zero = torch.tensor([0.0])
    loss = adapter.compute_dpo_loss_from_logps(zero, big, zero, zero, beta=1.0)
    assert torch.isfinite(loss) and loss.item() == pytest.approx(1e4)


def test_dpo_initial_loss_is_log_two_when_policy_equals_reference():
    tok = ToyTokenizer()
    policy = ToyCausalLM(tok.vocab_size, bias_scale=0.4)
    reference = ToyCausalLM(tok.vocab_size, bias_scale=0.4)
    loss = adapter.compute_dpo_loss(policy, reference, tok, ["P: "], ["good"], ["bad"], 0.1, 16)
    assert loss.item() == pytest.approx(math.log(2), abs=1e-6)


def test_dpo_training_steps_reduce_loss_and_freeze_reference():
    torch.manual_seed(0)
    tok = ToyTokenizer()
    policy = ToyCausalLM(tok.vocab_size, bias_scale=0.4)
    reference = ToyCausalLM(tok.vocab_size, bias_scale=0.4)
    ref_before = reference.logit_bias.detach().clone()
    opt = torch.optim.SGD(policy.parameters(), lr=1.0)
    args = (tok, ["P: ", "Q: "], ["good", "yes"], ["bad", "no"], 0.5, 16)
    losses = []
    for _ in range(5):
        opt.zero_grad()
        loss = adapter.compute_dpo_loss(policy, reference, *args)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    assert reference.logit_bias.grad is None
    torch.testing.assert_close(reference.logit_bias.detach(), ref_before)


def test_dpo_batch_rows_align_with_sft_batches():
    tok = ToyTokenizer()
    batch = adapter.build_dpo_batch(tok, ["A: ", "B: "], ["yes", "ok"], ["no", "nope"], max_length=10)
    for side, responses in (("chosen", ["yes", "ok"]), ("rejected", ["no", "nope"])):
        expected = adapter.build_sft_batch(tok, ["A: ", "B: "], responses, max_length=10)
        for key in ("input_ids", "attention_mask", "labels"):
            assert torch.equal(batch[side][key], expected[key])


def test_concatenated_forward_matches_separate_forwards():
    from alignment.batching import build_preference_batch
    from alignment.losses import concatenated_logprobs, sequence_logprobs

    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.8)
    batch = build_preference_batch(
        tok, ["A: ", "Bb: "], ["a long chosen answer", "y"], ["n", "a long rejected"], 32, pad_to_max_length=False
    )
    chosen, rejected = concatenated_logprobs(model, batch, tok.pad_token_id)
    torch.testing.assert_close(chosen, sequence_logprobs(model, batch["chosen"]))
    torch.testing.assert_close(rejected, sequence_logprobs(model, batch["rejected"]))
