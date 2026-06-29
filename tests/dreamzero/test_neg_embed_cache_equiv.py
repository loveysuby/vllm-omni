# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Host-side bitwise-safety: constant negative-prompt embed cache equivalence.

DreamZero re-tokenizes + UMT5-encodes ``self.negative_prompt`` (a model constant)
on *every* step to build the CFG uncond branch. The host-side W7 optimization
(``VLLM_OMNI_CACHE_NEG_PROMPT_EMBED=1``) computes it once and reuses it per device
(``pipeline_dreamzero.py:585-616``).

Why this test exists (and why an end-to-end on/off comparison does NOT prove the
contract): DreamZero's GPU forward is *not* bitwise-reproducible run-to-run
(bf16 + cudnn flash-attention + W1 CUDA-graph/torch.compile). Measured on an
RTX PRO 6000 (Blackwell), two independent ``VLLM_OMNI_CACHE_NEG_PROMPT_EMBED=0`` processes
already disagree on 3/5 steps, and an OPT=0 run matches an OPT=1 run on 4/5 steps
— i.e. the flag is statistically invisible against run-to-run kernel noise. So a
cross-process digest of (video, action) cannot isolate the optimization.

Instead we prove the contract *structurally and in-process*: the optimization is
pure constant-folding of a deterministic pure function. GIVEN a deterministic
encoder, the cached embed is byte-identical to the recomputed one, and the cache
recomputes only when the device key changes. The encoder's own run-to-run
nondeterminism is orthogonal — it affects the OPT=0 baseline identically.

CPU-only; the real UMT5 encoder is replaced by a deterministic, call-counting stub.
"""

from __future__ import annotations

import types

import pytest
import torch

from vllm_omni.diffusion.models.dreamzero import pipeline_dreamzero as P

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_stub_pipeline(counter: dict) -> object:
    """Minimal object exposing exactly what ``_encode_negative_prompt`` touches.

    ``tokenizer`` returns fixed token ids for the constant prompt; ``_encode_text``
    is a deterministic pure function of its inputs that counts how many times the
    (expensive, real) UMT5 forward *would* run.
    """
    obj = types.SimpleNamespace()
    obj.negative_prompt = "色调艳丽，过曝，静态"  # any constant; value is irrelevant

    def fake_tokenizer(_text, **kwargs):
        # Constant prompt -> constant tokens, every call (mirrors real behavior).
        ids = torch.tensor([[1, 2, 3, 4, 0, 0]], dtype=torch.long)
        mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask}

    def fake_encode_text(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        counter["n"] += 1
        # Deterministic pure function of the (constant) inputs => repeatable.
        return input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 4) * 0.5

    obj.tokenizer = fake_tokenizer
    obj._encode_text = fake_encode_text
    # Bind the REAL method under test to our stub object.
    obj._encode_negative_prompt = P.DreamZeroPipeline._encode_negative_prompt.__get__(obj)
    return obj


def test_off_recomputes_every_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """W7 OFF == original path: the encoder runs once per step (no cache)."""
    monkeypatch.setattr(P, "_CACHE_NEG_PROMPT_EMBED", False)
    counter = {"n": 0}
    obj = _make_stub_pipeline(counter)
    dev = torch.device("cpu")

    for _ in range(3):
        obj._encode_negative_prompt(dev)

    assert counter["n"] == 3
    assert not hasattr(obj, "_neg_embed_cache")  # OFF leaves no cache state


def test_on_computes_once_and_reuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """W7 ON: encode runs exactly once; every later call returns the cached tensor."""
    monkeypatch.setattr(P, "_CACHE_NEG_PROMPT_EMBED", True)
    counter = {"n": 0}
    obj = _make_stub_pipeline(counter)
    dev = torch.device("cpu")

    first = obj._encode_negative_prompt(dev)
    again = obj._encode_negative_prompt(dev)
    third = obj._encode_negative_prompt(dev)

    assert counter["n"] == 1  # only the first call hit the encoder
    # Cache hits return the *same object*, not a recomputation.
    assert again is first
    assert third is first
    assert torch.equal(again, first)


def test_on_equals_off_bitwise(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cached embed is byte-identical to the recomputed one (no math change).

    This is the bitwise contract: given a deterministic encoder, caching the
    constant negative prompt yields exactly the same tensor as recomputing it.
    """
    monkeypatch.setattr(P, "_CACHE_NEG_PROMPT_EMBED", False)
    off = _make_stub_pipeline({"n": 0})._encode_negative_prompt(torch.device("cpu"))

    monkeypatch.setattr(P, "_CACHE_NEG_PROMPT_EMBED", True)
    on = _make_stub_pipeline({"n": 0})._encode_negative_prompt(torch.device("cpu"))

    assert on.dtype == off.dtype
    assert on.shape == off.shape
    assert torch.equal(on, off)


def test_cache_is_device_keyed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A different device key forces a recompute (no cross-device tensor reuse)."""
    monkeypatch.setattr(P, "_CACHE_NEG_PROMPT_EMBED", True)
    counter = {"n": 0}
    obj = _make_stub_pipeline(counter)

    obj._encode_negative_prompt(torch.device("cpu"))
    obj._encode_negative_prompt(torch.device("cpu"))  # cache hit
    assert counter["n"] == 1

    obj._encode_negative_prompt(torch.device("meta"))  # different device -> recompute
    assert counter["n"] == 2
