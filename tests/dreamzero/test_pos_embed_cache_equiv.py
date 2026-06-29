# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Host-side bitwise-safety: positive (task) prompt encode cache equivalence.

DreamZero re-tokenizes + UMT5-encodes the task prompt on *every* step
(``encode_text_pos`` ~28ms, p99 221ms). Within a closed-loop session the prompt
is constant, so the host-side W7 optimization (``VLLM_OMNI_CACHE_PROMPT_EMBED=1``) caches the
``(text_tokens, attention_mask, prompt_embeds)`` triple per ``(prompt_str,
device)`` in a small LRU (``pipeline_dreamzero.py:_encode_positive_prompt``).

Like the neg-embed cache, this cannot be validated by an end-to-end on/off output
digest -- DreamZero's GPU forward is not bitwise-reproducible run-to-run
(bf16 + cudnn flash-attn + CUDA graph; see test_neg_embed_cache_equiv.py). The
contract is proven structurally and in-process: caching is pure constant-folding
of a deterministic pure function (identical prompt -> identical tokens ->
identical UMT5 eval -> identical embeds). A different prompt recomputes; the LRU
evicts; the cache is device-keyed.

CPU-only; the real UMT5 encoder is replaced by a deterministic, counting stub.
"""

from __future__ import annotations

import types

import pytest
import torch

from vllm_omni.diffusion.models.dreamzero import pipeline_dreamzero as P

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_stub_pipeline(counter: dict) -> object:
    """Minimal object exposing what ``_encode_positive_prompt`` touches.

    ``tokenizer`` maps a prompt string -> deterministic tokens (counting calls);
    ``_encode_text`` is a deterministic pure function of its inputs (counting the
    expensive UMT5 forwards that would run).
    """
    obj = types.SimpleNamespace()

    def fake_tokenizer(text, **_kwargs):
        counter["tok"] += 1
        v = (sum(ord(c) for c in text) % 50) + 1  # deterministic per prompt string
        ids = torch.arange(1, 7, dtype=torch.long).unsqueeze(0) + v
        mask = torch.ones(1, 6, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask}

    def fake_encode_text(input_ids: torch.Tensor, _attention_mask: torch.Tensor) -> torch.Tensor:
        counter["enc"] += 1
        return input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 4) * 0.5

    obj.tokenizer = fake_tokenizer
    obj._encode_text = fake_encode_text
    obj._encode_positive_prompt = P.DreamZeroPipeline._encode_positive_prompt.__get__(obj)
    return obj


def test_off_recomputes_every_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """W7 OFF == original path: tokenize + encode run once per step (no cache)."""
    monkeypatch.setattr(P, "_CACHE_PROMPT_EMBED", False)
    counter = {"tok": 0, "enc": 0}
    obj = _make_stub_pipeline(counter)
    dev = torch.device("cpu")

    for _ in range(3):
        obj._encode_positive_prompt("put the cube in the bowl", dev)

    assert counter["tok"] == 3
    assert counter["enc"] == 3
    assert not hasattr(obj, "_pos_cache")  # OFF leaves no cache state


def test_on_computes_once_per_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """W7 ON: same prompt -> tokenize + encode once; later calls reuse the triple."""
    monkeypatch.setattr(P, "_CACHE_PROMPT_EMBED", True)
    counter = {"tok": 0, "enc": 0}
    obj = _make_stub_pipeline(counter)
    dev = torch.device("cpu")
    prompt = "put the cube in the bowl"

    t0, m0, e0 = obj._encode_positive_prompt(prompt, dev)
    t1, m1, e1 = obj._encode_positive_prompt(prompt, dev)

    assert counter["tok"] == 1 and counter["enc"] == 1
    # Cache hits return the same objects, not a recomputation.
    assert t1 is t0 and m1 is m0 and e1 is e0
    assert torch.equal(t1, t0) and torch.equal(e1, e0)


def test_on_equals_off_bitwise(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cached triple is byte-identical to the recomputed one (no math change)."""
    prompt = "put the cube in the bowl"
    dev = torch.device("cpu")

    monkeypatch.setattr(P, "_CACHE_PROMPT_EMBED", False)
    t_off, m_off, e_off = _make_stub_pipeline({"tok": 0, "enc": 0})._encode_positive_prompt(prompt, dev)

    monkeypatch.setattr(P, "_CACHE_PROMPT_EMBED", True)
    t_on, m_on, e_on = _make_stub_pipeline({"tok": 0, "enc": 0})._encode_positive_prompt(prompt, dev)

    assert e_on.dtype == e_off.dtype and e_on.shape == e_off.shape
    assert torch.equal(t_on, t_off)
    assert torch.equal(m_on, m_off)
    assert torch.equal(e_on, e_off)


def test_different_prompt_recomputes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A different prompt is a cache miss; distinct prompts -> distinct embeds."""
    monkeypatch.setattr(P, "_CACHE_PROMPT_EMBED", True)
    counter = {"tok": 0, "enc": 0}
    obj = _make_stub_pipeline(counter)
    dev = torch.device("cpu")

    _, _, ea = obj._encode_positive_prompt("prompt A", dev)
    _, _, eb = obj._encode_positive_prompt("prompt B different", dev)
    _, _, ea2 = obj._encode_positive_prompt("prompt A", dev)  # still cached

    assert counter["enc"] == 2  # A and B computed once each; second A is a hit
    assert torch.equal(ea, ea2)
    assert not torch.equal(ea, eb)


def test_lru_evicts_oldest(monkeypatch: pytest.MonkeyPatch) -> None:
    """The LRU is bounded; the oldest prompt is evicted and then recomputed."""
    monkeypatch.setattr(P, "_CACHE_PROMPT_EMBED", True)
    counter = {"tok": 0, "enc": 0}
    obj = _make_stub_pipeline(counter)
    dev = torch.device("cpu")
    n = P._POS_CACHE_MAX

    # Fill the cache exactly, then insert one more -> evicts the first ("p0").
    for i in range(n + 1):
        obj._encode_positive_prompt(f"prompt {i}", dev)
    assert counter["enc"] == n + 1
    assert len(obj._pos_cache) == n

    obj._encode_positive_prompt("prompt 0", dev)  # evicted -> recompute
    assert counter["enc"] == n + 2


def test_cache_is_device_keyed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A different device key forces a recompute (no cross-device tensor reuse)."""
    monkeypatch.setattr(P, "_CACHE_PROMPT_EMBED", True)
    counter = {"tok": 0, "enc": 0}
    obj = _make_stub_pipeline(counter)
    prompt = "put the cube in the bowl"

    obj._encode_positive_prompt(prompt, torch.device("cpu"))
    obj._encode_positive_prompt(prompt, torch.device("cpu"))  # hit
    assert counter["enc"] == 1

    obj._encode_positive_prompt(prompt, torch.device("meta"))  # different device
    assert counter["enc"] == 2
