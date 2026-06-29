# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Host-side bitwise-safety: per-forward scheduler build (deepcopy -> fresh).

DreamZero's ``forward()`` builds a pristine scheduler per denoise loop. The
baseline does ``copy.deepcopy(self.scheduler)`` twice per forward (host label
``sched_deepcopy``); the W7 host-opt (``VLLM_OMNI_SCHEDULER_FRESH_BUILD=1``) replaces each
deepcopy with a fresh ``FlowUniPCMultistepScheduler`` built from the same
constructor args via ``DreamZeroPipeline._make_scheduler()``. ``self.scheduler``
is built once and never mutated after ``__init__`` (only its per-forward copies
are), so fresh == deepcopy(template) bitwise.

Unlike the embed caches, this contract is *directly* checkable: scheduler
construction has no bf16/flash-attn/CUDA-graph nondeterminism.
``tests/dreamzero/test_scheduler_equiv.py`` proves the math swap standalone; this
file pins the actual pipeline wiring -- that ``_make_scheduler()`` is the single
source of truth (drift guard) and the ``_build_inference_scheduler`` flag gate
selects the right path with byte-identical schedules.

CPU-only; no GPU and no model weights required.
"""

from __future__ import annotations

import types

import pytest
import torch

from vllm_omni.diffusion.models.dreamzero import pipeline_dreamzero as P
from vllm_omni.diffusion.models.dreamzero.utils import (
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SIGMA_SHIFT,
)
from vllm_omni.diffusion.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _stub(template: FlowUniPCMultistepScheduler) -> object:
    """Minimal object exposing exactly what ``_build_inference_scheduler`` touches.

    ``scheduler`` is the pristine template the baseline deepcopies; the real
    static ``_make_scheduler`` and the real ``_build_inference_scheduler`` method
    are bound to the stub so the test exercises the actual pipeline code.
    """
    obj = types.SimpleNamespace()
    obj.scheduler = template
    obj._make_scheduler = P.DreamZeroPipeline._make_scheduler  # staticmethod
    obj._build_inference_scheduler = P.DreamZeroPipeline._build_inference_scheduler.__get__(obj)
    return obj


def _schedule(sched: FlowUniPCMultistepScheduler) -> FlowUniPCMultistepScheduler:
    sched.set_timesteps(DEFAULT_NUM_INFERENCE_STEPS, device="cpu", shift=DEFAULT_SIGMA_SHIFT)
    return sched


def test_make_scheduler_matches_documented_template() -> None:
    """Drift guard: the real ``_make_scheduler()`` must equal the documented
    ``__init__`` template (``num_train_timesteps=1000, shift=1,
    use_dynamic_shifting=False``).

    The standalone math test reconstructs its own template; this ties that
    contract to the *actual* pipeline helper so a constructor-arg change there
    cannot silently break the deepcopy == fresh equivalence.
    """
    documented = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1,
        use_dynamic_shifting=False,
    )
    real = P.DreamZeroPipeline._make_scheduler()

    base = _schedule(documented)
    cand = _schedule(real)
    assert torch.equal(base.timesteps, cand.timesteps)
    assert base.timesteps.dtype == cand.timesteps.dtype
    assert torch.equal(base.sigmas, cand.sigmas)
    assert base.sigmas.dtype == cand.sigmas.dtype


def test_off_path_deepcopies_template(monkeypatch: pytest.MonkeyPatch) -> None:
    """W7 OFF == original path: returns a deepcopy of ``self.scheduler`` (a
    distinct object), so set_timesteps on it leaves the template unmutated."""
    monkeypatch.setattr(P, "_SCHEDULER_FRESH_BUILD", False)
    template = P.DreamZeroPipeline._make_scheduler()
    obj = _stub(template)

    built = obj._build_inference_scheduler()
    assert built is not template
    assert template.num_inference_steps is None

    _schedule(built)  # mutates the copy only
    assert template.num_inference_steps is None  # template stays pristine


def test_on_path_builds_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """W7 ON: returns a freshly constructed scheduler, not aliased to the
    template (so mutating it cannot leak into ``self.scheduler`` either)."""
    monkeypatch.setattr(P, "_SCHEDULER_FRESH_BUILD", True)
    template = P.DreamZeroPipeline._make_scheduler()
    obj = _stub(template)

    built = obj._build_inference_scheduler()
    assert built is not template

    _schedule(built)
    assert template.num_inference_steps is None


def test_on_equals_off_bitwise(monkeypatch: pytest.MonkeyPatch) -> None:
    """fresh-build (ON) and deepcopy (OFF) produce byte-identical schedules.

    This is the bitwise contract that makes the host-opt invisible to outputs:
    the per-forward scheduler's ``timesteps``/``sigmas`` are identical either way.
    """
    template = P.DreamZeroPipeline._make_scheduler()

    monkeypatch.setattr(P, "_SCHEDULER_FRESH_BUILD", False)
    off = _schedule(_stub(template)._build_inference_scheduler())

    monkeypatch.setattr(P, "_SCHEDULER_FRESH_BUILD", True)
    on = _schedule(_stub(template)._build_inference_scheduler())

    assert torch.equal(off.timesteps, on.timesteps)
    assert off.timesteps.dtype == on.timesteps.dtype
    assert torch.equal(off.sigmas, on.sigmas)
    assert off.sigmas.dtype == on.sigmas.dtype


def test_two_builds_are_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    """forward() builds the video + action schedulers from the same template;
    mutating one must not affect the other (no shared mutable state)."""
    monkeypatch.setattr(P, "_SCHEDULER_FRESH_BUILD", True)
    obj = _stub(P.DreamZeroPipeline._make_scheduler())

    a = _schedule(obj._build_inference_scheduler())
    b = obj._build_inference_scheduler()

    assert a.num_inference_steps == DEFAULT_NUM_INFERENCE_STEPS
    assert b.num_inference_steps is None  # a's set_timesteps did not leak into b
