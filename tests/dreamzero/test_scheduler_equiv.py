# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Host-side bitwise-safety: scheduler construction equivalence.

DreamZero's forward() currently does ``copy.deepcopy(self.scheduler)`` twice per
step and then calls ``set_timesteps(...)`` on each copy. ``self.scheduler`` is
built once in ``__init__`` and is never mutated afterwards (only its per-step
copies are), so the host-side optimization replaces the per-step deepcopy with a
fresh ``FlowUniPCMultistepScheduler(...)`` built from the same constructor args.

These tests prove that swap is *bitwise-identical*: a deepcopy of the pristine
template followed by ``set_timesteps`` produces exactly the same ``timesteps``
and ``sigmas`` as a freshly constructed scheduler followed by ``set_timesteps``,
including the ``decouple_inference_noise`` sigma rescale applied afterwards
(``pipeline_dreamzero.py:930-936``).

CPU-only; runs without a GPU (set_timesteps targets device="cpu").
"""

from __future__ import annotations

import copy

import pytest
import torch

from vllm_omni.diffusion.models.dreamzero.utils import (
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SIGMA_SHIFT,
)
from vllm_omni.diffusion.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_template() -> FlowUniPCMultistepScheduler:
    """Mirror DreamZeroPipeline.__init__ scheduler construction (pipeline:198)."""
    return FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1,
        use_dynamic_shifting=False,
    )


def _deepcopy_path(num_inference_steps: int, shift: float) -> FlowUniPCMultistepScheduler:
    """Baseline: copy.deepcopy(template) then set_timesteps (current code)."""
    template = _make_template()
    sched = copy.deepcopy(template)
    sched.set_timesteps(num_inference_steps, device="cpu", shift=shift)
    return sched


def _fresh_path(num_inference_steps: int, shift: float) -> FlowUniPCMultistepScheduler:
    """W7 optimized: construct a new scheduler then set_timesteps (no deepcopy)."""
    sched = _make_template()
    sched.set_timesteps(num_inference_steps, device="cpu", shift=shift)
    return sched


@pytest.mark.parametrize(
    "num_inference_steps, shift",
    [
        (DEFAULT_NUM_INFERENCE_STEPS, DEFAULT_SIGMA_SHIFT),
        (1, DEFAULT_SIGMA_SHIFT),
        (4, 1.0),
        (16, 3.0),
        (50, 7.0),
    ],
)
def test_deepcopy_vs_fresh_is_bitwise_identical(num_inference_steps: int, shift: float) -> None:
    """deepcopy(template)+set_timesteps == fresh()+set_timesteps, exactly."""
    base = _deepcopy_path(num_inference_steps, shift)
    cand = _fresh_path(num_inference_steps, shift)

    assert torch.equal(base.timesteps, cand.timesteps)
    assert base.timesteps.dtype == cand.timesteps.dtype
    assert torch.equal(base.sigmas, cand.sigmas)
    assert base.sigmas.dtype == cand.sigmas.dtype


def test_template_is_reusable_across_two_copies() -> None:
    """The pipeline deepcopies the same template twice per step (video + action).

    Both copies must yield identical schedules, confirming the template stays
    pristine and a single shared template is safe to construct N schedulers from.
    """
    template = _make_template()

    a = copy.deepcopy(template)
    a.set_timesteps(DEFAULT_NUM_INFERENCE_STEPS, device="cpu", shift=DEFAULT_SIGMA_SHIFT)
    b = copy.deepcopy(template)
    b.set_timesteps(DEFAULT_NUM_INFERENCE_STEPS, device="cpu", shift=DEFAULT_SIGMA_SHIFT)

    assert torch.equal(a.timesteps, b.timesteps)
    assert torch.equal(a.sigmas, b.sigmas)


def test_decouple_inference_noise_rescale_is_identical() -> None:
    """Equivalence holds through the decouple_inference_noise sigma rescale.

    Mirrors pipeline_dreamzero.py:930-936, which rescales sigmas and recomputes
    int64 timesteps after set_timesteps. Applied identically to both paths, the
    result must stay bitwise-identical regardless of how the scheduler was built.
    """
    video_final_noise = 0.05  # representative; any constant works for equivalence

    def _apply_decouple(sched: FlowUniPCMultistepScheduler) -> None:
        sigma_max = sched.sigmas[0].item()
        sched.sigmas = sched.sigmas * (sigma_max - video_final_noise) / sigma_max + video_final_noise
        sched.timesteps = (sched.sigmas[:-1] * 1000).to(torch.int64)

    base = _deepcopy_path(DEFAULT_NUM_INFERENCE_STEPS, DEFAULT_SIGMA_SHIFT)
    cand = _fresh_path(DEFAULT_NUM_INFERENCE_STEPS, DEFAULT_SIGMA_SHIFT)
    _apply_decouple(base)
    _apply_decouple(cand)

    assert torch.equal(base.timesteps, cand.timesteps)
    assert torch.equal(base.sigmas, cand.sigmas)
