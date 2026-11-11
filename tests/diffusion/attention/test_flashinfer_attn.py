# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends import flashinfer_attn
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flashinfer_attn import (
    FlashInferAttentionBackend,
    FlashInferAttentionImpl,
)
from vllm_omni.diffusion.data import AttentionSpec, AttnQuantSpec

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _impl(*, causal: bool = False, backend_explicit: bool = False):
    # Avoid CUDA/wrapper init with an initialized-state double.
    obj = FlashInferAttentionImpl.__new__(FlashInferAttentionImpl)
    obj.causal = causal
    obj.softmax_scale = 0.5
    obj.flashinfer_backend = "fa2"
    obj.dtype_qk = None
    obj.dtype_vo = None
    obj.backend_explicit = backend_explicit
    obj._sdpa_fallback = None
    # __new__ bypasses __init__; the mask reject/fallback decisions run inside
    # the boundary op now, so those paths need the op token __init__ would
    # have built.
    obj._op_handle = flashinfer_attn._register_impl_for_boundary_op(obj)
    obj._op_token = torch.tensor(obj._op_handle, dtype=torch.int64)
    return obj


@pytest.mark.cpu
def test_flashinfer_rejects_float_mask_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    query = torch.randn(1, 2, 2, 8)
    metadata = AttentionMetadata(attn_mask=torch.zeros(2, 2))

    with pytest.raises(ValueError, match="boolean-only"):
        _impl(backend_explicit=True).forward_cuda(query, query, query, metadata)


@pytest.mark.cpu
def test_flashinfer_rejects_causal_custom_mask_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    query = torch.randn(1, 2, 2, 8)
    metadata = AttentionMetadata(attn_mask=torch.tensor([[True, False], [True, True]]))

    with pytest.raises(ValueError, match="causal=True"):
        _impl(causal=True, backend_explicit=True).forward_cuda(query, query, query, metadata)


@pytest.mark.cpu
def test_explicit_cute_dsl_rejects_custom_mask_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    impl = _impl(backend_explicit=True)
    impl.flashinfer_backend = "cute-dsl"
    query = torch.randn(1, 2, 2, 8)
    metadata = AttentionMetadata(attn_mask=torch.tensor([[True, False], [True, True]]))

    with pytest.raises(ValueError, match="cute-dsl"):
        impl.forward_cuda(query, query, query, metadata)


@pytest.mark.cpu
def test_auto_cute_dsl_falls_back_to_sdpa_for_custom_mask(monkeypatch):
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    impl = _impl(backend_explicit=False)
    impl.flashinfer_backend = "cute-dsl"
    impl._run_batch_prefill = lambda *args, **kwargs: pytest.fail("unexpected cute-dsl prefill")
    query = torch.randn(1, 2, 2, 8)
    metadata = AttentionMetadata(attn_mask=torch.tensor([[True, False], [True, True]]))

    output = impl.forward_cuda(query, query, query, metadata)

    assert output.shape == query.shape


@pytest.mark.cpu
def test_explicit_cute_dsl_is_not_mask_capable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (10, 0))
    spec = AttentionSpec(backend="FLASHINFER_ATTN")

    assert FlashInferAttentionBackend.supports_attention_mask() is True
    assert FlashInferAttentionBackend.supports_attention_mask(spec) is False


@pytest.mark.cpu
def test_explicit_fa2_flashinfer_is_mask_capable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (10, 0))
    spec = AttentionSpec(backend="FLASHINFER_ATTN", quant=AttnQuantSpec(flashinfer_backend="fa2"))

    assert FlashInferAttentionBackend.supports_attention_mask(spec) is True


@pytest.mark.cpu
def test_explicit_fa2_backend_kwargs_select_fa2_on_blackwell(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (10, 0))
    spec = AttentionSpec(backend="FLASHINFER_ATTN", quant=AttnQuantSpec(flashinfer_backend="fa2"))
    kwargs = spec.backend_kwargs() or {}
    requested = (kwargs.get("quant") or {}).get("flashinfer_backend", "auto")

    assert kwargs == {"quant": {"flashinfer_backend": "fa2"}}
    assert FlashInferAttentionImpl._select_backend(requested) == "fa2"
    assert FlashInferAttentionBackend.supports_attention_mask(spec) is True


@pytest.mark.cpu
def test_auto_flashinfer_backend_kwargs_select_cute_dsl_on_blackwell(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (10, 0))
    spec = AttentionSpec(backend="FLASHINFER_ATTN")
    kwargs = spec.backend_kwargs() or {}
    requested = (kwargs.get("quant") or {}).get("flashinfer_backend", "auto")

    assert "quant" not in kwargs
    assert FlashInferAttentionImpl._select_backend(requested) == "cute-dsl"
    assert FlashInferAttentionBackend.supports_attention_mask(spec) is False


@pytest.mark.cpu
def test_hopper_explicit_flashinfer_is_mask_capable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (9, 0))
    spec = AttentionSpec(backend="FLASHINFER_ATTN")

    assert FlashInferAttentionBackend.supports_attention_mask(spec) is True


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("backend", "k_dtype", "v_dtype"),
    [
        ("fa2", torch.bfloat16, torch.float8_e4m3fn),
        ("fa3", torch.bfloat16, torch.float8_e4m3fn),
        ("fa2", None, torch.float8_e4m3fn),
        ("fa2", torch.float16, torch.bfloat16),
    ],
)
def test_single_kv_dtype_backend_rejects_mixed_kv_dtypes(backend, k_dtype, v_dtype):
    with pytest.raises(ValueError, match="reads V with the K dtype"):
        FlashInferAttentionImpl._check_kv_dtypes(backend, k_dtype, v_dtype)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("backend", "k_dtype", "v_dtype"),
    [
        ("cute-dsl", torch.bfloat16, torch.float8_e4m3fn),
        ("fa2", torch.bfloat16, torch.bfloat16),
        ("fa3", torch.float16, torch.float16),
    ],
)
def test_kv_dtype_check_allows_supported_combinations(backend, k_dtype, v_dtype):
    FlashInferAttentionImpl._check_kv_dtypes(backend, k_dtype, v_dtype)


@pytest.mark.cpu
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
@pytest.mark.parametrize("dtype_qk", ["bfloat16", None])
def test_init_rejects_fp8_v_on_single_kv_dtype_backend(monkeypatch, backend, dtype_qk):
    # Stop before CUDA allocation: the check runs right after backend selection.
    monkeypatch.setattr(flashinfer_attn, "HAS_FLASHINFER", True)
    monkeypatch.setattr(flashinfer_attn, "flashinfer", SimpleNamespace(__version__="0.7.2"), raising=False)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

    with pytest.raises(ValueError, match="reads V with the K dtype"):
        FlashInferAttentionImpl(
            num_heads=2,
            head_size=128,
            softmax_scale=128**-0.5,
            backend_kwargs={
                "quant": {"flashinfer_backend": backend, "dtype_qk": dtype_qk, "dtype_vo": "fp8_e4m3"},
            },
        )


@pytest.mark.cpu
def test_plan_rejects_mixed_kv_dtypes_on_fa2():
    # dtype_qk alone cannot be checked at init: V keeps the input dtype.
    impl = _impl()
    impl._qo_indptr = torch.zeros(2, dtype=torch.int32)
    impl._kv_indptr = torch.zeros(2, dtype=torch.int32)
    impl._wrapper = SimpleNamespace(plan=lambda *args, **kwargs: pytest.fail("unexpected plan"))
    key = FlashInferAttentionImpl._WrapperPlanKey(
        batch_size=1,
        qo_len=4,
        kv_len=4,
        num_q_heads=2,
        num_kv_heads=2,
        head_dim_qk=8,
        head_dim_k=8,
        head_dim_vo=8,
        q_dtype=torch.float16,
        k_dtype=torch.float16,
        v_dtype=torch.bfloat16,
        o_dtype=torch.bfloat16,
        causal=False,
        softmax_scale=0.5,
        has_custom_mask=False,
    )

    with pytest.raises(ValueError, match="reads V with the K dtype"):
        impl._plan_wrapper(key, None)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_flashinfer_real_init_selects_fa2_on_sm80():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for real FlashInfer initialization")
    if not flashinfer_attn.HAS_FLASHINFER:
        pytest.skip("FlashInfer is not installed")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("This regression test targets A100/SM80")

    impl = FlashInferAttentionImpl(
        num_heads=2,
        head_size=128,
        softmax_scale=128**-0.5,
        backend_kwargs={"quant": {"flashinfer_backend": "auto"}},
    )

    assert impl.flashinfer_backend == "fa2"
