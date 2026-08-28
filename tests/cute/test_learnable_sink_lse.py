# Copyright (c) 2026, Tri Dao.

import math

import pytest
import torch
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
from flash_attn.cute.interface import flash_attn_func


def _normalized_bf16_normal(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    tensor = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
    tensor_float = tensor.float()
    rms_square = (tensor_float * tensor_float).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(rms_square.clamp_min(torch.finfo(torch.float32).tiny))
    return (tensor_float * inv_rms).to(dtype=tensor.dtype)


def _fp64_sink_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor,
    softmax_scale: float,
    *,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_ref = q.double()
    k_ref = k.double().repeat_interleave(q.shape[2] // k.shape[2], dim=2)
    v_ref = v.double().repeat_interleave(q.shape[2] // v.shape[2], dim=2)
    scores = torch.einsum("bthd,bshd->bhts", q_ref * softmax_scale, k_ref)
    if causal:
        causal_mask = torch.triu(
            torch.ones(q.shape[1], k.shape[1], dtype=torch.bool, device=q.device),
            diagonal=1,
        )
        scores.masked_fill_(causal_mask.view(1, 1, q.shape[1], k.shape[1]), -math.inf)
    sink_scores = (
        sink.double().view(1, q.shape[2], 1, 1).expand(q.shape[0], q.shape[2], q.shape[1], 1)
    )
    scores_with_sink = torch.cat((scores, sink_scores), dim=-1)
    probability = torch.softmax(scores_with_sink, dim=-1)[..., :-1]
    output = torch.einsum("bhts,bshd->bthd", probability, v_ref)
    return output.float(), torch.logsumexp(scores_with_sink, dim=-1).float()


def _max_positive_float32_ulp_distance(actual: torch.Tensor, expected: torch.Tensor) -> int:
    assert torch.all(actual > 0.0)
    assert torch.all(expected > 0.0)
    actual_bits = actual.contiguous().view(torch.int32).to(torch.int64)
    expected_bits = expected.contiguous().view(torch.int32).to(torch.int64)
    return int((actual_bits - expected_bits).abs().max())


def _similarity_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.detach().double().flatten() + 1
    expected = expected.detach().double().flatten() + 1
    denominator = (actual * actual + expected * expected).sum()
    if denominator == 0:
        return 0.0
    similarity = 2 * (actual * expected).sum() / denominator
    return (1 - similarity).item()


def _assert_grad_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    error = _similarity_error(actual, expected)
    assert error <= 1e-4, f"{name} gradient similarity error is {error}"


@pytest.mark.parametrize(
    ("q_seed", "softmax_scale"),
    [(27, None), (42, None), (42, 1e-8)],
    ids=["hopper_regression", "blackwell_regression", "tiny_scale"],
)
def test_learnable_sink_lse_uses_sink_in_final_max(
    q_seed: int, softmax_scale: float | None
) -> None:
    batch, seqlen, nheads, nheads_kv, headdim = 2, 64, 8, 2, 128
    q = _normalized_bf16_normal((batch, seqlen, nheads, headdim), q_seed)
    k = _normalized_bf16_normal((batch, seqlen, nheads_kv, headdim), q_seed + 1)
    torch.manual_seed(q_seed + 2)
    torch.cuda.manual_seed(q_seed + 2)
    v = torch.randn(
        batch,
        seqlen,
        nheads_kv,
        headdim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    sink = torch.full(
        (nheads,),
        2.0 * math.sqrt(2.0 * math.log(seqlen)),
        dtype=torch.float32,
        device="cuda",
    )

    out, lse = flash_attn_func(
        q,
        k,
        v,
        causal=True,
        learnable_sink=sink,
        softmax_scale=softmax_scale,
        return_lse=True,
    )
    reference_scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(headdim)
    out_ref, lse_ref = _fp64_sink_reference(q, k, v, sink, reference_scale)

    max_ulp_distance = _max_positive_float32_ulp_distance(lse, lse_ref)
    assert max_ulp_distance <= 2, f"LSE differs from the fp64 reference by {max_ulp_distance} ULP"
    torch.testing.assert_close(out.float(), out_ref, rtol=2e-2, atol=2e-2)


def test_learnable_sink_dominant_backward_uses_saved_lse() -> None:
    batch, seqlen, nheads, headdim = 1, 128, 4, 128
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    q = torch.randn(
        batch,
        seqlen,
        nheads,
        headdim,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    sink = torch.full(
        (nheads,),
        6.0,
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )

    out, lse = flash_attn_func(q, k, v, learnable_sink=sink, return_lse=False)

    q_ref = q.detach().double().requires_grad_()
    k_ref = k.detach().double().requires_grad_()
    v_ref = v.detach().double().requires_grad_()
    sink_ref = sink.detach().double().requires_grad_()
    out_ref, lse_ref = _fp64_sink_reference(
        q_ref,
        k_ref,
        v_ref,
        sink_ref,
        1.0 / math.sqrt(headdim),
        causal=False,
    )

    assert torch.isfinite(lse).all()
    torch.testing.assert_close(out.float(), out_ref, rtol=2e-2, atol=2e-2)
    max_ulp_distance = _max_positive_float32_ulp_distance(lse, lse_ref)
    assert max_ulp_distance <= 2, f"LSE differs from the fp64 reference by {max_ulp_distance} ULP"

    dout = torch.randn_like(out)
    grads = torch.autograd.grad((out * dout).sum(), (q, k, v, sink))
    grads_ref = torch.autograd.grad(
        (out_ref * dout.float()).sum(),
        (q_ref, k_ref, v_ref, sink_ref),
    )
    for name, grad, grad_ref in zip(("dq", "dk", "dv", "dsink"), grads, grads_ref):
        _assert_grad_close(name, grad, grad_ref)


def test_learnable_sink_overflow_backward_stays_zero() -> None:
    """Keep backward zero when the old-frame sink exponential overflows."""
    batch, seqlen, nheads, headdim = 1, 1, 8, 128
    sink_gaps = torch.tensor(
        [80.0, 84.0, 86.0, 87.0, 88.0, 89.0, 90.0, 92.0],
        dtype=torch.float32,
        device="cuda",
    )
    q = torch.zeros(batch, seqlen, nheads, headdim, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros_like(q)
    q[..., 0] = 1.0
    k[..., 0] = 1.0
    q.requires_grad_()
    k.requires_grad_()
    v = torch.ones_like(q, requires_grad=True)
    sink = (1.0 + sink_gaps).requires_grad_()

    out, lse = flash_attn_func(
        q,
        k,
        v,
        softmax_scale=1.0,
        learnable_sink=sink,
        return_lse=False,
    )
    dout = torch.zeros_like(out)
    dout[..., 0] = 1.0
    dq, dk, dv, dsink = torch.autograd.grad((out * dout).sum(), (q, k, v, sink))

    forward_probability = out[0, 0, :, 0].float()
    backward_probability = dv[0, 0, :, 0].float()
    nonzero = forward_probability != 0.0
    zero = ~nonzero
    assert nonzero.any(), "range sweep did not retain any token probabilities"
    assert zero.any(), "range sweep did not flush any token probabilities"
    assert torch.count_nonzero(forward_probability[sink_gaps >= 89.0]) == 0, (
        "token probability did not flush after the old-frame sink exponential "
        "exceeded float32 range"
    )

    torch.testing.assert_close(
        backward_probability[nonzero],
        forward_probability[nonzero],
        rtol=1e-2,
        atol=0.0,
    )
    assert torch.count_nonzero(backward_probability[zero]) == 0

    expected_probability = torch.sigmoid(-sink_gaps.double()).float()
    expected_score_gradient = expected_probability * (1.0 - expected_probability)
    for name, actual, expected in (
        ("output", forward_probability, expected_probability),
        ("dv", backward_probability, expected_probability),
        ("dq", dq[0, 0, :, 0].float(), expected_score_gradient),
        ("dk", dk[0, 0, :, 0].float(), expected_score_gradient),
        ("negative dsink", -dsink.float(), expected_score_gradient),
    ):
        torch.testing.assert_close(
            actual[nonzero],
            expected[nonzero],
            rtol=5e-3,
            atol=0.0,
            msg=lambda message, name=name: f"{name}: {message}",
        )
        assert torch.count_nonzero(actual[zero]) == 0, (
            f"{name} remained nonzero where forward token probability flushed"
        )

    assert torch.isfinite(lse).all()
    max_ulp_distance = _max_positive_float32_ulp_distance(lse[0, :, 0], sink)
    assert max_ulp_distance <= 1, (
        f"extreme-sink LSE differs from the sink by {max_ulp_distance} ULP"
    )


@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] not in [10, 11],
    reason="Split-KV is only supported on SM100 and SM110",
)
def test_learnable_sink_split_kv_lse() -> None:
    batch, seqlen, nheads, nheads_kv, headdim = 1, 384, 8, 2, 128
    q = _normalized_bf16_normal((batch, seqlen, nheads, headdim), 81)
    k = _normalized_bf16_normal((batch, seqlen, nheads_kv, headdim), 82)
    torch.manual_seed(83)
    torch.cuda.manual_seed(83)
    v = torch.randn(
        batch,
        seqlen,
        nheads_kv,
        headdim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    sink = torch.full((nheads,), 6.0, dtype=torch.float32, device="cuda")

    out, lse = flash_attn_func(
        q,
        k,
        v,
        learnable_sink=sink,
        num_splits=3,
        return_lse=True,
    )
    out_ref, lse_ref = _fp64_sink_reference(
        q,
        k,
        v,
        sink,
        1.0 / math.sqrt(headdim),
        causal=False,
    )

    max_ulp_distance = _max_positive_float32_ulp_distance(lse, lse_ref)
    assert max_ulp_distance <= 2, (
        f"Split-KV LSE differs from the fp64 reference by {max_ulp_distance} ULP"
    )
    torch.testing.assert_close(out.float(), out_ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] not in [10, 11],
    reason="The modified block-sparse correction path is SM100/SM110-only",
)
def test_learnable_sink_nonempty_block_sparse_lse() -> None:
    batch, seqlen_q, seqlen_k, nheads, headdim = 1, 256, 128, 8, 128
    q = _normalized_bf16_normal((batch, seqlen_q, nheads, headdim), 91)
    k = _normalized_bf16_normal((batch, seqlen_k, nheads, headdim), 92)
    torch.manual_seed(93)
    torch.cuda.manual_seed(93)
    v = torch.randn(
        batch,
        seqlen_k,
        nheads,
        headdim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    sink = torch.full((nheads,), 6.0, dtype=torch.float32, device="cuda")
    block_count_shape = (batch, nheads, 1)
    block_index_shape = (*block_count_shape, 1)
    sparse = BlockSparseTensorsTorch(
        mask_block_cnt=torch.zeros(block_count_shape, dtype=torch.int32, device="cuda"),
        mask_block_idx=torch.zeros(block_index_shape, dtype=torch.int32, device="cuda"),
        full_block_cnt=torch.ones(block_count_shape, dtype=torch.int32, device="cuda"),
        full_block_idx=torch.zeros(block_index_shape, dtype=torch.int32, device="cuda"),
        block_size=(256, 128),
    )

    out, lse = flash_attn_func(
        q,
        k,
        v,
        learnable_sink=sink,
        block_sparse_tensors=sparse,
        return_lse=True,
    )
    out_ref, lse_ref = _fp64_sink_reference(
        q,
        k,
        v,
        sink,
        1.0 / math.sqrt(headdim),
        causal=False,
    )

    max_ulp_distance = _max_positive_float32_ulp_distance(lse, lse_ref)
    assert max_ulp_distance <= 2, (
        f"Block-sparse LSE differs from the fp64 reference by {max_ulp_distance} ULP"
    )
    torch.testing.assert_close(out.float(), out_ref, rtol=2e-2, atol=2e-2)
