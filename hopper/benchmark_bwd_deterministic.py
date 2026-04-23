"""Benchmark the deterministic FA3 backward vs the non-deterministic prod path.

Measures the cost of enabling deterministic=True in flash_attn_func at
head_dim=256 on H100/H200. Representative shapes include GQA, causal, and
sliding-window-attention (SWA) — the case where our smaller-tile
deterministic config pays the highest overhead because each CTA must
arrive_inc for all m_blocks past its local window to keep later CTAs from
deadlocking.

Usage:
    cd hopper && pip install -e . --no-build-isolation  # build FA3 first
    PYTHONPATH=<torch-site-packages>:. python benchmark_bwd_deterministic.py
"""

import torch

from flash_attn_interface import flash_attn_func


def bench(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]
    for a, b in events:
        a.record()
        fn()
        b.record()
    torch.cuda.synchronize()
    times = sorted(a.elapsed_time(b) for a, b in events)
    return times[iters // 2]  # median


def bench_fwd_bwd(q, k, v, causal, window_size, deterministic):
    def step():
        if q.grad is not None:
            q.grad = None
        if k.grad is not None:
            k.grad = None
        if v.grad is not None:
            v.grad = None
        out, _ = flash_attn_func(
            q,
            k,
            v,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            return_attn_probs=True,
        )
        g = torch.randn_like(out)
        out.backward(g)

    return bench(step)


def cfg(label, *, batch, seqlen, heads, kv_heads, head_dim, causal, window):
    torch.manual_seed(0)
    q = torch.randn(
        batch, seqlen, heads, head_dim, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    k = torch.randn(
        batch, seqlen, kv_heads, head_dim, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    v = torch.randn(
        batch, seqlen, kv_heads, head_dim, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    nondet = bench_fwd_bwd(q, k, v, causal=causal, window_size=window, deterministic=False)
    det = bench_fwd_bwd(q, k, v, causal=causal, window_size=window, deterministic=True)
    ratio = det / nondet
    print(f"{label:30s} non-det: {nondet:7.2f}ms   det: {det:7.2f}ms   ratio: {ratio:4.2f}x")


def main():
    print("FA3 hdim=256 backward benchmark (bf16)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Small shape — overhead should be negligible
    cfg("mha 256x256 causal",
        batch=2, seqlen=256, heads=6, kv_heads=6, head_dim=256,
        causal=True, window=(-1, -1))
    cfg("gqa 256x256 causal",
        batch=2, seqlen=256, heads=6, kv_heads=2, head_dim=256,
        causal=True, window=(-1, -1))

    # Medium — representative of training steps
    cfg("mha 2048x2048 causal",
        batch=2, seqlen=2048, heads=6, kv_heads=6, head_dim=256,
        causal=True, window=(-1, -1))
    cfg("gqa 2048x2048 causal",
        batch=2, seqlen=2048, heads=6, kv_heads=2, head_dim=256,
        causal=True, window=(-1, -1))

    # Long-seq — the regime most sensitive to kBlockN shrink
    cfg("gqa 4096x4096 causal",
        batch=1, seqlen=4096, heads=6, kv_heads=2, head_dim=256,
        causal=True, window=(-1, -1))
    cfg("gqa 4096x4096 SWA=1024",
        batch=1, seqlen=4096, heads=6, kv_heads=2, head_dim=256,
        causal=True, window=(1024, 0))


if __name__ == "__main__":
    main()
