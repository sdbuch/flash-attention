"""Run one FA3 backward in a profiler-friendly shape.

Used by the ncu / nsys wrappers to get per-kernel metrics on the
deterministic-vs-non-deterministic backward at hdim=256.

Two modes:

- `--mode warmup_then_one`: do `--warmup` calls (untimed), then exactly one
  timed call. Works with `ncu --launch-skip <warmup> --launch-count 1` to
  profile only the fresh, post-warmup launch.

- `--mode loop`: do `--iters` calls. For nsys timeline traces.

We deliberately call `_flash_attn_backward` directly (not
`flash_attn_func.backward`) so the only kernels in the trace are the ones
we care about: bwd preprocess, mainloop, postprocess. The fwd is run once
during setup, before any profiling iteration.

Usage:
    python profile_det_bwd.py \\
        --hdim 256 --batch 1 --seqlen 4096 --heads 6 --kv-heads 2 \\
        --window 1024 --deterministic --warmup 5 --iters 1
"""

from __future__ import annotations

import argparse
import time

import torch

from flash_attn_interface import _flash_attn_backward, flash_attn_func


def setup(args) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    q = torch.randn(args.batch, args.seqlen, args.heads, args.hdim,
                    dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.randn(args.batch, args.seqlen, args.kv_heads, args.hdim,
                    dtype=torch.bfloat16, device="cuda", requires_grad=True)
    v = torch.randn(args.batch, args.seqlen, args.kv_heads, args.hdim,
                    dtype=torch.bfloat16, device="cuda", requires_grad=True)
    out, softmax_lse = flash_attn_func(
        q, k, v,
        causal=args.causal,
        window_size=(args.window, 0) if args.window > 0 else (-1, -1),
        deterministic=args.deterministic,
        return_attn_probs=True,
    )
    g = torch.randn_like(out)
    return q, k, v, out, softmax_lse, g


def call_bwd(q, k, v, out, softmax_lse, g, args, dq, dk, dv):
    """Single backward call. dq/dk/dv are pre-allocated, modified in place."""
    _flash_attn_backward(
        g, q, k, v, out, softmax_lse,
        None, None,  # cu_seqlens
        None, None,  # seqused
        None, None,  # max_seqlens
        dq, dk, dv,
        args.hdim ** -0.5,
        args.causal,
        args.window if args.window > 0 else -1,  # window_size_left
        0 if args.window > 0 else -1,             # window_size_right
        0.0,                                       # softcap
        args.deterministic,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hdim", type=int, default=256)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seqlen", type=int, default=4096)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--causal", action="store_true", default=True)
    p.add_argument("--no-causal", dest="causal", action="store_false")
    p.add_argument("--window", type=int, default=-1,
                   help=">0 enables causal SWA with that lookback. -1 disables.")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--mode", choices=["warmup_then_one", "loop"],
                   default="warmup_then_one")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()

    print(
        f"hdim={args.hdim} seqlen={args.seqlen} heads={args.heads}/{args.kv_heads}"
        f" causal={args.causal} window={args.window} det={args.deterministic}"
        f" mode={args.mode}"
    )

    q, k, v, out, softmax_lse, g = setup(args)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    if args.mode == "warmup_then_one":
        for _ in range(args.warmup):
            call_bwd(q, k, v, out, softmax_lse, g, args, dq, dk, dv)
        torch.cuda.synchronize()
        # profiled launch
        t0 = time.perf_counter()
        call_bwd(q, k, v, out, softmax_lse, g, args, dq, dk, dv)
        torch.cuda.synchronize()
        print(f"single timed bwd: {(time.perf_counter() - t0) * 1000:.3f} ms")
    else:
        # Some warmup so the timeline doesn't include first-launch JIT noise
        for _ in range(args.warmup):
            call_bwd(q, k, v, out, softmax_lse, g, args, dq, dk, dv)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            call_bwd(q, k, v, out, softmax_lse, g, args, dq, dk, dv)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        print(f"{args.iters} iters total: {dt:.3f} ms ({dt / args.iters:.3f} ms/iter)")


if __name__ == "__main__":
    main()
