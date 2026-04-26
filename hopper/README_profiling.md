# Kernel-level profiling for FA3 backward

Tools for understanding *why* the deterministic FA3 backward at hdim=256 is
slow on SWA, and CUDA-kernel profiling more generally.

## Files

- `profile_det_bwd.py` — Python harness that runs one or N backward calls
  in a profiler-friendly shape.
- `profile_det_bwd.sh` — wrapper that runs ncu deep-profile + nsys timeline
  for both deterministic and non-deterministic at the same shape, then
  invokes the diff script.
- `profile_diff.py` — pulls a curated metric set out of two ncu reports and
  prints them side-by-side.

## Quick start (on devpod)

```bash
cd ~/flash-attention/hopper
bash profile_det_bwd.sh
```

That will produce `profile_out/`:

```
ncu_nondet.ncu-rep   ncu_det.ncu-rep      ← deep per-kernel profile
nsys_nondet.nsys-rep nsys_det.nsys-rep    ← timeline trace
```

…and print a side-by-side metric comparison to stdout.

To target a different shape:

```bash
SHAPE_ARGS="--hdim 256 --batch 1 --seqlen 4096 --heads 6 --kv-heads 2 --window 512" \
  bash profile_det_bwd.sh
```

## ncu (Nsight Compute) — per-kernel deep dive

`ncu` collects detailed metrics for individual kernel launches. When
`profile_det_bwd.sh` runs ncu it:
- skips `--launch-skip 5` warmup launches,
- captures the next ~100 kernel launches,
- filters to kernels matching `--kernel-name regex:flash` (skip torch
  internals like rng, copy, etc),
- collects the `--set full` metric set (occupancy, stalls, bandwidth,
  pipe utilization — about 100 metrics).

Open the report in the Nsight Compute UI for an interactive view:

```bash
# Local mac:
scp devpod:~/flash-attention/hopper/profile_out/ncu_det.ncu-rep ~/
# Then either:
ncu-ui ~/ncu_det.ncu-rep                # if installed locally
# Or run the GUI on the devpod and X-forward.
```

CLI inspection (no UI needed):

```bash
ncu --import profile_out/ncu_det.ncu-rep --print-summary per-kernel
ncu --import profile_out/ncu_det.ncu-rep --section SchedulerStats   # warp stalls
ncu --import profile_out/ncu_det.ncu-rep --section MemoryWorkloadAnalysis
ncu --import profile_out/ncu_det.ncu-rep --csv --metrics smsp__warp_issue_stalled_barrier_per_warp_active.pct
```

### What to look for first

Given our hypothesis that the SWA cost is dominated by tail-arrive
barrier work + many CTAs:

1. **`smsp__warp_issue_stalled_barrier_per_warp_active.pct`** — fraction
   of warp-issue cycles spent stalled at a barrier. **Hypothesis: high
   for det, low for non-det.** If it's 30%+ that's a big confirmation.

2. **`launch__grid_size`** — number of CTAs. Should be **2.5x larger for
   det** at hdim=256 (768 vs ~308 for our SWA shape), zero difference at
   hdim=128.

3. **`sm__warps_active.avg.pct_of_peak_sustained_active`** (achieved
   occupancy). If det's occupancy *drops* relative to non-det even with
   more CTAs, that's a sign that the barriers are forcing serialization.

4. **`l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum`** — global
   atomic ops issued. The dq/dk/dv semaphore writes show up here.
   Big delta between det and non-det.

5. **`launch__registers_per_thread`** — if higher in det than non-det,
   that may indicate spilling; the smaller tile changes register pressure.

## nsys (Nsight Systems) — timeline view

`nsys` traces multi-kernel sequences. Useful for:
- Seeing the order of kernels in one bwd call
  (preprocess → mainloop → postprocess (dq) → maybe postprocess (dkv)).
- Spotting unexpected gaps (CPU-side scheduling overhead) between
  kernel launches.
- Comparing wall-clock time across iterations.

Open in Nsight Systems UI:

```bash
scp devpod:~/flash-attention/hopper/profile_out/nsys_det.nsys-rep ~/
nsys-ui ~/nsys_det.nsys-rep
# Or open the .nsys-rep file in the Nsight Systems Mac/Win/Linux GUI.
```

CLI summary:

```bash
nsys stats --report cuda_gpu_kern_sum profile_out/nsys_det.nsys-rep
nsys stats --report cuda_api_sum profile_out/nsys_det.nsys-rep
```

## profile_diff.py — quick textual comparison

```bash
./profile_diff.py profile_out/ncu_nondet.ncu-rep profile_out/ncu_det.ncu-rep
```

Pulls a curated subset of metrics (occupancy, stall reasons, atomic counts,
DRAM bandwidth, grid shape, register/smem usage), groups by kernel, and
prints `nondet | det | ratio`.

## Other useful CUDA profiling tricks

- **`cute::printf` with thread guards** — the FA3 kernels already use this.
  `if (threadIdx.x == 0 && blockIdx.x == 0) cute::printf(...)` in any of
  the kernel files; rebuild; the printf shows up in stdout. See
  `AI/DEBUG_2CTA.md` in the repo root.

- **Inspect emitted PTX** — set `CUTE_DSL_KEEP_PTX=1` at build time and
  the `.ptx` files are saved under the build dir. Useful to verify what
  the compiler did for register allocation.

- **`cuobjdump --dump-sass _C.abi3.so`** — dump compiled SASS for every
  kernel. For one specific kernel:
  `cuobjdump --dump-sass-name "<kernel-name-regex>" _C.abi3.so`.

- **`compute-sanitizer --tool=racecheck`** — find race conditions. Can
  produce false positives with raw TMA — see `AI/RACECHECK_TMA_HAZARD.md`
  in the repo root.

- **`ptxas info` from build logs** — already present in the build output:
  `ptxas info : Used N registers, M bytes spill stores, M bytes spill
  loads`. Gives per-kernel register pressure for free.

- **`smem`/`reg` budget queries**:
  ```python
  import torch
  d = torch.cuda.get_device_properties(0)
  print("max smem per CTA:", d.shared_memory_per_block_optin)   # 227 KB on H100
  print("max registers per CTA:", d.regs_per_block)
  ```

## Caveats

- ncu requires GPU performance counter access. On Lepton pods running as
  root this is fine; if you ever see `ERR_NVGPUCTRPERM`, run with `sudo`
  or set `RmProfilingAdminOnly=0` via the kernel module params.
- `ncu --set full` does **kernel replay** — the kernel runs many times to
  gather different metric sections. For a 1.6 ms bwd that means ~50 s of
  ncu wall time per kernel. Use `--set basic` or specific metrics to
  speed up iteration.
- nsys does NOT replay — it observes the real run. Cheap to capture
  (~1 s overhead).
- nsys traces grow fast on long runs — the harness uses 20 iterations to
  keep the file small.
