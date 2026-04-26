#!/usr/bin/env python3
"""Diff two ncu reports — pulls a curated metric set and prints side-by-side.

Used after `profile_det_bwd.sh` to compare det-vs-non-det at the same shape.

Usage:  profile_diff.py NCU_REPORT_A NCU_REPORT_B [--metrics-csv path]
"""

from __future__ import annotations

import csv
import io
import os
import subprocess
import sys


# Curated metrics that tell the story of "where did the time go?"
# Names follow ncu's metric IDs. See `ncu --query-metrics-mode all` for the catalog.
KEY_METRICS = [
    # Wall-clock (per-kernel). When comparing, biggest delta first.
    ("gpu__time_duration.sum", "gpu time", "us"),
    ("smsp__cycles_active.avg.pct_of_peak_sustained_elapsed", "SM busy", "%"),
    # Achieved occupancy — how full the SMs actually are.
    ("sm__warps_active.avg.pct_of_peak_sustained_active", "achieved occupancy", "%"),
    # Where warps stall. Big differences here pinpoint why kernels are slow.
    ("smsp__warp_issue_stalled_barrier_per_warp_active.pct", "stall: barrier", "%"),
    ("smsp__warp_issue_stalled_membar_per_warp_active.pct", "stall: membar", "%"),
    ("smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct", "stall: short scoreboard (RAW)", "%"),
    ("smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct", "stall: long scoreboard (memory)", "%"),
    ("smsp__warp_issue_stalled_wait_per_warp_active.pct", "stall: wait", "%"),
    # Atomic ops — det path uses these heavily for dq_semaphore.
    ("l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum", "global atom sectors", ""),
    # Memory throughput.
    ("dram__bytes_read.sum", "DRAM bytes read", "B"),
    ("dram__bytes_write.sum", "DRAM bytes write", "B"),
    ("l2__throughput.avg.pct_of_peak_sustained_elapsed", "L2 throughput", "%"),
    # CTA dispatch / parallelism shape.
    ("launch__grid_size", "grid (CTAs)", ""),
    ("launch__block_size", "block (threads)", ""),
    ("launch__registers_per_thread", "registers/thread", ""),
    ("launch__shared_mem_per_block_static", "static smem/CTA", "B"),
    ("launch__shared_mem_per_block_dynamic", "dynamic smem/CTA", "B"),
]


def fetch(report_path: str) -> dict[str, dict[str, str]]:
    """Run ncu --import --csv and return a {kernel_name: {metric_id: value}} dict."""
    metric_ids = ",".join(m[0] for m in KEY_METRICS)
    result = subprocess.run(
        [
            "/usr/local/cuda/bin/ncu",
            "--import", report_path,
            "--csv",
            "--page", "raw",
            "--metrics", metric_ids,
        ],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        print(f"ncu import failed for {report_path}:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)
    out: dict[str, dict[str, str]] = {}
    reader = csv.DictReader(io.StringIO(result.stdout))
    for row in reader:
        # Each row corresponds to one kernel launch + one metric.
        kernel = row.get("Kernel Name", "<?>") or "<?>"
        metric = row.get("Metric Name", "")
        value = row.get("Metric Value", "")
        if not metric:
            continue
        out.setdefault(kernel, {})[metric] = value
    return out


def short_kernel_name(full: str) -> str:
    # ncu kernel names are mangled C++ — chop to a readable prefix.
    s = full.split("(")[0]
    s = s.split("<")[0]
    return s[:60]


def fmt(s: str) -> str:
    if not s:
        return "-"
    try:
        v = float(s.replace(",", ""))
    except ValueError:
        return s
    if abs(v) >= 1e9:
        return f"{v / 1e9:.2f}G"
    if abs(v) >= 1e6:
        return f"{v / 1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"{v / 1e3:.2f}K"
    return f"{v:.2f}" if v - int(v) else str(int(v))


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    a_path, b_path = sys.argv[1], sys.argv[2]
    a = fetch(a_path)
    b = fetch(b_path)

    common_kernels = sorted(set(a) & set(b))
    if not common_kernels:
        print("no kernels common to both reports — abort", file=sys.stderr)
        print(f"A had: {list(a)}", file=sys.stderr)
        print(f"B had: {list(b)}", file=sys.stderr)
        sys.exit(1)

    label_a = os.path.splitext(os.path.basename(a_path))[0]
    label_b = os.path.splitext(os.path.basename(b_path))[0]

    for kernel in common_kernels:
        ma = a[kernel]
        mb = b[kernel]
        print(f"\n=== kernel: {short_kernel_name(kernel)} ===")
        print(f"  {'metric':45s}  {label_a:>14s}  {label_b:>14s}  {'ratio':>8s}")
        print(f"  {'-' * 45}  {'-' * 14:>14s}  {'-' * 14:>14s}  {'-' * 8:>8s}")
        for metric_id, friendly, unit in KEY_METRICS:
            va = ma.get(metric_id, "")
            vb = mb.get(metric_id, "")
            label = f"{friendly} ({unit})" if unit else friendly
            ratio = ""
            try:
                fa = float(va.replace(",", ""))
                fb = float(vb.replace(",", ""))
                if fa != 0:
                    ratio = f"{fb / fa:.2f}x"
            except ValueError:
                pass
            print(f"  {label:45s}  {fmt(va):>14s}  {fmt(vb):>14s}  {ratio:>8s}")


if __name__ == "__main__":
    main()
