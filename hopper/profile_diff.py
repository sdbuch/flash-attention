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
# Names follow ncu's metric IDs. To explore the catalog:
#   ncu --query-metrics
#   ncu --query-metrics-mode suffix --metrics smsp__warp_issue_stalled
KEY_METRICS = [
    # Wall-clock (per-kernel). When comparing, biggest delta first.
    ("gpu__time_duration.sum", "gpu time", "ms"),
    ("smsp__cycles_active.avg.pct_of_peak_sustained_elapsed", "SM busy", "%"),
    # Achieved occupancy — how full the SMs actually are.
    ("sm__warps_active.avg.pct_of_peak_sustained_active", "achieved occupancy", "%"),
    # Where warps stall. The "_pct" suffix gives a fraction directly.
    ("smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
        "stall: barrier", ""),
    ("smsp__average_warps_issue_stalled_membar_per_issue_active.ratio",
        "stall: membar", ""),
    ("smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
        "stall: short scoreboard", ""),
    ("smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
        "stall: long scoreboard (mem)", ""),
    ("smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
        "stall: wait", ""),
    ("smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
        "stall: lg throttle", ""),
    # Atomic ops — det path uses these heavily for dq_semaphore.
    ("l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum",
        "global atom requests", ""),
    ("l1tex__t_requests_pipe_lsu_mem_global_op_red.sum",
        "global red requests", ""),
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


# ncu auto-scales units per metric across each report (us / ms / GB / MB / ...).
# Rather than fight it, we normalize every metric to a canonical unit at parse time.
UNIT_TO_CANONICAL: dict[str, tuple[str, float]] = {
    "ns": ("us", 1e-3),
    "us": ("us", 1.0),
    "ms": ("us", 1e3),
    "s":  ("us", 1e6),
    "B":  ("KB", 1 / 1024),
    "KB": ("KB", 1.0),
    "MB": ("KB", 1024.0),
    "GB": ("KB", 1024 * 1024.0),
    "TB": ("KB", 1024 * 1024 * 1024.0),
    "byte": ("KB", 1 / 1024),
    "Kbyte": ("KB", 1.0),
    "Mbyte": ("KB", 1024.0),
    "Gbyte": ("KB", 1024 * 1024.0),
}


def fetch(report_path: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Run ncu --import --csv and return rows + per-metric unit map.

    ncu CSV: header row 0 is column names, header row 1 is units, rest are data.
    We dispense with csv.DictReader since the units row breaks its expectations.
    """
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
    raw_rows = list(csv.reader(io.StringIO(result.stdout)))
    if len(raw_rows) < 2:
        return [], {}
    headers = raw_rows[0]
    units = raw_rows[1]
    unit_map = {h: u for h, u in zip(headers, units)}
    rows: list[dict[str, str]] = []
    for r in raw_rows[2:]:
        if not r or not any(r):
            continue
        d = {h: v for h, v in zip(headers, r)}
        if d.get("Kernel Name", "").strip():
            rows.append(d)
    return rows, unit_map


def short_kernel_label(full: str) -> str:
    """Extract a short, human-readable label for a mangled FA3 kernel name."""
    # cutlass::device_kernel<flash::XXX<...>>(...)  →  XXX
    if "flash::" in full:
        i = full.index("flash::") + len("flash::")
        rest = full[i:]
        # take the C++ identifier after flash::
        j = 0
        while j < len(rest) and (rest[j].isalnum() or rest[j] == "_"):
            j += 1
        name = rest[:j]
        # Some kernels (FlashAttnBwdSm90) are wrapped in enable_sm90<...>;
        # peek at the next flash:: to get the real kernel.
        if name == "enable_sm90" and "flash::" in rest:
            return short_kernel_label(rest)
        return name
    # at::vectorized_elementwise_kernel etc — shorten
    s = full.split("(")[0]
    s = s.split("<")[0]
    return s.split("::")[-1][:40]


def aggregate(
    rows: list[dict[str, str]], unit_map: dict[str, str]
) -> dict[str, dict[str, list[float]]]:
    """Group rows by short kernel label, collect metric values normalized to canonical units."""
    out: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        label = short_kernel_label(row["Kernel Name"])
        bucket = out.setdefault(label, {})
        for metric_id, _, _ in KEY_METRICS:
            v = row.get(metric_id, "")
            try:
                fv = float(v.replace(",", "")) if v else None
            except ValueError:
                fv = None
            if fv is None:
                continue
            unit = unit_map.get(metric_id, "")
            if unit in UNIT_TO_CANONICAL:
                _, scale = UNIT_TO_CANONICAL[unit]
                fv = fv * scale
            bucket.setdefault(metric_id, []).append(fv)
    return out


def canonical_unit(metric_id: str, raw_unit_label: str) -> str:
    """The unit our normalized values are in, given the raw unit ncu emitted."""
    if raw_unit_label in UNIT_TO_CANONICAL:
        return UNIT_TO_CANONICAL[raw_unit_label][0]
    return raw_unit_label


def fmt(v: float | None) -> str:
    if v is None:
        return "-"
    if abs(v) >= 1e9:
        return f"{v / 1e9:.2f}G"
    if abs(v) >= 1e6:
        return f"{v / 1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"{v / 1e3:.2f}K"
    if v == int(v):
        return str(int(v))
    return f"{v:.2f}"


# Kernels relevant to the deterministic-vs-non-deterministic story.
# Other kernels (torch fill, etc.) get filtered out of the per-kernel report.
INTERESTING_KERNELS = {
    "FlashAttnBwdSm90",            # the mainloop — the one that matters most
    "FlashAttnBwdPreprocess",
    "FlashAttnBwdPostprocessConvertdQ",
}


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    a_path, b_path = sys.argv[1], sys.argv[2]
    a_rows, a_units = fetch(a_path)
    b_rows, b_units = fetch(b_path)
    a = aggregate(a_rows, a_units)
    b = aggregate(b_rows, b_units)
    # Use whichever side has units for the report header (they should match
    # post-normalization but the source unit can differ between reports).
    units = a_units if a_units else b_units

    common = sorted(
        k for k in (set(a) & set(b)) if k in INTERESTING_KERNELS
    )
    if not common:
        print(
            f"no FA3 kernels common to both reports.\n"
            f"  A had: {sorted(a)}\n  B had: {sorted(b)}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    label_a = os.path.splitext(os.path.basename(a_path))[0]
    label_b = os.path.splitext(os.path.basename(b_path))[0]

    print(
        f"diff: {label_a}  vs  {label_b}\n"
        f"  Each kernel may appear in multiple invocations; we report the\n"
        f"  median (across launches) of each metric. Kernels not common to\n"
        f"  both reports are skipped.\n"
    )

    for kernel in common:
        ma = a[kernel]
        mb = b[kernel]
        print(f"=== {kernel} ===")
        print(
            f"  {'metric':45s}  {label_a:>14s}  {label_b:>14s}  {'B/A':>8s}"
        )
        print(
            f"  {'-' * 45}  {'-' * 14}  {'-' * 14}  {'-' * 8}"
        )

        def median(xs: list[float] | None) -> float | None:
            if not xs:
                return None
            xs = sorted(xs)
            mid = len(xs) // 2
            if len(xs) % 2 == 1:
                return xs[mid]
            return (xs[mid - 1] + xs[mid]) / 2

        for metric_id, friendly, unit_hint in KEY_METRICS:
            va = median(ma.get(metric_id))
            vb = median(mb.get(metric_id))
            raw_unit = units.get(metric_id, "") or unit_hint
            display_unit = canonical_unit(metric_id, raw_unit) if raw_unit else ""
            label = f"{friendly} ({display_unit})" if display_unit else friendly
            ratio = (
                f"{vb / va:.2f}x"
                if va is not None and vb is not None and va != 0
                else ""
            )
            print(
                f"  {label:45s}  {fmt(va):>14s}  {fmt(vb):>14s}  {ratio:>8s}"
            )
        print()


if __name__ == "__main__":
    main()
