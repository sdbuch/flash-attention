#!/usr/bin/env bash
#
# Profile the deterministic FA3 backward at head_dim=256 with ncu and nsys.
# Generates side-by-side reports for det vs non-det at the same shape so
# you can see where the extra wall time goes.
#
# Run on devpod:  bash profile_det_bwd.sh
# Outputs land in ./profile_out/
#
# Tunables (env vars):
#   PYTHON     — interpreter, default ~/venv-fa3-dev/bin/python
#   SHAPE_ARGS — args passed to profile_det_bwd.py for shape selection
#                (default: hdim=256, seqlen=4096, GQA 6/2, causal SWA=1024)

set -euo pipefail

PYTHON="${PYTHON:-$HOME/venv-fa3-dev/bin/python}"
NCU="/usr/local/cuda/bin/ncu"
NSYS="/opt/nvidia/nsight-compute/2025.3.0/host/target-linux-x64/nsys"
HOPPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${HOPPER_DIR}/profile_out"
mkdir -p "$OUT"

# Shape args — defaults match the SWA case where det blows up to 4.6x non-det.
# Override by setting SHAPE_ARGS in the environment.
SHAPE_ARGS="${SHAPE_ARGS:---hdim 256 --batch 1 --seqlen 4096 --heads 6 --kv-heads 2 --window 1024}"

cd "$HOPPER_DIR"
export PYTHONPATH="$HOPPER_DIR"
export CUDA_VISIBLE_DEVICES=0

WARMUP=5

run_ncu() {
    local label="$1" det_flag="$2"
    local report="$OUT/ncu_${label}.ncu-rep"
    echo "=== ncu ${label}: ${SHAPE_ARGS} ${det_flag} ==="
    # Skip warmup launches, profile exactly the one timed launch.
    # --target-processes all so we catch any subprocess.
    # --set full pulls a comprehensive metric set (slow ~30s but rich).
    # The mainloop kernel name we care about is run_mha_bwd_*Sm90*.
    "$NCU" \
        --target-processes all \
        --launch-skip "$WARMUP" \
        --launch-count 100 \
        --set full \
        --kernel-name regex:flash \
        --force-overwrite \
        --export "${report%.ncu-rep}" \
        "$PYTHON" profile_det_bwd.py \
            $SHAPE_ARGS $det_flag \
            --warmup "$WARMUP" \
            --mode warmup_then_one
    echo "wrote $report"
}

run_nsys() {
    local label="$1" det_flag="$2"
    local report="$OUT/nsys_${label}.nsys-rep"
    echo "=== nsys ${label}: ${SHAPE_ARGS} ${det_flag} ==="
    "$NSYS" profile \
        --trace=cuda,nvtx \
        --output "${report%.nsys-rep}" \
        --force-overwrite=true \
        --capture-range cudaProfilerApi --capture-range-end stop \
        "$PYTHON" profile_det_bwd.py \
            $SHAPE_ARGS $det_flag \
            --warmup 5 --iters 20 --mode loop \
        2>&1 | tail -3 || true
    echo "wrote $report"
}

echo "## ncu deep profiles (single kernel, full metric set)"
run_ncu "nondet" ""
run_ncu "det"    "--deterministic"

echo
echo "## nsys timelines (20-iter loop)"
run_nsys "nondet" ""
run_nsys "det"    "--deterministic"

echo
echo "## key metric diff (ncu)"
"$HOPPER_DIR/profile_diff.py" "$OUT/ncu_nondet.ncu-rep" "$OUT/ncu_det.ncu-rep"

echo
echo "Open the nsys timelines in Nsight Systems UI (locally):"
echo "  scp devpod:$OUT/nsys_*.nsys-rep ~/nsys-traces/"
echo "  open ~/nsys-traces/nsys_det.nsys-rep   # or use nsys-ui"
