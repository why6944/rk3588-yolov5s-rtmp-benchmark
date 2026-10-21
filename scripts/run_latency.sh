#!/usr/bin/env bash
# 固定相机 DMA-BUF / RKNN fd / MPP copy，逐轮保存日志并检查完整性。
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: bash scripts/run_latency.sh [runs=3] [frames=1000] [warmup=100]"
    echo "Optional: CAMERA_ID=0 LATENCY_TIMEOUT=180s"
    exit 0
fi
if (( $# > 3 )); then
    echo "Too many arguments; use --help." >&2
    exit 2
fi
RUNS="${1:-3}"
FRAMES="${2:-1000}"
WARMUP="${3:-100}"
for value in "$RUNS" "$FRAMES" "$WARMUP"; do
    if [[ ! "$value" =~ ^(0|[1-9][0-9]*)$ ]]; then
        echo "runs, frames and warmup must be decimal integers." >&2
        exit 2
    fi
done
if (( RUNS < 1 || FRAMES < 1 || WARMUP >= FRAMES )); then
    echo "Require runs >= 1 and frames > warmup >= 0." >&2
    exit 2
fi
command -v python3 >/dev/null
command -v timeout >/dev/null
test -x "$PROJECT_DIR/build/app"
test -r "$PROJECT_DIR/model/yolov5s_relu_INT8_Normal.rknn"
export LD_LIBRARY_PATH="$PROJECT_DIR/3rdparty/librknn_api/aarch64:$PROJECT_DIR/3rdparty/rga/RK3588/lib/Linux/aarch64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$PROJECT_DIR/debug_records"
RESULT_DIR="$(mktemp -d "$PROJECT_DIR/debug_records/latency_XXXXXX")"
echo "LATENCY_RESULT_DIR=$RESULT_DIR"
cd "$PROJECT_DIR/build"
for (( run=1; run<=RUNS; run++ )); do
    if ! env V4L2_BUFFER_COUNT=8 timeout "${LATENCY_TIMEOUT:-180s}" ./app \
        --mode mpp-only --input camera --input-backend dmabuf \
        --camera-id "${CAMERA_ID:-0}" --camera-width 640 --camera-height 480 \
        --camera-fps 30 --camera-format YUYV --threads 6 --loops "$FRAMES" \
        --rknn-input-mode fd --rknn-output-mode copy --mpp-input-mode copy \
        --preprocess-backend rga --postprocess neon \
        --box-threshold 0.5 --nms-threshold 0.5 \
        --model-path ../model/yolov5s_relu_INT8_Normal.rknn --skip-sigmoid \
        --latency-warmup "$WARMUP" --latency-csv "$RESULT_DIR/run${run}.csv" \
        --latency-summary > "$RESULT_DIR/run${run}.log" 2>&1; then
        tail -n 40 "$RESULT_DIR/run${run}.log" >&2
        echo "Run $run failed; stopping." >&2
        exit 1
    fi
    if grep -Eq 'select timeout/error|MPP process frame failed|RGA .*failed|duplicate DQBUF' "$RESULT_DIR/run${run}.log"; then
        echo "Pipeline error detected in run $run; see log." >&2
        exit 1
    fi
    python3 "$PROJECT_DIR/tools/summarize_latency.py" \
        --expected "$FRAMES" --warmup "$WARMUP" "$RESULT_DIR/run${run}.csv" \
        > "$RESULT_DIR/run${run}_summary.json"
    echo "Run $run passed: $((FRAMES - WARMUP)) post-warmup samples."
done
echo "Results: $RESULT_DIR"
echo "Check kernel logs separately for any new RGA Oops during this interval."
