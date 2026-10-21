#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LD_LIBRARY_PATH="$PROJECT_DIR/3rdparty/librknn_api/aarch64:$PROJECT_DIR/3rdparty/rga/RK3588/lib/Linux/aarch64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$PROJECT_DIR/build"
echo '[legacy] model=../model/yolov5s.rknn, runtime=project libraries'
exec ./app --model-path ../model/yolov5s.rknn --threads 6 --warmup 50 "$@"
