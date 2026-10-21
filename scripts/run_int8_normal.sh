#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LD_LIBRARY_PATH="$PROJECT_DIR/3rdparty/librknn_api/aarch64:$PROJECT_DIR/3rdparty/rga/RK3588/lib/Linux/aarch64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$PROJECT_DIR/build"
echo '[int8_normal] model=../model/yolov5s_relu_INT8_Normal.rknn, runtime=2.3.2, skip-sigmoid'
exec ./app --model-path ../model/yolov5s_relu_INT8_Normal.rknn --threads 6 --warmup 50 --skip-sigmoid "$@"
