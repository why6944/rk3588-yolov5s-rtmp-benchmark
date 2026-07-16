#!/usr/bin/env bash
# ============================================================================
# YOLOv5 摄像头 DMA-BUF 管线 CPU 测试脚本
#
# 用法:
#   ./run_camera_bench.sh scalar old     # 标量版 + 旧模型
#   ./run_camera_bench.sh neon   new     # NEON版  + 新模型 (relu)
#   ./run_camera_bench.sh scalar new     # 标量版 + 新模型
#   ./run_camera_bench.sh neon   old     # NEON版  + 旧模型
#
# 依赖:
#   build/app         (标量后处理, 使用 post_process.cpp)
#   build/app_neon    (NEON后处理, 使用 post_process_neon_full.cpp)
#   /tmp/monitor_cpu.py
#
# 输出目录: debug_records/camera_bench/
# ============================================================================
set -euo pipefail

# ---- 参数 ----
MODE="${1:-scalar}"        # scalar | neon
MODEL="${2:-old}"          # old | new
LOOPS="${3:-600}"
THREADS="${4:-6}"
BOX_THRESHOLD="${5:-0.6}"

# ---- 路径 ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
MODEL_DIR="$SCRIPT_DIR/model"
RUN_DIR="$SCRIPT_DIR/debug_records/camera_bench"
MONITOR_PY="/tmp/monitor_cpu.py"

mkdir -p "$RUN_DIR"

# ---- 选择二进制 ----
if [ "$MODE" = "neon" ]; then
    BIN="$BUILD_DIR/app_neon"
else
    BIN="$BUILD_DIR/app"
fi

# ---- 选择模型 ----
if [ "$MODEL" = "new" ]; then
    MODEL_PATH="$MODEL_DIR/yolov5s_relu_INT8_Normal.rknn"
    SKIP_SIGMOID="--skip-sigmoid"
    MODEL_LABEL="new_relu"
else
    MODEL_PATH="$MODEL_DIR/yolov5s.rknn"
    SKIP_SIGMOID=""
    MODEL_LABEL="old_sigmoid"
fi

LABEL="${MODE}_${MODEL_LABEL}"
LOG="$RUN_DIR/${LABEL}.log"
CPU_CSV="$RUN_DIR/${LABEL}_cpu.csv"

echo "============================================"
echo "  Camera DMA-BUF CPU Benchmark"
echo "============================================"
echo "  Mode:        $MODE (bin: $(basename $BIN))"
echo "  Model:       $MODEL ($(basename $MODEL_PATH))"
echo "  Loops:       $LOOPS"
echo "  Threads:     $THREADS"
echo "  Box thresh:  $BOX_THRESHOLD"
echo "  Skip sigmoid: ${SKIP_SIGMOID:-no}"
echo "  Log:         $LOG"
echo "  CPU CSV:     $CPU_CSV"
echo "============================================"
echo ""

# ---- 检查二进制 ----
if [ ! -x "$BIN" ]; then
    echo "ERROR: $BIN not found. Build first: cd build && make app app_neon"
    exit 1
fi

# ---- 检查监控脚本 ----
if [ ! -f "$MONITOR_PY" ]; then
    echo "ERROR: $MONITOR_PY not found."
    echo "Create it with:"
    echo '  cat > /tmp/monitor_cpu.py << PYEOF'
    echo '  ... (see debug_records for the script)'
    echo '  PYEOF'
    exit 1
fi

# ---- 清理函数 ----
cleanup() {
    echo ""
    echo "=== Cleaning up ==="
    if [ -n "${APP_PID:-}" ]; then
        kill -9 "$APP_PID" 2>/dev/null && echo "  Killed app (PID $APP_PID)"
    fi
    if [ -n "${MON_PID:-}" ]; then
        kill -9 "$MON_PID" 2>/dev/null && echo "  Killed monitor (PID $MON_PID)"
    fi
    # 释放摄像头
    fuser -k /dev/video0 2>/dev/null && echo "  Released /dev/video0"
    sleep 1
}
trap cleanup EXIT INT TERM

# ---- 启动监控脚本 (后台) ----
# 需要提前准备好 monitor_cpu.py，此处提供内联创建
if [ ! -f "$MONITOR_PY" ]; then
    echo "=== Creating monitor_cpu.py ==="
    cat > "$MONITOR_PY" << 'PYEOF'
import os, sys, time, csv
pid = int(sys.argv[1])
out = sys.argv[2]
clk = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
def tj():
    with open("/proc/stat") as f: return sum(int(x) for x in f.readline().split()[1:])
def pj(pid_val):
    total = 0
    try: tids = os.listdir("/proc/%d/task" % pid_val)
    except FileNotFoundError: return None
    for tid in tids:
        try:
            with open("/proc/%d/task/%s/stat" % (pid_val, tid)) as f: s = f.read()
            total += int(s.rsplit(")",1)[1].split()[11]) + int(s.rsplit(")",1)[1].split()[12])
        except: continue
    return total
with open(out,"w",newline="") as f:
    w = csv.writer(f); w.writerow(["ts","cpu_total","cpu_onecore","pj","tj"])
    pt, pp, pw = tj(), pj(pid), time.time()
    while pp is not None and os.path.exists("/proc/%d" % pid):
        time.sleep(0.5)
        ct, cp, nw = tj(), pj(pid), time.time()
        if cp is None: break
        dt, dp, wall = ct-pt, cp-pp, nw-pw
        w.writerow(["%.3f"%nw, "%.3f"%(dp/dt*100 if dt>0 else 0), "%.3f"%(dp/clk/wall*100 if wall>0 else 0), dp, dt])
        f.flush()
        pt, pp, pw = ct, cp, nw
PYEOF
    echo "  Created $MONITOR_PY"
fi

# ---- 运行测试 ----
echo "=== Starting $BIN ==="
cd "$BUILD_DIR"
./$(basename "$BIN") \
    --mode mpp-only \
    --input camera \
    --input-backend dmabuf \
    --camera-id 0 \
    --camera-width 640 \
    --camera-height 480 \
    --camera-fps 30 \
    --camera-format YUYV \
    --threads "$THREADS" \
    --loops "$LOOPS" \
    --box-threshold "$BOX_THRESHOLD" \
    --model-path "$MODEL_PATH" \
    $SKIP_SIGMOID \
    > "$LOG" 2>&1 &
APP_PID=$!
echo "  PID=$APP_PID"

sleep 2

# 检查 app 是否还在运行
if ! kill -0 "$APP_PID" 2>/dev/null; then
    echo "ERROR: App exited immediately!"
    tail -10 "$LOG"
    exit 1
fi

# ---- 启动 CPU 监控 ----
echo "=== Starting CPU monitor ==="
python3 "$MONITOR_PY" "$APP_PID" "$CPU_CSV" &
MON_PID=$!
echo "  Monitor PID=$MON_PID"

# ---- 等待完成 ----
echo "=== Waiting (loops=$LOOPS, ~${LOOPS}/30 seconds)... ==="
wait "$APP_PID" 2>/dev/null || true
APP_EXIT=$?
echo "  App exited with code $APP_EXIT"

kill "$MON_PID" 2>/dev/null || true
wait "$MON_PID" 2>/dev/null || true
echo "  Monitor stopped"

# ---- 输出结果 ----
echo ""
echo "============================================"
echo "  Results: $LABEL"
echo "============================================"

# Benchmark 汇总
echo "--- Benchmark Summary ---"
tail -5 "$LOG" | grep -E "Benchmark|PerfMonitor" || echo "  (no benchmark output)"

# CPU 统计
echo ""
echo "--- CPU Statistics ---"
python3 -c "
import csv
try:
    rows = list(csv.DictReader(open('$CPU_CSV')))
    stable = [r for r in rows[2:] if float(r.get('pj',0))>=0]
    if stable:
        oc = sum(float(r['cpu_onecore']) for r in stable)/len(stable)
        tc = sum(float(r['cpu_total']) for r in stable)/len(stable)
        print('  CPU单核: {:.1f}%  CPU整板: {:.1f}%  采样: {}'.format(oc, tc, len(stable)))
    else:
        print('  No valid CPU samples')
except Exception as e:
    print('  Error reading CPU CSV:', e)
"

echo ""
echo "  Log:    $LOG"
echo "  CPU CSV: $CPU_CSV"
echo "============================================"
