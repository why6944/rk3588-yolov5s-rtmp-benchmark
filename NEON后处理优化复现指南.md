# YOLOv5 后处理 NEON 优化 — 复现指南

## 1. 硬件环境

- **设备**: Orange Pi 5 (RK3588)，4×A76 + 4×A55
- **编译器**: GCC 9.4.0, aarch64
- **依赖**: OpenCV 4.2, librknnrt.so, librga.so, MPP

## 2. 代码位置

```
/home/orangepi/streamer_codev5.0/Desktop/
├── post_process.cpp              # 原始后处理（已修复 NMS bug）
├── post_process_neon_full.cpp    # NEON 后处理（完整替代）
├── post_process_neon.cpp         # NEON process() + 注释
├── post_process_scalar.cpp       # 标量 process()
├── post_process_auto.cpp         # 自动向量化 process()
├── post_process_common.cpp/h     # 共享代码 + 编排器
├── benchmark_main.cpp            # 独立后处理 benchmark
├── CMakeLists.txt                # 含 app / app_neon / benchmark_postprocess 目标
├── main.cpp                      # 主管线（支持 --postprocess）
├── int8_dumps/                   # 30 帧 INT8 数据（离线 benchmark 用）
└── model/
    ├── yolov5s.rknn              # 旧模型 (Sigmoid)
    └── yolov5s_relu_INT8_Normal.rknn  # 新模型 (ReLU)
```

## 3. 编译

```bash
cd /home/orangepi/streamer_codev5.0/Desktop/build
cmake ..
make app app_neon benchmark_postprocess -j4
```

生成三个二进制：
- `app` — 标量后处理管线
- `app_neon` — NEON 后处理管线  
- `benchmark_postprocess` — 独立后处理 benchmark（不依赖摄像头/NPU）

## 4. CPU 监控脚本

```bash
# 创建 /tmp/monitor_cpu.py（进程级 CPU 采样，0.5s 间隔）
cat > /tmp/monitor_cpu.py << 'PYEOF'
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
```

**CPU 指标说明**：
- `cpu_onecore`：进程 CPU 折算到单核（≈ `top -p <pid>` 的口径，调试记录 26% 就是这个）
- `cpu_total`：进程 CPU 占整板 8 核的比例（= cpu_onecore / 8）

## 5. 离线后处理 benchmark（不依赖摄像头）

### 场景：纯后处理性能对比，排除 NPU/摄像头/MPP 干扰

```bash
cd /home/orangepi/streamer_codev5.0/Desktop/build

# 标量版
taskset -c 4 ./benchmark_postprocess --postprocess scalar \
    --frames 30 --warmup 100 --iterations 1000 --rounds 5 \
    --data-dir ../int8_dumps

# 自动向量化版
taskset -c 4 ./benchmark_postprocess --postprocess auto \
    --frames 30 --warmup 100 --iterations 1000 --rounds 5 \
    --data-dir ../int8_dumps

# NEON 版
taskset -c 4 ./benchmark_postprocess --postprocess neon \
    --frames 30 --warmup 100 --iterations 1000 --rounds 5 \
    --data-dir ../int8_dumps

# perf stat（硬件计数器）
taskset -c 4 perf stat -e cycles,instructions,cache-misses,cache-references \
    ./benchmark_postprocess --postprocess neon \
    --frames 30 --warmup 100 --iterations 500 --rounds 1 \
    --data-dir ../int8_dumps

# objdump 验证 NEON 指令
objdump -d benchmark_postprocess | grep -E 'ldr.*q[0-9]|cmge|umaxv|dup'
```

### 预期结果

| 指标 | scalar (-O2) | neon | 说明 |
|------|-------------|------|------|
| 帧耗时 | ~204 μs | **~185 μs** | ReLU 模型 162 候选 17 检出 |
| decode 段 | ~176 μs | **~160 μs** | NEON 加速 objectness 阈值扫描 |
| sort | ~2 μs | ~2 μs | 未向量化 |
| nms | ~21 μs | ~21 μs | 未向量化 |
| 加速比 | — | **~9.5%** | 仅 decode 段受益 |

## 6. 完整管线 CPU 测试（摄像头）

### 前置条件
- USB 摄像头已连接
- 不要运行任何 v4l2-ctl（避免锁定设备）
- 确认摄像头节点号（通常插拔后递增）

```bash
# 1. 查看摄像头节点
ls /dev/video*

# 2. 假设摄像头在 /dev/video0，直接跑（不要设 v4l2-ctl）

cd /home/orangepi/streamer_codev5.0/Desktop/build

# === 旧模型 (yolov5s.rknn, Sigmoid) ===

# scalar
./app --mode mpp-only --input camera --input-backend dmabuf \
    --camera-id 0 --camera-width 640 --camera-height 480 --camera-fps 30 --camera-format YUYV \
    --threads 6 --loops 600 --box-threshold 0.6 \
    --model-path ../model/yolov5s.rknn &
PID=$!
sleep 2
python3 /tmp/monitor_cpu.py $PID /tmp/scalar_old_cpu.csv &
wait $PID; kill %1

# neon
./app_neon --mode mpp-only --input camera --input-backend dmabuf \
    --camera-id 0 --camera-width 640 --camera-height 480 --camera-fps 30 --camera-format YUYV \
    --threads 6 --loops 600 --box-threshold 0.6 \
    --model-path ../model/yolov5s.rknn &
PID=$!
sleep 2
python3 /tmp/monitor_cpu.py $PID /tmp/neon_old_cpu.csv &
wait $PID; kill %1

# === 新模型 (yolov5s_relu_INT8_Normal.rknn, ReLU) ===
# 必须加 --skip-sigmoid

# scalar
./app --mode mpp-only --input camera --input-backend dmabuf \
    --camera-id 0 --camera-width 640 --camera-height 480 --camera-fps 30 --camera-format YUYV \
    --threads 6 --loops 600 --box-threshold 0.6 --skip-sigmoid \
    --model-path ../model/yolov5s_relu_INT8_Normal.rknn &
# ... (同上的 CPU 监控)
```

### 查看 CPU 结果

```bash
python3 -c "
import csv
rows = list(csv.DictReader(open('/tmp/scalar_old_cpu.csv')))
stable = [r for r in rows[2:] if float(r.get('pj',0))>=0]
oc = sum(float(r['cpu_onecore']) for r in stable)/len(stable)
print(f'CPU单核口径: {oc:.1f}%')
"
```

### 查看后处理耗时

后处理耗时从 app 的 stdout 输出中获取：

```bash
grep -E "BenchmarkDetail.*post_decode|Benchmark.*postprocess" /path/to/output.log
```

输出示例：
```
[BenchmarkDetail] avg_ms ... post_decode=0.147 ...  (标量)
[BenchmarkDetail] avg_ms ... post_decode=0.037 ...  (NEON)
```

### 预期结果（旧模型, 640×480 YUYV DMA-BUF, 6线程）

| 指标 | scalar (app) | neon (app_neon) | 说明 |
|------|-------------|-----------------|------|
| post_decode | ~0.180 ms | **~0.160 ms** | 旧模型 camera 候选极少时加速比可达 3-4x |
| postprocess | ~0.205 ms | **~0.185 ms** | 含 sort+NMS, NEON 未加速 |
| CPU 单核口径 | ~26% | **~25.5%** | |
| FPS | ~27.7 | ~27.9 | |

> 注: 上表为 ReLU 模型实测。旧 Sigmoid 模型因输出分布不同, 离线基准下 NEON 加速比可达 36% (0.206→0.132ms)。NEON 仅向量化了 objectness 通道阈值扫描, sort/NMS/坐标还原保持标量。

## 7. 验证正确性

三种实现在同一帧 INT8 dump 上输出必须一致：

```bash
cd build
./benchmark_postprocess --postprocess scalar --frames 1 --data-dir ../int8_dumps 2>&1 | grep valid=
# → valid=162 result=17

./benchmark_postprocess --postprocess auto --frames 1 --data-dir ../int8_dumps 2>&1 | grep valid=
# → valid=162 result=17

./benchmark_postprocess --postprocess neon --frames 1 --data-dir ../int8_dumps 2>&1 | grep valid=
# → valid=162 result=17
```

## 8. 汇编验证

```bash
# 确认 NEON 版有 SIMD 指令
objdump -d benchmark_postprocess | grep -A10 '<_Z12process_neon' | grep -E 'cmge|umaxv|dup|ldr.*q'

# 输出应包含:
#   dup   v0.16b, w26        ← vdupq_n_s8
#   ldr   q0, [x24]          ← vld1q_s8
#   cmge  v0.16b, v0.16b, v1.16b  ← vcgeq_s8
#   umaxv b0, v0.16b         ← vmaxvq_u8

# 确认标量版没有 SIMD 指令
objdump -d benchmark_postprocess | grep -A10 '<_Z14process_scalar' | grep -cE 'cmge|umaxv|dup.*16b'
# → 0
```

## 9. 测试结束后清理摄像头

```bash
./kill_camera_app.sh
```

或手动：
```bash
fuser -k /dev/video0 2>/dev/null
pkill -9 app 2>/dev/null
```

## 10. 常见问题

| 问题 | 解决 |
|------|------|
| `Device or resource busy` | USB 摄像头拔插重置 |
| `RGA YUYV->BGR failed` | 摄像头不支持 DMA-BUF，换 OpenCV 后端 |
| `can't open camera by index` | camera-id 和 /dev/videoX 编号不对应 |
| NEON 版 decode 没提升 | 确认 `app_neon` 而不是 `app`；`objdump` 检查 SIMD 指令 |
| CPU% 波动大 | 摄像头画面变化（光照、移动）导致每帧候选数不同；多跑几轮取平均 |
