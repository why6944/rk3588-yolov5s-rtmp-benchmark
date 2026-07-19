# RK3588 YOLOv5s 项目 perf 性能分析教学调试文档

> 适用项目：`/home/orangepi/streamer_codev5.0/Desktop`  
> 适用板卡：Orange Pi 5 Pro / RK3588  
> 文档目标：建立可复现的 CPU 性能分析流程，定位预处理、后处理、线程调度、内存访问和驱动调用中的热点。

## 1. 先明确 perf 能分析什么

`perf` 是 Linux CPU 与内核性能分析工具，适合回答：

- app 在 CPU 上消耗了多少时间；
- 哪些函数占用了最多的 CPU 采样；
- 指令数、CPU 周期、IPC、分支预测和 cache miss 情况；
- 多线程程序是否存在频繁上下文切换、迁核或锁竞争；
- 用户态、内核态和驱动调用分别消耗了多少 CPU 时间。

`perf` 不能直接测量：

- RKNN 三核 NPU 的真实硬件利用率；
- RGA 内部像素处理单元的忙碌比例；
- MPP/VPU 编码器内部利用率；
- RTMP 服务端或网络链路的真实处理能力。

这些硬件的工作通常由 CPU 发起，再由 CPU 等待驱动完成。因此 `perf` 可能只看到 `ioctl()`、驱动函数或等待过程，不能把 `rknn_run` 的墙钟时间直接解释为 CPU 计算时间。NPU 利用率仍应结合项目的 `PerfMonitor` 和 `/sys/kernel/debug/rknpu/load`。

## 2. 当前板端环境快照

2026-07-11 实际检查结果：

| 项目 | 当前结果 | 解释 |
|---|---|---|
| 主机 | `orangepi5pro` | Orange Pi 5 Pro |
| 架构 | `aarch64` | ARM 64 位 |
| 内核 | `5.10.160-rockchip-rk3588` | Rockchip 定制内核 |
| CPU | 8 核 | CPU0-3 小核，CPU4-7 大核 |
| CPU0-3 | capacity=397，最高 1.8GHz | Cortex-A55 集群 |
| CPU4-7 | capacity=1024，最高 2.4GHz | Cortex-A76 集群 |
| PMU | `armv8_pmuv3` | ARMv8 硬件性能计数器存在 |
| 内核配置 | `CONFIG_PERF_EVENTS=y` | 支持 perf events |
| 内核配置 | `CONFIG_HW_PERF_EVENTS=y` | 支持硬件计数器 |
| 内核配置 | `CONFIG_ARM_PMU=y` | 支持 ARM PMU |
| 内核配置 | `CONFIG_FRAME_POINTER=y` | 内核栈回溯条件较好 |
| `perf_event_paranoid` | `2` | 普通用户权限较受限 |
| `kptr_restrict` | `1` | 普通用户不能完整查看内核符号 |
| perf 工具 | 已安装 `perf 5.4.291` | `/usr/local/bin/perf` |
| 当前 app | 未 strip，有 `.symtab` | 能看到部分函数名 |
| 当前 app 调试信息 | 无 `.debug_info/.debug_line` | 不能可靠映射到源码行 |
| 当前 CMake 构建类型 | 空 | 不是标准 RelWithDebInfo 构建 |

结论：内核和硬件支持 perf，Ubuntu 版 perf 已完成安装并通过基础 ARM PMU 验证；下一步需要建立一个保留优化和调试信息的独立构建目录。

## 3. 安装 perf

### 3.1 快速方案：安装 Ubuntu 提供的 linux-tools

```bash
sudo apt update
sudo apt install linux-tools-common linux-tools-generic

command -v perf
perf --version
```

当前系统的软件源提供的 `linux-tools-generic` 是 Ubuntu 5.4 系列，而运行内核是 Rockchip 5.10.160。不同版本的 perf 对通用事件通常仍可能可用，但以下功能可能不完全匹配：

- Rockchip 私有 tracepoint；
- 新旧内核字段；
- 特定 PMU 事件名称；
- 内核调用栈解析。

安装后必须先验证：

```bash
sudo perf list
sudo perf stat -e task-clock,cycles,instructions -- sleep 1
```

如果能正常输出计数，可先用于本项目的 CPU 热点分析。

### 3.1.1 本板实际安装结果（2026-07-11）

已执行：

```bash
sudo apt-get update
sudo apt-get install -y linux-tools-common linux-tools-generic
```

实际安装版本：

```text
linux-tools-5.4.0-216
perf version 5.4.291
```

由于运行内核名为 `5.10.160-rockchip-rk3588`，Ubuntu 的 `/usr/bin/perf` 包装器会尝试寻找同名内核工具并报：

```text
WARNING: perf not found for kernel 5.10.160-rockchip
```

实际 perf 二进制可以工作，因此建立了优先级更高的链接：

```bash
sudo ln -s /usr/lib/linux-tools/5.4.0-216-generic/perf /usr/local/bin/perf
```

现在直接执行：

```bash
command -v perf
perf --version
```

结果为：

```text
/usr/local/bin/perf
perf version 5.4.291
```

基础验证成功：

```bash
sudo perf stat \
  -e task-clock,cycles,instructions,cache-references,cache-misses \
  -- sleep 1
```

当前 RK3588 PMU 支持上述事件。通用别名 `branches` / `branch-instructions` 返回 `<not supported>`，而 `branch-misses` 可以计数。因此当前版本不能用 `branch-misses / branch-instructions` 计算可靠的分支失败率，只能把 `branch-misses` 作为绝对值并按帧归一化比较。

### 3.2 严谨方案：从匹配内核源码构建 tools/perf

如果 Ubuntu 提供的 perf 报版本不匹配，建议获取与当前内核一致的 Orange Pi / Rockchip 5.10.160 源码，再构建：

```bash
sudo apt install build-essential flex bison \
  libelf-dev libdw-dev libunwind-dev libssl-dev \
  python3-dev binutils-dev libnuma-dev

cd <匹配当前内核的源码目录>
make -C tools/perf -j$(nproc)
./tools/perf/perf --version
```

板子当前没有完整 Linux 内核源码，所以这一方案需要先准备匹配的源码。不要随便拿一个不同厂商或不同分支的 5.10 源码替代。

## 4. 权限策略

当前：

```text
kernel.perf_event_paranoid = 2
kernel.kptr_restrict = 1
```

教学阶段建议直接使用：

```bash
sudo perf ...
```

这样不会永久修改系统安全参数。

如果后续确实需要普通用户采样，可临时调整：

```bash
sudo sysctl kernel.perf_event_paranoid=1
sudo sysctl kernel.kptr_restrict=0
```

调试结束后恢复：

```bash
sudo sysctl kernel.perf_event_paranoid=2
sudo sysctl kernel.kptr_restrict=1
```

不要为了省一条 `sudo`，长期把生产设备的 perf 权限完全放开。

## 5. 建立专用 perf 构建

不要覆盖当前可以运行的 `build/`。新建 `build-perf/`：

```bash
cd /home/orangepi/streamer_codev5.0/Desktop

cmake -S . -B build-perf \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_FLAGS_RELWITHDEBINFO="-O2 -g -DNDEBUG -fno-omit-frame-pointer" \
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O2 -g -DNDEBUG -fno-omit-frame-pointer"

cmake --build build-perf -j$(nproc)
```

这些选项的作用：

| 选项 | 作用 |
|---|---|
| `-O2` | 保留接近真实运行的编译优化 |
| `-g` | 生成函数、源码文件和行号信息 |
| `-fno-omit-frame-pointer` | 提高调用栈回溯可靠性 |
| `RelWithDebInfo` | 不用完全无优化的 Debug 版本冒充生产性能 |

验证二进制：

```bash
file build-perf/app
readelf -S build-perf/app | grep -E 'debug_info|debug_line|symtab'
```

期望至少看到：

```text
.debug_info
.debug_line
.symtab
```

## 6. 固定实验条件

perf 对频率、温度、输入源和线程数非常敏感。每次比较只改变一个变量，并记录：

- Git commit 或代码状态；
- 输入文件/摄像头参数；
- 模式：`infer-only`、`mpp-only` 或 `rtmp`；
- 线程数；
- BOX/NMS 阈值；
- RKNN input/output 模式；
- MPP input 模式；
- CPU governor、温度和后台负载；
- 测试帧数和重复次数。

记录环境：

```bash
uname -a
cat /proc/sys/kernel/perf_event_paranoid
cat /proc/sys/kernel/kptr_restrict
cat /sys/class/thermal/thermal_zone*/temp
ps -eo pid,comm,%cpu --sort=-%cpu | head
```

可选地固定 CPU governor，但必须记录并在测试后恢复：

```bash
grep . /sys/devices/system/cpu/cpufreq/policy*/scaling_governor
```

绑核只用于专门实验。当前板子 CPU0-3 是小核，CPU4-7 是大核。例如：

```bash
taskset -c 4-7 <命令>
```

不要把“绑大核后的结果”和“未绑核结果”放在同一张表里而不注明。

## 7. 统一保存 perf 结果

建议目录：

```bash
cd /home/orangepi/streamer_codev5.0/Desktop
RUN=debug_records/perf/video_mpp_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"

uname -a > "$RUN/environment.txt"
perf --version >> "$RUN/environment.txt"
git status --short > "$RUN/git_status.txt"
```

每次实验保存：

```text
environment.txt
git_status.txt
app.log
perf_stat.txt
perf.data
perf_report.txt
perf_script.txt
```

不要把不同模式生成的 `perf.data` 相互覆盖。

## 8. 第一阶段：perf stat 建立总览

### 8.1 先查看板子支持哪些事件

```bash
sudo perf list
```

ARM PMU 可用事件可能与 x86 不同。不要直接复制网上的 Intel 专用事件。

### 8.2 推荐从视频输入开始

视频文件输入可重复性更好，适合比较代码版本。当前视频为 `1920x1080@25fps`，程序会尽可能快地离线处理，不会按 25FPS 实时等待。

```bash
cd /home/orangepi/streamer_codev5.0/Desktop/build-perf

LD_PATH=/home/orangepi/streamer_codev5.0/Desktop/3rdparty/librknn_api/aarch64:/home/orangepi/streamer_codev5.0/Desktop/3rdparty/rga/RK3588/lib/Linux/aarch64

sudo perf stat \
  -e task-clock,context-switches,cpu-migrations,page-faults \
  -e cycles,instructions,cache-references,cache-misses,branch-misses \
  -e cache-references,cache-misses \
  -- env LD_LIBRARY_PATH="$LD_PATH" \
  ./app \
    --mode mpp-only \
    --input video \
    --video-path ../video.mp4 \
    --threads 6 \
    --box-threshold 0.6 \
    --nms-threshold 0.5 \
    --rknn-input-mode fd \
    --rknn-output-mode copy \
    --mpp-input-mode copy
```

如果事件太多，硬件计数器会发生 multiplex。输出中事件后面的运行比例明显低于 100% 时，应拆成两次：

```bash
sudo perf stat -e task-clock,context-switches,cpu-migrations,page-faults -- <app命令>
sudo perf stat -e cycles,instructions,branch-misses -- <app命令>
sudo perf stat -e cache-references,cache-misses -- <app命令>
```

### 8.3 perf stat 指标怎么解释

| 指标 | 含义 | 本项目中可能说明什么 |
|---|---|---|
| `task-clock` | 进程累计占用 CPU 的时间 | 比整机 CPU 更适合比较 app 开销 |
| `cycles` | CPU 周期数 | CPU 实际执行量的基础指标 |
| `instructions` | 退休指令数 | 与 cycles 组合计算 IPC |
| `instructions/cycle` | IPC | 低 IPC 可能是访存、分支或等待，但不能单独定性 |
| `cache-misses` | cache miss 次数 | 图像拷贝、Mat 遍历和后处理可能造成高 miss |
| `branch-misses` | 分支预测失败绝对次数 | 当前 `branch-instructions` 不支持，应按帧比较，不能计算失败率 |
| `context-switches` | 上下文切换 | 线程过多、队列阻塞、驱动等待可能升高 |
| `cpu-migrations` | 线程迁核 | 异构大小核迁移会影响性能稳定性 |
| `page-faults` | 缺页次数 | 首次运行、动态分配和 mmap 可能产生 |

推荐增加两个归一化指标：

```text
CPU task-clock / 处理帧数
cycles / 处理帧数
```

对于 camera 30FPS 链路，只看 CPU 百分比容易被实时等待影响；每帧 CPU 时间更适合判断 DMA-BUF 是否减少了 CPU 工作量。

## 9. 第二阶段：perf record 定位热点函数

### 9.1 启动程序并采样调用栈

```bash
ROOT=/home/orangepi/streamer_codev5.0/Desktop
RUN="$ROOT/debug_records/perf/video_record_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"

LD_PATH=/home/orangepi/streamer_codev5.0/Desktop/3rdparty/librknn_api/aarch64:/home/orangepi/streamer_codev5.0/Desktop/3rdparty/rga/RK3588/lib/Linux/aarch64

# 默认模型为 ../model/yolov5s_relu_INT8_Normal.rknn，必须从构建目录启动 app；
# 程序会对 ReLU 模型自动跳过 sigmoid。
cd "$ROOT/build-perf"

sudo perf record \
  -F 99 \
  --call-graph dwarf \
  -o "$RUN/perf.data" \
  -- env LD_LIBRARY_PATH="$LD_PATH" \
  ./app \
    --mode mpp-only \
    --input video \
    --video-path ../video.mp4 \
    --threads 6 \
    --box-threshold 0.6 \
    --nms-threshold 0.5 \
    --rknn-input-mode fd \
    --rknn-output-mode copy \
    --mpp-input-mode copy
```

成功运行应看到 `[Main] All done.`。如果出现多次 `open model failed!`，说明 app 没有从 `build-perf` 目录启动；这类 perf.data 只记录错误和崩溃路径，不能用于性能分析。

`-F 99` 表示每秒约 99 次采样，通常足够做函数级分析，又不会给嵌入式板卡带来过高采样开销。

如果 DWARF 回溯开销较高，可尝试：

```bash
sudo perf record -F 99 --call-graph fp ...
```

前提是 app 和关键库保留 frame pointer。

### 9.2 生成文本报告

```bash
sudo perf report \
  --stdio \
  --no-children \
  -i "$RUN/perf.data" \
  > "$RUN/perf_report.txt"

sudo perf script \
  -i "$RUN/perf.data" \
  > "$RUN/perf_script.txt"
```

查看前 50 行：

```bash
sed -n '1,50p' "$RUN/perf_report.txt"
```

### 9.3 本项目重点关注的热点

| 热点 | 可能含义 | 下一步验证 |
|---|---|---|
| `memcpy` / `__memcpy_*` | Mat 或 tensor 仍有大块复制 | 对比 copy/fd 模式，计算每帧 cycles |
| OpenCV color/resize | 预处理仍落在 CPU | 检查是否走 RGA 路径 |
| `post_process` / NMS / IoU | CPU 后处理瓶颈 | 对比候选框数、BOX 阈值、NMS 耗时 |
| `std::sort` | 候选框排序开销 | 减少候选框或 topK |
| `futex` / mutex / condition_variable | 队列、线程池或 backpressure | 看线程状态和调度时间线 |
| `ioctl` / 驱动函数 | CPU 正在频繁提交硬件任务 | 结合调用次数和墙钟时间判断 |
| `cv::rectangle` / 文本绘制 | OpenCV CPU 画框开销 | 对比 draw on/off |
| FFmpeg/OpenCV decode | 视频文件解码开销 | 与 camera 输入分开解释 |

注意：`perf record` 是 on-CPU 采样。如果线程睡眠等待 NPU 或等待队列，它可能不会成为 CPU 热点。高墙钟耗时、低 CPU 采样通常意味着“等待”，不是“CPU 算得慢”。

## 10. 第三阶段：查看源码行和汇编

前提：使用 `build-perf/app`，并确认有 `.debug_info/.debug_line`。

```bash
sudo perf annotate \
  -i "$RUN/perf.data" \
  --stdio \
  > "$RUN/perf_annotate.txt"
```

也可以在交互式 `perf report` 里选中函数后按 Enter，再进入 Annotate。

重点观察：

- 循环里的热点指令；
- `memcpy` 调用点；
- NMS 的 `min/max/mul/div`；
- 分支密集的阈值判断；
- 编译器是否自动生成 ARM NEON/SIMD 指令。

不要看到某行占比高就立即优化。先确认：

1. 该函数在完整程序中占比是否足够大；
2. 优化后是否影响检测结果；
3. 是否只是首次初始化或冷 cache；
4. 是否能通过 RGA/NPU/MPP 更合理地卸载，而不是手写 CPU 优化。

## 11. 多线程分析

### 11.1 查看 app 的线程和当前 CPU

先运行一个持续时间较长的 camera 测试，然后另开终端：

```bash
PID=$(pgrep -n app)
ps -L -p "$PID" -o pid,tid,psr,pcpu,stat,comm
top -H -p "$PID"
```

字段：

- `TID`：线程 ID；
- `PSR`：当前运行在哪个 CPU 核；
- `%CPU`：线程 CPU 占用；
- `STAT`：运行、睡眠或不可中断等待状态。

### 11.2 附加到正在运行的进程

```bash
PID=$(pgrep -n app)

sudo perf stat \
  -p "$PID" \
  -e task-clock,context-switches,cpu-migrations,cycles,instructions \
  -- sleep 10

sudo perf record \
  -F 99 \
  --call-graph dwarf \
  -p "$PID" \
  -- sleep 10
```

视频文件只有约 341 帧，程序结束较快，适合用 `perf record -- <app命令>` 启动采样。camera 持续运行更适合 `-p PID` 附加。

### 11.3 调度时间线

当怀疑 SafeQueue、ThreadPool、future/map 保序或写线程 backpressure 时：

```bash
sudo perf sched record -- <app命令>
sudo perf sched timehist
```

重点看：

- worker 是否频繁睡眠和唤醒；
- aggregator 是否长期等待某个按序 future；
- writeThread 是否让上游积压；
- 线程是否频繁在 A55/A76 之间迁移。

## 12. camera 链路怎么测

camera 是实时输入，固定命令示例：

```bash
sudo perf stat \
  -e task-clock,context-switches,cpu-migrations,cycles,instructions \
  -- env LD_LIBRARY_PATH="$LD_PATH" \
  ./build-perf/app \
    --mode mpp-only \
    --input camera \
    --camera-id 0 \
    --camera-width 640 \
    --camera-height 480 \
    --camera-fps 30 \
    --camera-format YUYV \
    --threads 6 \
    --loops 600 \
    --box-threshold 0.6 \
    --nms-threshold 0.5 \
    --input-backend dmabuf \
    --rknn-input-mode fd \
    --rknn-output-mode copy \
    --mpp-input-mode copy
```

camera 约 30FPS 时，程序会真实等待摄像头出帧。因此：

- 端到端 FPS 接近 30 不代表 CPU/NPU 已经满载；
- `task-clock / frames` 比整机 CPU 百分比更有解释力；
- V4L2 DQBUF/QBUF、DMA-BUF 生命周期和输入等待会影响墙钟时间；
- perf 的 CPU 热点不能替代 camera 单帧延迟测量。

## 13. DMA-BUF 对比实验

目标：判断 DMA-BUF 是否减少了 CPU 工作量，而不是只比较整机 CPU 百分比。

### 13.1 baseline

```text
input-backend=opencv
rknn-input-mode=copy
rknn-output-mode=copy
mpp-input-mode=copy
```

### 13.2 Camera DMA-BUF + RKNN input fd

```text
input-backend=dmabuf
rknn-input-mode=fd
rknn-output-mode=copy
mpp-input-mode=copy
```

两组固定：

```text
camera=640x480 YUYV@30
threads=6
loops=600
BOX=0.6
NMS=0.5
mode=mpp-only
```

比较：

- FPS；
- `task-clock / frame`；
- `cycles / frame`；
- `instructions / frame`；
- `cache-misses / frame`；
- `memcpy` 采样占比；
- 项目埋点的 `pre_copy` 和 `rknn_input_set`。

只有这些指标共同下降，才能有力地说明 DMA-BUF 减少了 CPU 拷贝和内存带宽压力。

## 14. perf 与项目现有埋点怎么配合

| 工具 | 回答的问题 |
|---|---|
| BenchmarkStats | 每帧各阶段墙钟耗时是多少 |
| PerfMonitor | 整机 CPU、三核 NPU 随时间怎么变化 |
| `top -p` / 进程 jiffies | app 进程占用了多少 CPU |
| `perf stat` | app 执行了多少 cycles/instructions/cache miss |
| `perf record/report` | CPU 时间花在哪些函数 |
| `perf sched` | 线程何时运行、睡眠、迁核和等待 |
| `strace -c -f` | 系统调用次数和系统调用墙钟时间 |

推荐分析顺序：

```text
BenchmarkStats 判断慢在哪个阶段
        ↓
perf stat 判断是计算、cache、分支还是调度问题
        ↓
perf record/report 定位具体函数
        ↓
perf annotate 定位源码行/汇编
        ↓
修改一个变量并重新验证
```

## 15. 常见错误与处理

### `perf: command not found`

板子当前就是这个状态。先安装 linux-tools 或构建匹配内核的 `tools/perf`。

### `No permission to enable cycles event`

```bash
cat /proc/sys/kernel/perf_event_paranoid
sudo perf stat ...
```

### `unknown tracepoint` 或内核版本不匹配

说明 perf 用户态工具与 Rockchip 内核不匹配。通用硬件事件可能仍可用，私有 tracepoint 不应强行解释；严谨做法是构建匹配内核的 perf。

### report 只有地址，没有函数名

检查：

```bash
file build-perf/app
readelf -S build-perf/app | grep -E 'debug_info|debug_line|symtab'
```

不要 strip；确保记录时和报告时使用的是同一个二进制及动态库。

### 调用栈大量 `[unknown]`

- 使用 `RelWithDebInfo`；
- 加 `-fno-omit-frame-pointer`；
- 尝试 `--call-graph dwarf`；
- 确认动态库是否包含符号；
- 不要在采样后替换 app 和库文件。

### perf 结果每次差异很大

- 固定输入、线程数和阈值；
- 避免后台任务；
- 记录温度和 CPU governor；
- 至少重复 3 次；
- 第一轮作为 warm-up；
- 分开 camera 实时吞吐和 video 离线吞吐。

### cycles/instructions 显示 `<not supported>`

先确认：

```bash
ls /sys/bus/event_source/devices
sudo perf list
```

当前内核已暴露 `armv8_pmuv3`，理论上具备 ARM PMU。若仍不支持，应检查 perf 版本、虚拟化限制、权限和具体事件名称。

## 16. 建议的第一轮实验矩阵

| 序号 | 输入/模式 | 目的 |
|---:|---|---|
| 1 | video + infer-only + 6 threads | 看预处理、RKNN 调度、后处理的 CPU 开销 |
| 2 | video + mpp-only + 6 threads | 加入画框、RGA NV12、MPP 后的 CPU 变化 |
| 3 | camera OpenCV/copy + mpp-only | camera baseline |
| 4 | camera DMA-BUF/RKNN fd + mpp-only | 验证 DMA-BUF 是否减少 CPU 工作量 |
| 5 | camera DMA-BUF + RTMP | 查看网络封装和发送带来的 CPU/调度开销 |
| 6 | 3/6/9/12 threads | 看线程增加后的 IPC、切换、迁核和吞吐变化 |

每组至少运行 3 次，并输出：

```text
FPS
task-clock/frame
cycles/frame
instructions/frame
IPC
cache miss rate
branch miss rate
context switches
CPU 热点前 10 函数
```

## 17. 面试表达模板

> 我先通过项目内的阶段埋点判断瓶颈属于预处理、推理等待、后处理还是输出链路，再使用 perf stat 统计 task-clock、cycles、instructions、cache miss、分支失败和上下文切换，并按帧归一化，避免不同吞吐量导致 CPU 百分比不可比。之后通过 perf record/report 定位 CPU 热点函数，通过 perf annotate 查看源码行和 ARM 指令。对于多线程链路，我再结合 perf sched、线程 TID 和 CPU 迁移情况判断 ThreadPool、保序 future/map 和写线程 backpressure。NPU/RGA/MPP 属于专用硬件，因此 perf 只用于分析 CPU 提交与等待开销，硬件利用率另外结合 RKNN debugfs 和项目 PerfMonitor 判断。

这套回答的重点不是“会运行 perf 命令”，而是能把数据口径、硬件边界、控制变量和优化验证连成完整闭环。

## 18. 下一步执行顺序

1. 安装并验证 perf；
2. 创建 `build-perf`；
3. 先跑 video + mpp-only 的 `perf stat`；
4. 再生成第一份 `perf record/report`；
5. 根据热点决定是否分析 `memcpy`、NMS、线程调度或驱动等待；
6. 把原始数据和解释追加到项目调试记录，而不是只保存截图。
