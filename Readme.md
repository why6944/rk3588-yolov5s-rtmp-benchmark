# RK3588 边缘视频流 AI 分析网关

基于 Orange Pi 5 Pro / RK3588 的边缘侧视频流分析项目。系统支持视频文件或 USB 摄像头输入，使用 RGA 完成预处理、RKNN Runtime 在三核 NPU 上执行 YOLOv5s INT8 推理，并通过瑞芯微 MPP 进行 H.264 硬编码与 RTMP 推流。

## 项目能力

- 视频文件与 V4L2 USB Camera 输入；
- YOLOv5s ONNX -> RKNN INT8 部署，多 context 并发推理；
- RGA 颜色转换与 resize，MPP H.264 编码，RTMP 推流；
- Camera/RGA/RKNN 的 DMA-BUF fd 输入路径；
- 线程池异步推理、按帧号保序输出与阶段性能埋点；
- `perf stat` / `perf record` 性能分析与 CPU 计数器对比。

## 快速开始

默认模型为 `model/yolov5s_relu_INT8_Normal.rknn`。该模型输出已激活，程序会自动使用 `skip_sigmoid=1`；通常不需要再传 `--model-path` 或 `--skip-sigmoid`。切回旧的 `model/yolov5s.rknn` 时，程序自动使用标准 sigmoid，也可通过 `--apply-sigmoid` 显式指定。

在 `Desktop` 目录编译：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4

export LD_LIBRARY_PATH="$PWD/3rdparty/librknn_api/aarch64:$PWD/3rdparty/rga/RK3588/lib/Linux/aarch64:$LD_LIBRARY_PATH"
cd build
```

以 1080p 视频文件运行 MPP 编码链路：

```bash
./app --mode mpp-only --input video --video-path ../video.mp4 \
  --threads 6 --box-threshold 0.6 --nms-threshold 0.5 \
  --rknn-input-mode fd --rknn-output-mode copy --mpp-input-mode copy
```

以 USB Camera 运行 DMA-BUF 输入链路：

```bash
V4L2_BUFFER_COUNT=8 ./app --mode mpp-only --input camera \
  --camera-id 0 --camera-width 640 --camera-height 480 \
  --camera-fps 30 --camera-format YUYV --threads 6 \
  --input-backend dmabuf --rknn-input-mode fd \
  --rknn-output-mode copy --mpp-input-mode copy
```

摄像头重新插拔后节点可能变化，请先执行：

```bash
v4l2-ctl --list-devices
```

然后将正确的编号传给 `--camera-id`。

固定配置的三轮链路延时测试（从工程根目录执行）：

```bash
bash scripts/run_latency.sh
```

```bash
# 每轮 1000 帧，前 100 帧预热；固定 6 线程、8 个 V4L2 buffer、RKNN fd 与 MPP copy
# 脚本会新建 debug_records/latency_*，保存每轮 CSV、日志和完整性校验后的统计
bash scripts/run_latency.sh
```

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [系统设计与架构](docs/设计文档_系统架构.md) | 模块边界、线程模型、数据流与 DMA-BUF 生命周期 |
| [API 与运行参数](docs/API文档_运行参数与模块接口.md) | 命令行、运行模式、输入后端和关键 C++ 接口 |
| [测试报告](docs/测试报告_性能与验证.md) | 性能口径、吞吐、DMA-BUF 对比与限制 |
| [模型替换与复现](docs/模型替换对比报告.md) | 旧模型与 ReLU INT8 模型对比、输出语义适配与复现方式 |

## 性能摘要

### 模型推理 (纯 NPU, rknn-only)

| 指标 | 旧模型 (YOLOv5s, SiLU) | 新模型 (YOLOv5s, ReLU) | 加速比 |
|------|----------------------|----------------------|:---:|
| P50 rknn_run | 45.2 ms | **17.8 ms** | **2.5x** |
| P95 rknn_run | 64.6 ms | **18.0 ms** | **3.6x** |
| P99 rknn_run | 75.6 ms | **18.0 ms** | **4.2x** |
| 延迟波动 (P99-P50) | 30.4 ms | **0.2 ms** | **152x** |
| 1 线程 FPS | 18.5 | **52.6** | **2.8x** |
| 6 线程 FPS | 113 | **159** | **+41%** |
| 模型大小 | 8.4 MB | 7.99 MB | 相近 |

> 测试条件: Orange Pi 5 Pro, RK3588S, NPU driver 0.9.6, librknnrt.so 2.3.2, INT8 Normal 量化, 50 张 COCO 校准集。SiLU 与 ReLU 模型均为 YOLOv5s 架构, 同一 Toolkit 2.3.2 转换。两份模型来源不同, 性能差异主要来自激活函数及模型图结构差异, 各因素贡献仍需进一步量化。

编译图观察到 ConvReLU 融合节点, 结合板端 A/B 结果判断性能提升与 ReLU 算子融合及模型图结构差异相关; 各因素贡献仍需结合 RKNPU 性能分析进一步量化。

### 线程扩展性 (ReLU INT8-Normal, rknn-only)

| 线程数 | FPS | rknn_avg | Core0 | Core1 | Core2 | 说明 |
|:---:|------|----------|:---:|:---:|:---:|------|
| 1 | 52.6 | 19.0ms | — | — | — | 单核基准 |
| 3 | **142.5** | 20.9ms | 80% | 80% | 80% | 三核均衡, 未饱和 |
| 6 | **158.8** | 37.6ms | 88% | 88% | 89% | **生产推荐** |
| 9 | 159.4 | 54.1ms | 91% | 91% | 91% | 边际收益 |
| 12 | 165.9 | 76.7ms | 91% | 92% | 90% | 收益递减 |

> 3 线程时三核 NPU 利用率仅 80%, 存在空闲窗口。6 线程将利用率推至 89%, FPS 提升 11.5%。9/12 线程 NPU 利用率停滞在 91%, FPS 仅增 3-4%, 但额外线程增加内存占用与调度开销。**推荐 6 线程**作为默认配置。

> 核心绑定已修正为 npu_index % 3 均匀轮转; 12 线程使用 rknn_dup_context 共享模型数据, 避免重复加载导致 OOM。



### Camera + MPP 端到端

- V4L2 Camera (640x480 YUYV@30fps) + RGA + RKNN + MPP: **28.1 FPS** (6 线程)
- rknn_run: 18.4ms, preprocess: 1.77ms, MPP: 2.4ms
- DMA-BUF fd 链路 (ReLU 模型): app 进程 CPU 单核等效占用约 **13%** (实测 12-15.5%, 条件: V4L2_BUFFER_COUNT=8, dmabuf 后端, rknn-input-mode fd)

### Camera + MPP 链路延时

测试链路为摄像头采集 → RGA 预处理 → RKNN 检测 → NEON 后处理与画框 → RGA 转 NV12 → MPP H.264 编码完成。下表合并三轮测试数据，每轮处理 1000 帧并排除前 100 帧预热，共 2700 个有效样本。

| 测量范围 | 样本数 | 均值（ms） | P50（ms） | P95（ms） | P99（ms） |
|---|---:|---:|---:|---:|---:|
| DQBUF 成功后 → MPP 完整编码包取得后 | 2700 | 32.38 | 29.71 | 46.64 | 61.43 |
| V4L2 驱动帧时间戳 → MPP 完整编码包取得后 | 2700 | 66.41 | 64.08 | 81.55 | 98.79 |

> 测试条件：640×480 YUYV@30 FPS、8 个 V4L2 buffer、6 个推理线程、YOLOv5s ReLU INT8、DMA-BUF 采集、RKNN fd 输入 / copy 输出、MPP copy 输入。两种口径使用同一单调时钟基准，不含网络传输、接收端解码和显示；驱动帧时间戳不等同于已验证的曝光起点。数据来自 `ec21bfc` 基线加板端修改的历史测试版本；延时埋点、RGA 缓冲修复和采集背压现已整合进源码，目录调整后的新验证不覆盖这组历史结果。

### NEON 后处理优化

在同一 INT8 输出样本上 (ReLU 模型, 162 候选 17 检出), 完整后处理耗时由约 **0.204 ms/帧** 降至约 **0.185 ms/帧** (9.5%); NEON 仅加速 objectness 通道阈值扫描, sort/NMS/坐标还原未向量化; 固定 dump 离线基准下加速比有限, 相机管线候选极少场景下收益更明显。

### 输出能力

基于 RGA 完成 YUYV/RGB/NV12 转换, 使用瑞芯微 MPP 进行 H.264 硬编码, 并支持将检测后视频通过 RTMP 推送。

详细测试环境、采样口径和历史对比见 [测试报告](docs/测试报告_性能与验证.md)。

## 目录

```text
Desktop/
├── Readme.md / CMakeLists.txt
├── src/                     # 正式应用；入口 src/main.cpp
│   ├── camera/              # V4L2 MMAP / DMA-BUF 采集
│   ├── common/              # 队列、帧与 buffer 生命周期、日志开关
│   ├── inference/           # RKNN、YOLO、worker 线程池
│   ├── postprocess/         # 标量 / NEON 后处理
│   ├── streaming/           # MPP、RTMP、streamer
│   └── monitoring/          # 阶段统计、延时埋点、CPU/NPU 监控
├── benchmarks/              # 独立后处理性能测试入口
├── tools/                   # DMA/MPP 探针、精度评估、延时 CSV 校验
├── scripts/                 # 启动与延时复现；archive/ 保存历史脚本
├── experiments/             # 精度数据、结果、INT8 dump、历史实现
├── model/                   # 模型与标签；保持原路径
├── 3rdparty/                # 第三方依赖；保持原路径
├── docs/                    # 文档与已发布的 latency_data/
└── debug_records/           # 新实验的本地输出，不提交
```

编译产物仍位于 `build/`，主程序仍为 `build/app`。程序默认模型等相对路径以 `build/` 为工作目录；`scripts/` 下的新启动脚本会自动进入该目录。诊断工具默认参与构建，可用 `-DBUILD_DIAGNOSTIC_TOOLS=OFF` 关闭。离线后处理样本已移到 `experiments/int8_dumps/`，历史脚本与历史报告保留当时的路径记载。

## ReLU 模型量化精度

同一份 ReLU ONNX 在 Ubuntu RKNN-Toolkit2 2.3.2 PC simulator 中对比未量化与 INT8 Normal。验证集排除 50 张校准重叠图片，保留 78 张图片、559 个标注、65 个有标注类别。

| 指标 | ReLU 未量化 | ReLU INT8 Normal | 变化百分点 |
| --- | ---: | ---: | ---: |
| mAP@0.5 | 0.547454 | 0.531546 | -1.5908 |
| mAP@0.5:0.95 | 0.394651 | 0.374578 | -2.0073 |
| Precision@0.25 宏平均 | 0.556062 | 0.560794 | +0.4731 |
| Recall@0.25 宏平均 | 0.463858 | 0.461739 | -0.2118 |

> 这是模拟器量化回归结果，不是板端或完整 COCO 精度成绩。模型仍为 80 类，本子集仅覆盖 65 类标注；已排除校准重叠，但未确认与预训练数据隔离。未量化基线使用 `do_quantization=False`。

逐类别指标、计算方法和复现命令见 [ReLU 模型精度验证报告](docs/ReLU模型mAP统计与计算说明.md)。
