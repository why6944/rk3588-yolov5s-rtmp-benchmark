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

### 算子融合与耗时热点分析

通过 RKNN 逐层性能报告，对比两份部署模型的融合执行层。SiLU 版本包含 57 个 `ConvExSwish`；ReLU 版本包含 50 个 `ConvRelu`、7 个 `ConvReluAdd` 和 3 个 `ConvSigmoid`。两版均已发生算子融合，差别不是“SiLU 未融合、ReLU 才融合”。

| 热点位置 | 实测中位耗时 | 耗时占比 |
| --- | ---: | ---: |
| SiLU 首个 ConvExSwish | 4.074 ms | 13.78% |
| ReLU 首部 Conv + ConvRelu | 合计 2.177 ms | 9.42% |
| ReLU 80×80 检测头 ConvSigmoid | 0.966 ms | 4.18% |
| ReLU 全部 13 个 Concat | 合计 3.602 ms | 15.58% |

SiLU 的最大单层热点集中在首部融合算子；ReLU 的热点更分散，主要涉及首部卷积、高分辨率特征拼接和检测头，且总耗时降低。两版 Concat 合计耗时接近，ReLU 中其占比上升主要是总耗时下降，并不代表 Concat 变慢。

> 本节测试条件：Runtime 1.5.3b6，单 context、NPU core 0、batch=1、640×640 输入，开启性能采集，每模型采样 90 次。单层时间为样本中位数，分组时间为各层中位数之和；占比按全部执行层中位数之和计算。本节与上表 Runtime 2.3.2 的延迟测试分开统计，不直接混算。

详见 [YOLOv5 双模型算子融合与耗时分析](docs/YOLOv5双模型算子融合与耗时分析.md)。

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

> FPS 为并发吞吐，不是单帧延迟：6 线程平均每隔约 6.3 ms 完成一帧，但每次 `rknn_run` 平均耗时仍为 37.6 ms（含等待）。


### Camera + MPP 端到端

- V4L2 Camera (640x480 YUYV@30fps) + RGA + RKNN + MPP: **28.1 FPS** (6 线程)
- rknn_run: 18.4ms, preprocess: 1.77ms, MPP: 2.4ms
- DMA-BUF fd 链路 (ReLU 模型): app 进程 CPU 单核等效占用约 **13%** (实测 12-15.5%, 条件: V4L2_BUFFER_COUNT=8, dmabuf 后端, rknn-input-mode fd)

采集、推理与编码并行处理不同帧，因此**约 35 ms 输出一帧（28～29 FPS）与单帧延迟约 66 ms 不矛盾**。

![流水线吞吐与单帧延迟：不同帧重叠处理，单帧约 66 ms，相邻完成时刻约隔 35 ms](docs/camera_pipeline_analysis_20261025/throughput_latency.png)

> 概念示意，参考下文复测；不是实测调度轨迹。约 80 ms 为 P95，不是平均延迟。

### Camera + MPP 链路延时

历史测试（2026-10-21）：摄像头 → RGA → RKNN → 后处理与画框 → MPP 编码，3 轮共 2700 个有效样本（每轮 1000 帧，预热 100 帧）。

| 测量范围 | 样本数 | 均值（ms） | P50（ms） | P95（ms） | P99（ms） |
|---|---:|---:|---:|---:|---:|
| DQBUF 成功后 → MPP 完整编码包取得后 | 2700 | 32.38 | 29.71 | 46.64 | 61.43 |
| V4L2 驱动帧时间戳 → MPP 完整编码包取得后 | 2700 | 66.41 | 64.08 | 81.55 | 98.79 |

> 条件：640×480 YUYV@30 FPS、8 buffers、6 线程、ReLU INT8；DMA-BUF 采集、RKNN fd 输入 / copy 输出、MPP copy 输入。不含网络、接收端解码与显示。历史版本与测量方法见[延时测试报告](docs/板端链路延时测试报告_20261021.md)。

#### 复测：链路耗时分解（2026-10-25 至 26）

稳态输出 **28.77 FPS**（三轮中位数）；取帧至编码完成平均 **32.02 ms**，驱动时间戳至编码完成平均 **65.95 ms**、P95 **78.81 ms**。

![6 线程完整链路 2700 帧平均耗时分解：取帧前 33.93 ms，取帧后至编码完成 32.02 ms](docs/camera_pipeline_analysis_20261025/latency_breakdown.png)

推理工作段平均 **22.53 ms**：前处理 1.33 ms、`rknn_run` 18.37 ms、取输出 2.57 ms、后处理 0.055 ms、画框 0.189 ms，其余为少量框架开销。

详见 [RK3588 摄像头链路帧率与延迟瓶颈实验](docs/RK3588摄像头链路帧率与延迟瓶颈实验.md)。

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
