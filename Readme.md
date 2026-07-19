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

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [系统设计与架构](docs/设计文档_系统架构.md) | 模块边界、线程模型、数据流与 DMA-BUF 生命周期 |
| [API 与运行参数](docs/API文档_运行参数与模块接口.md) | 命令行、运行模式、输入后端和关键 C++ 接口 |
| [测试报告](docs/测试报告_性能与验证.md) | 性能口径、吞吐、DMA-BUF 对比与限制 |
| [SiLU vs ReLU A/B 实验](docs/AB_REPORT_SiLU_vs_ReLU.md) | 激活函数对比实验: 延迟、P50/P95、算子融合分析 |
| [模型替换对比](docs/模型替换对比报告.md) | 旧模型→INT8-Normal 替换验证与性能基准 |
| [Perf 教学与调试](docs/RK3588_perf性能分析教学调试文档.md) | perf 安装、采样、报告解读和实测过程 |
| [代码问题深度讲解](docs/代码问题深度讲解.md) | 早期并发与预处理问题的分析和修复思路 |

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
- DMA-BUF 链路: app 进程 CPU 单核等效占用约 **18.3%**

### NEON 后处理优化

在同一 INT8 输出样本上 (ReLU 模型, 162 候选 17 检出), 完整后处理耗时由约 **0.204 ms/帧** 降至约 **0.189 ms/帧** (7.1%); NEON 仅加速 objectness 通道阈值扫描, sort/NMS/坐标还原未向量化; 固定 dump 离线基准下加速比有限, 相机管线候选极少场景下收益更明显。

### 输出能力

基于 RGA 完成 YUYV/RGB/NV12 转换, 使用瑞芯微 MPP 进行 H.264 硬编码, 并支持将检测后视频通过 RTMP 推送。

详细测试环境、采样口径和历史对比见 [测试报告](docs/测试报告_性能与验证.md)。

## 目录

```text
Desktop/
├── main.cpp                 # 命令行和流水线编排
├── camera_dmabuf.*          # V4L2 mmap / DMA-BUF Camera 采集
├── frame_data.h             # 帧与 Camera buffer 生命周期
├── yolov5s.*                # RGA、RKNN、后处理、画框
├── thread_poll.*            # 多 context worker 线程池
├── mpp.* / streamer.*       # H.264 编码和 RTMP 输出
├── benchmark_stats.*        # 阶段统计和 CSV 输出
├── docs/                    # 项目文档
└── debug_records/           # 本地日志和 perf 数据，不提交
```
