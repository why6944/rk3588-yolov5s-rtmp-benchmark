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
| [Perf 教学与调试](docs/RK3588_perf性能分析教学调试文档.md) | perf 安装、采样、报告解读和实测过程 |
| [代码问题深度讲解](docs/代码问题深度讲解.md) | 早期并发与预处理问题的分析和修复思路 |

## 性能摘要

性能应与测试条件一起看：纯 RKNN 固定输入吞吐约 111 FPS；1080p 视频输入下，MPP-only 约 62 FPS、RTMP 约 56 FPS；640x480 YUYV@30 的 Camera DMA-BUF + RKNN fd 正常运行约 27.8 FPS。详细口径见 [测试报告](docs/测试报告_性能与验证.md)。

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
