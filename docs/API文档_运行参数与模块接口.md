# API 文档：运行参数与模块接口

## 1. 命令行参数

| 参数 | 可选值/示例 | 说明 |
| --- | --- | --- |
| `--mode` | `full`、`infer-only`、`rknn-only`、`mpp-only`、`rtmp`、`snapshot` | 运行模式 |
| `--input` | `video`、`camera` | 输入来源 |
| `--input-backend` | `opencv`、`v4l2-mmap`、`dmabuf` | Camera 输入后端 |
| `--video-path` | `../video.mp4` | 视频文件路径 |
| `--camera-id` | `0` | `/dev/videoN` 的 N |
| `--camera-width` / `--camera-height` | `640` / `480` | 申请的摄像头尺寸 |
| `--camera-fps` | `30` | 申请的摄像头帧率 |
| `--camera-format` | `YUYV`、`MJPG` | 申请的像素格式；直读 RGA 路径要求 YUYV |
| `--threads` | `6` | `ThreadPoll` worker / RKNN context 数量 |
| `--loops` | `600` | Camera 处理帧数；rknn-only 循环次数 |
| `--box-threshold` | `0.6` | 置信度阈值 |
| `--nms-threshold` | `0.5` | NMS IoU 阈值 |
| `--rknn-input-mode` | `copy`、`fd` | RKNN 常规输入或 fd 预绑定输入 |
| `--rknn-output-mode` | `copy`、`mem` | RKNN 输出获取方式 |
| `--mpp-input-mode` | `copy`、`fd` | MPP 内部/外部 DMA-BUF 输入 |
| `--rtmp-url` | `rtmp://host:1935/live/key` | RTMP 推流地址 |
| `--snapshot-frame` | `120` | snapshot 模式导出帧号 |
| `--snapshot-output` | `../debug_records/snapshot.png` | snapshot 文件路径 |

环境变量：

```bash
V4L2_BUFFER_COUNT=8
```

设置 V4L2 mmap buffer 数量，支持范围由程序限制为 2 至 32。

## 2. 核心类型

### `FrameData`

定义于 `frame_data.h`，是读队列、线程池和写队列之间传递的一帧数据。它包含：

- `index`：用于 future 聚合和按帧号保序；
- `cv::Mat frame`：CPU 侧图像，供画框和 copy 路径使用；
- `DmaBufFrameRef`：Camera fd、尺寸、格式、RGA handle、V4L2 buffer index 与 QBUF 回调；
- `FrameStorageType`：区分 Mat、Camera mmap 与 Camera DMA-BUF 来源。

### `DmaBufFrameRef`

其析构函数执行 `release` 回调。Camera 帧在 worker/写线程引用全部结束前不会 QBUF，因此必须避免随意复制并提前销毁该对象。

## 3. 关键模块接口

| 模块 | 关键接口 | 职责 |
| --- | --- | --- |
| `CameraDmaBufCapture` | `open()`、`readFrameData()`、`close()` | V4L2 设置、mmap、DQBUF/QBUF、可选 fd 导出 |
| `ThreadPoll` | `submit_task_async(FrameData)` | 将帧提交给 worker，并返回 `future<ProcessResult>` |
| `Yolov5s` | `inference_frame()`、`inference_image()`、`draw_result()` | RGA 预处理、RKNN 推理、后处理与画框 |
| `BenchmarkStats` | `record_inference()`、`append_csv()` | 汇总 FPS 与阶段耗时 |
| `PerfMonitor` | `start()`、`stop()` | 采样 CPU/NPU 指标并写 CSV |
| `streamer` / `mpp` | `init_streamer()`、`process_image()`、`close_streamer()` | MPP 编码和可选 RTMP 封装 |

## 4. 常用运行示例

纯 NPU 吞吐：

```bash
./app --mode rknn-only --input video --threads 6 --loops 900
```

Camera mmap baseline：

```bash
V4L2_BUFFER_COUNT=8 ./app --mode mpp-only --input camera \
  --camera-id 0 --camera-width 640 --camera-height 480 --camera-fps 30 \
  --camera-format YUYV --threads 6 --input-backend v4l2-mmap \
  --rknn-input-mode copy --rknn-output-mode copy --mpp-input-mode copy
```

Camera DMA-BUF + RKNN fd：

```bash
V4L2_BUFFER_COUNT=8 ./app --mode mpp-only --input camera \
  --camera-id 0 --camera-width 640 --camera-height 480 --camera-fps 30 \
  --camera-format YUYV --threads 6 --input-backend dmabuf \
  --rknn-input-mode fd --rknn-output-mode copy --mpp-input-mode copy
```
