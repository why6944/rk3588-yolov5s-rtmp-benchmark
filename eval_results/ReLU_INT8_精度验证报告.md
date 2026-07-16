# YOLOv5s-ReLU INT8 部署精度验证

## 范围

本次验证评估的是同一个 YOLOv5s-ReLU ONNX 模型从 FP32 到 RKNN INT8 Normal 的量化影响。

它不能证明 SiLU 和 ReLU 两个模型的精度差异，因为 Ubuntu 上没有旧 SiLU 的 ONNX 或权重。

## 环境与数据

- Ubuntu x86_64，RKNN-Toolkit2 2.3.2 PC simulator。
- 模型：`yolov5s_relu.onnx`，输入 640x640。
- FP32：`do_quantization=False`。
- INT8 Normal：`asymmetric_quantized-8`、channel quantization、normal algorithm。
- 测试集：课程自带 `coco128` 的 128 张图像和 YOLO 标签。
- 预处理：RGB 后直接 resize 到 640x640；后处理为 YOLOv5 anchor decode、class-aware NMS (IoU 0.60)。
- mAP 候选框阈值 0.001；Precision/Recall 报告为置信度阈值 0.25 下的宏平均值。

## 校准集隔离

INT8 构建使用的 50 张校准图与 `coco128` 的 128 张图逐个 MD5 比对，发现 50 张完全重叠。

因此不能使用全量 128 张的结果作为独立验证。正式结果已剔除这 50 张校准图，只保留未参与校准的 78 张图像、559 个标注、65 个有标注类别。

## 独立 78 张结果

| 模型 | mAP@0.5 | mAP@0.5:0.95 | Precision@0.25 | Recall@0.25 |
| --- | ---: | ---: | ---: | ---: |
| ReLU FP32 RKNN | 0.5475 | 0.3947 | 0.5561 | 0.4639 |
| ReLU INT8 Normal RKNN | 0.5315 | 0.3746 | 0.5608 | 0.4617 |
| INT8 相对 FP32 | -1.59 pp | -2.01 pp | +0.47 pp | -0.21 pp |

## 结论

- INT8 Normal 相对 FP32 存在可测量的 mAP 回退，mAP@0.5 下降 1.59 个百分点，mAP@0.5:0.95 下降 2.01 个百分点。
- Precision 的微小升高不能单独说明模型更好；它与 Recall 的轻微下降一起看，主要反映量化后置信度和框位置发生了小幅变化。
- 对这个 78 张的独立子集而言，量化没有出现失效级退化。但是否“精度可接受”仍应以业务阈值和更大、无重叠的验证集确认。

## RK3588 板端 SiLU/ReLU A/B

为比较当前实际部署的两份 RKNN 文件，已在 RK3588 上新增独立离线评估器。它不经过摄像头、DMA-BUF、RGA、MPP 或线程池，只做单线程逐图 RKNN Runtime 推理，避免性能链路影响精度比较。

- 数据：同一批 78 张非校准 `coco128` 图片，559 个标注，65 个有标注类别。
- 预处理：OpenCV BGR -> RGB 后直接 resize 到 640x640。
- 后处理：候选阈值 0.001、class-aware NMS IoU 0.60、mAP 使用 IoU 0.50 到 0.95。
- 模型解码：旧 SiLU 模型执行 Sigmoid；新 ReLU 模型按部署参数跳过 Sigmoid。

| RK3588 实际部署模型 | mAP@0.5 | mAP@0.5:0.95 | Precision@0.25 | Recall@0.25 |
| --- | ---: | ---: | ---: | ---: |
| SiLU `yolov5s.rknn` | 0.6607 | 0.4293 | 0.6933 | 0.5625 |
| ReLU `yolov5s_relu_INT8_Normal.rknn` | 0.6261 | 0.4081 | 0.6343 | 0.5327 |
| ReLU 相对 SiLU | -3.46 pp | -2.12 pp | -5.90 pp | -2.98 pp |

这说明在该独立子集上，新 ReLU 部署模型的吞吐提升伴随精度回退。该回退不能全部归因于 SiLU -> ReLU：两个已导出的模型还可能存在训练权重、量化校准或转换配置差异。要精确归因，需要同一训练配方下只改变激活函数，再使用更大验证集进行 A/B。

板端复现产物：

- 指标：`/home/orangepi/streamer_codev5.0/Desktop/eval_results/board_accuracy_metrics.json`
- 预测框：`silu_predictions.json`、`relu_predictions.json`
- 评估器：`rknn_eval.cpp`，编译产物 `build/rknn_eval`
- 模型 SHA-256：SiLU `9619248441d1b2008b55b1f74ef720159f558df8b945d01400a1c12ea192b90d`；ReLU `6dc67c3c5fd1e18e6e80d24ab6c91606c22c80b818790f3c2089c96c6b56535b`。

## 限制与下一步

1. RKNN PC simulator 用于量化回归，不能替代 RK3588 NPU 实机精度验收。
2. `coco128` 仅 78 张独立图像，不能替代 COCO val2017 或项目自己的验证集。
3. 已完成实际 RK3588 上的已部署 SiLU/ReLU RKNN A/B；但若要归因于激活函数，需要同一训练配方下的 SiLU/ReLU 权重或 ONNX。
4. 下一轮应使用不参与校准的更大验证集，优先选择项目自有的安全帽/安防数据，以确认 COCO 子集上的精度取舍是否会在目标场景复现。

## 复现

在 Ubuntu 的量化实验目录中执行：

```bash
/home/hao/miniconda3/envs/rknn-toolkit2/bin/python eval_rknn_coco128.py \
  --modes fp32 normal \
  --exclude-calibration \
  --output logs/rknn_coco128_independent_metrics.json
```

评估脚本：`eval_rknn_coco128.py`。
结果 JSON：`logs/rknn_coco128_independent_metrics.json`。
