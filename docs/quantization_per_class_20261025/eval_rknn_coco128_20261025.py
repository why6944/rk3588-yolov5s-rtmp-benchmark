#!/usr/bin/env python3
"""Evaluate YOLOv5 RKNN models on the bundled coco128 YOLO-label dataset.

This is an RKNN PC-simulator deployment regression check. It compares the
FP32 and quantized RKNN exports with identical preprocessing and postprocess.
It is not a substitute for COCO val2017 or a task-specific validation set.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np


MODEL_BY_MODE = {
    "fp32": "yolov5s_relu_FP32.rknn",
    "normal": "yolov5s_relu_INT8_Normal.rknn",
    "mmse": "yolov5s_relu_INT8_MMSE.rknn",
    "hybrid": "yolov5s_relu_Hybrid.rknn",
}

IMAGE_SIZE = 640
NUM_CLASSES = 80
ANCHORS = np.array(
    [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119],
     [116, 90], [156, 198], [373, 326]],
    dtype=np.float32,
)
MASKS = ((0, 1, 2), (3, 4, 5), (6, 7, 8))


def xywh_to_xyxy(boxes):
    result = boxes.copy()
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return result


def box_iou(box1, box2):
    if len(box1) == 0 or len(box2) == 0:
        return np.zeros((len(box1), len(box2)), dtype=np.float32)

    area1 = np.maximum(0, box1[:, 2] - box1[:, 0]) * np.maximum(0, box1[:, 3] - box1[:, 1])
    area2 = np.maximum(0, box2[:, 2] - box2[:, 0]) * np.maximum(0, box2[:, 3] - box2[:, 1])
    inter_x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])
    inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    return inter / (area1[:, None] + area2[None, :] - inter + 1e-16)


def nms_indices(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)
    xywh = np.column_stack((
        boxes[:, 0], boxes[:, 1],
        np.maximum(0, boxes[:, 2] - boxes[:, 0]),
        np.maximum(0, boxes[:, 3] - boxes[:, 1]),
    ))
    kept = cv2.dnn.NMSBoxes(xywh.tolist(), scores.astype(float).tolist(), 0.0, iou_threshold)
    if len(kept) == 0:
        return np.empty(0, dtype=np.int64)
    return np.asarray(kept, dtype=np.int64).reshape(-1)


def decode_head(head, anchors, candidate_threshold):
    grid_h, grid_w = head.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
    grid = np.stack((grid_x, grid_y), axis=-1).reshape(grid_h, grid_w, 1, 2)

    objectness = head[..., 4]
    class_prob = head[..., 5:]
    class_id = np.argmax(class_prob, axis=-1)
    class_score = np.max(class_prob, axis=-1)
    score = objectness * class_score
    keep = score >= candidate_threshold
    if not np.any(keep):
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )

    xy = (head[..., :2] * 2.0 - 0.5 + grid) * (IMAGE_SIZE / grid_h)
    wh = (head[..., 2:4] * 2.0) ** 2 * anchors.reshape(1, 1, 3, 2)
    boxes = xywh_to_xyxy(np.concatenate((xy, wh), axis=-1).reshape(-1, 4))
    return boxes[keep.reshape(-1)], score[keep].astype(np.float32), class_id[keep].astype(np.int64)


def decode_outputs(outputs, candidate_threshold, nms_threshold):
    all_boxes, all_scores, all_classes = [], [], []
    if len(outputs) != 3:
        raise RuntimeError("Expected 3 YOLO heads, got {}".format(len(outputs)))

    for output, mask in zip(outputs, MASKS):
        if output.ndim != 4:
            raise RuntimeError("Unexpected RKNN output shape: {}".format(output.shape))
        reshaped = output.reshape([3, -1] + list(output.shape[-2:]))
        head = np.transpose(reshaped, (2, 3, 0, 1))
        boxes, scores, classes = decode_head(head, ANCHORS[list(mask)], candidate_threshold)
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_classes.append(classes)

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    classes = np.concatenate(all_classes, axis=0)
    if len(boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)

    selected = []
    for class_id in np.unique(classes):
        indices = np.where(classes == class_id)[0]
        selected.extend(indices[nms_indices(boxes[indices], scores[indices], nms_threshold)].tolist())
    selected = np.asarray(selected, dtype=np.int64)
    detections = np.column_stack((boxes[selected], scores[selected], classes[selected]))
    return detections[np.argsort(-detections[:, 4])].astype(np.float32)


def load_labels(label_path):
    if not label_path.exists() or label_path.stat().st_size == 0:
        return np.empty((0, 5), dtype=np.float32)
    labels = np.loadtxt(label_path, dtype=np.float32, ndmin=2)
    boxes = labels[:, 1:5].copy()
    boxes[:, [0, 2]] *= IMAGE_SIZE
    boxes[:, [1, 3]] *= IMAGE_SIZE
    return np.column_stack((labels[:, 0], xywh_to_xyxy(boxes))).astype(np.float32)


def match_detections(detections, labels, iouv):
    correct = np.zeros((len(detections), len(iouv)), dtype=bool)
    if len(detections) == 0 or len(labels) == 0:
        return correct

    iou = box_iou(labels[:, 1:5], detections[:, :4])
    same_class = labels[:, 0:1] == detections[:, 5]
    for index, threshold in enumerate(iouv):
        label_indices, detection_indices = np.where((iou >= threshold) & same_class)
        if len(label_indices) == 0:
            continue
        matches = np.column_stack((label_indices, detection_indices, iou[label_indices, detection_indices]))
        matches = matches[np.argsort(-matches[:, 2])]
        _, keep_detection = np.unique(matches[:, 1], return_index=True)
        matches = matches[np.sort(keep_detection)]
        matches = matches[np.argsort(-matches[:, 2])]
        _, keep_label = np.unique(matches[:, 0], return_index=True)
        matches = matches[np.sort(keep_label)]
        correct[matches[:, 1].astype(np.int64), index] = True
    return correct


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    points = np.linspace(0.0, 1.0, 101)
    return np.trapz(np.interp(points, mrec, mpre), points)


def summarize_metrics(correct, confidences, prediction_classes, target_classes, pr_threshold):
    if len(target_classes) == 0:
        raise RuntimeError("The dataset contains no labels.")
    classes = np.unique(target_classes.astype(np.int64))
    ap = np.zeros((len(classes), 10), dtype=np.float32)
    precision_at_threshold = []
    recall_at_threshold = []
    per_class = []

    for class_index, class_id in enumerate(classes):
        selected_predictions = prediction_classes == class_id
        n_labels = int((target_classes == class_id).sum())
        class_confidence = confidences[selected_predictions]
        class_correct = correct[selected_predictions]
        tp, fp = 0, 0
        if len(class_confidence):
            order = np.argsort(-class_confidence)
            class_confidence = class_confidence[order]
            class_correct = class_correct[order]
            true_positive = np.cumsum(class_correct, axis=0)
            false_positive = np.cumsum(~class_correct, axis=0)
            for iou_index in range(10):
                recall = true_positive[:, iou_index] / (n_labels + 1e-16)
                precision = true_positive[:, iou_index] / (true_positive[:, iou_index] + false_positive[:, iou_index] + 1e-16)
                ap[class_index, iou_index] = compute_ap(recall, precision)

            threshold_index = np.where(class_confidence >= pr_threshold)[0]
            if len(threshold_index):
                last = threshold_index[-1]
                tp = true_positive[last, 0]
                fp = false_positive[last, 0]
                precision_at_threshold.append(float(tp / (tp + fp + 1e-16)))
                recall_at_threshold.append(float(tp / (n_labels + 1e-16)))
            else:
                precision_at_threshold.append(0.0)
                recall_at_threshold.append(0.0)
        else:
            precision_at_threshold.append(0.0)
            recall_at_threshold.append(0.0)

        per_class.append({
            "class_id": int(class_id),
            "labels": n_labels,
            "detections": int(len(class_confidence)),
            "ap50": float(ap[class_index, 0]),
            "ap50_95": float(ap[class_index].mean()),
            "precision_at_threshold": precision_at_threshold[-1],
            "recall_at_threshold": recall_at_threshold[-1],
            "tp_at_threshold": int(tp),
            "fp_at_threshold": int(fp),
            "fn_at_threshold": n_labels - int(tp),
        })

    return {
        "classes_with_labels": int(len(classes)),
        "precision_at_threshold_macro": float(np.mean(precision_at_threshold)),
        "recall_at_threshold_macro": float(np.mean(recall_at_threshold)),
        "map50": float(ap[:, 0].mean()),
        "map50_95": float(ap.mean()),
        "per_class": per_class,
    }


def md5sum(path):
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_paths(dataset_dir, limit, excluded_hashes=None):
    paths = sorted((dataset_dir / "images" / "train2017").glob("*.jpg"))
    if not paths:
        raise RuntimeError("No JPEG images found under {}".format(dataset_dir))
    if excluded_hashes:
        paths = [path for path in paths if md5sum(path) not in excluded_hashes]
        if not paths:
            raise RuntimeError("All evaluation images were excluded.")
    return paths[:limit] if limit else paths


def evaluate_mode(mode, workspace, dataset_dir, candidate_threshold, nms_threshold, pr_threshold, limit, excluded_hashes, artifacts_dir=None):
    from rknn.api import RKNN

    onnx_path = workspace / "yolov5s_relu.onnx"
    calibration_path = workspace / "dataset.txt"
    if not onnx_path.exists():
        raise RuntimeError("ONNX model not found: {}".format(onnx_path))

    rknn = RKNN(verbose=False)
    config = {
        "mean_values": [[0, 0, 0]],
        "std_values": [[255, 255, 255]],
        "target_platform": "rk3588",
        "quantized_method": "channel",
    }
    if mode == "normal":
        config["quantized_algorithm"] = "normal"
    if rknn.config(**config) != 0:
        rknn.release()
        raise RuntimeError("Failed to configure {}".format(mode))
    if rknn.load_onnx(model=str(onnx_path)) != 0:
        rknn.release()
        raise RuntimeError("Failed to load {}".format(onnx_path))
    if mode == "fp32":
        build_result = rknn.build(do_quantization=False)
    elif mode == "normal":
        if not calibration_path.exists():
            rknn.release()
            raise RuntimeError("Calibration list not found: {}".format(calibration_path))
        build_result = rknn.build(do_quantization=True, dataset=str(calibration_path))
    else:
        rknn.release()
        raise RuntimeError("Only fp32 and normal are supported by this evaluator.")
    if build_result != 0:
        rknn.release()
        raise RuntimeError("Failed to build {} model in the RKNN simulator.".format(mode))
    if rknn.init_runtime() != 0:
        rknn.release()
        raise RuntimeError("Failed to initialize RKNN simulator for {}".format(mode))

    all_correct, all_confidences, all_prediction_classes, all_target_classes = [], [], [], []
    all_detections, all_labels, manifest = [], [], []
    label_images = np.zeros(NUM_CLASSES, dtype=np.int64)
    paths = image_paths(dataset_dir, limit, excluded_hashes)
    start = time.monotonic()
    try:
        for number, image_path in enumerate(paths, 1):
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError("Unable to read {}".format(image_path))
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            model_input = np.expand_dims(cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE)), axis=0)
            outputs = rknn.inference(inputs=[model_input], data_format=["nhwc"])
            detections = decode_outputs(outputs, candidate_threshold, nms_threshold)
            label_path = dataset_dir / "labels" / "train2017" / (image_path.stem + ".txt")
            labels = load_labels(label_path)
            all_correct.append(match_detections(detections, labels, np.linspace(0.5, 0.95, 10)))
            all_confidences.append(detections[:, 4])
            all_prediction_classes.append(detections[:, 5])
            all_target_classes.append(labels[:, 0])
            all_detections.append(detections)
            all_labels.append(labels)
            for class_id in np.unique(labels[:, 0].astype(np.int64)):
                label_images[class_id] += 1
            manifest.append({
                "image": image_path.name,
                "image_md5": md5sum(image_path),
                "label_md5": md5sum(label_path) if label_path.exists() else None,
                "labels": int(len(labels)),
                "detections": int(len(detections)),
            })
            if number % 16 == 0 or number == len(paths):
                print("  {}: {}/{} images".format(mode, number, len(paths)), flush=True)
    finally:
        rknn.release()

    correct = np.concatenate(all_correct, axis=0) if all_correct else np.empty((0, 10), dtype=bool)
    confidences = np.concatenate(all_confidences, axis=0) if all_confidences else np.empty(0, dtype=np.float32)
    prediction_classes = np.concatenate(all_prediction_classes, axis=0) if all_prediction_classes else np.empty(0, dtype=np.float32)
    target_classes = np.concatenate(all_target_classes, axis=0) if all_target_classes else np.empty(0, dtype=np.float32)
    metrics = summarize_metrics(correct, confidences, prediction_classes, target_classes, pr_threshold)
    for row in metrics["per_class"]:
        row["images_with_labels"] = int(label_images[row["class_id"]])
    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifacts_dir / (mode + "_predictions.npz"),
            correct=correct, confidences=confidences,
            prediction_classes=prediction_classes, target_classes=target_classes,
            detections=np.concatenate(all_detections), labels=np.concatenate(all_labels),
            detection_offsets=np.cumsum([0] + [len(x) for x in all_detections]),
            label_offsets=np.cumsum([0] + [len(x) for x in all_labels]),
            image_names=np.asarray([path.name for path in paths]),
        )
        (artifacts_dir / (mode + "_manifest.json")).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    metrics.update({
        "mode": mode,
        "model": MODEL_BY_MODE[mode],
        "source_onnx": onnx_path.name,
        "images": len(paths),
        "labels": int(len(target_classes)),
        "detections": int(len(confidences)),
        "elapsed_seconds": round(time.monotonic() - start, 3),
    })
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--modes", nargs="+", choices=sorted(MODEL_BY_MODE), default=["fp32", "normal"])
    parser.add_argument("--limit", type=int, default=0, help="Limit images for a smoke test; 0 evaluates all images.")
    parser.add_argument("--candidate-threshold", type=float, default=0.001)
    parser.add_argument("--nms-threshold", type=float, default=0.60)
    parser.add_argument("--pr-threshold", type=float, default=0.25)
    parser.add_argument("--exclude-calibration", action="store_true", help="Exclude images reused by the INT8 calibration set.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    dataset = (args.dataset or workspace / "assets" / "coco128").resolve()
    output = (args.output or workspace / "logs" / "rknn_coco128_metrics.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    excluded_hashes = set()
    if args.exclude_calibration:
        calibration_dir = workspace / "calibration"
        if not calibration_dir.exists():
            raise RuntimeError("Calibration directory not found: {}".format(calibration_dir))
        excluded_hashes = {md5sum(path) for path in calibration_dir.glob("*.jpg")}
        print("Excluding {} calibration images by content hash.".format(len(excluded_hashes)), flush=True)

    results = []
    for mode in args.modes:
        print("Evaluating {} on {}...".format(mode, dataset), flush=True)
        result = evaluate_mode(
            mode, workspace, dataset, args.candidate_threshold,
            args.nms_threshold, args.pr_threshold, args.limit, excluded_hashes,
            args.artifacts_dir,
        )
        results.append(result)
        print(
            "  mAP50={:.4f} mAP50-95={:.4f} P@{:.2f}={:.4f} R@{:.2f}={:.4f}".format(
                result["map50"], result["map50_95"], args.pr_threshold,
                result["precision_at_threshold_macro"], args.pr_threshold,
                result["recall_at_threshold_macro"],
            ),
            flush=True,
        )

    report = {
        "dataset": str(dataset),
        "evaluation": "RKNN PC simulator with direct 640x640 resize and YOLOv5 anchor decode",
        "candidate_threshold": args.candidate_threshold,
        "nms_threshold": args.nms_threshold,
        "precision_recall_threshold": args.pr_threshold,
        "excluded_calibration_images": len(excluded_hashes),
        "source_onnx_sha256": hashlib.sha256((workspace / "yolov5s_relu.onnx").read_bytes()).hexdigest(),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "calibration_list_sha256": hashlib.sha256((workspace / "dataset.txt").read_bytes()).hexdigest(),
        "results": results,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Saved metrics to {}".format(output))


if __name__ == "__main__":
    main()

