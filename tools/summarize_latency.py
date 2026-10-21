#!/usr/bin/env python3
"""Validate and summarize frame traces using raw CLOCK_MONOTONIC nanoseconds."""
import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path


TIMESTAMPS = (
    "dqbuf_ns", "read_enqueue_ns", "aggregator_dequeue_ns", "worker_start_ns",
    "inference_end_ns", "aggregator_ready_ns", "write_dequeue_ns", "rga_start_ns",
    "rga_end_ns", "mpp_submit_ns", "mpp_complete_ns",
)
INTERVALS = {
    "dqbuf_to_mpp_ms": ("dqbuf_ns", "mpp_complete_ns"),
    "capture_prepare_ms": ("dqbuf_ns", "read_enqueue_ns"),
    "read_queue_ms": ("read_enqueue_ns", "worker_start_ns"),
    "worker_processing_ms": ("worker_start_ns", "inference_end_ns"),
    "reorder_write_queue_ms": ("inference_end_ns", "write_dequeue_ns"),
    "rga_ms": ("rga_start_ns", "rga_end_ns"),
    "mpp_ms": ("mpp_submit_ns", "mpp_complete_ns"),
}


def summarize(values):
    values = sorted(values)
    def percentile(fraction):
        position = fraction * (len(values) - 1)
        lower = math.floor(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)
    return {"samples": len(values), "mean": statistics.fmean(values),
            **{f"p{p}": percentile(p / 100) for p in (50, 90, 95, 99)},
            "max": values[-1]}


def metrics(rows):
    result = {name: summarize([(int(r[end]) - int(r[start])) / 1e6 for r in rows])
              for name, (start, end) in INTERVALS.items()}
    valid_driver_rows = [r for r in rows if r["v4l2_timestamp_valid"] == "1"]
    result["driver_timestamp_valid_samples"] = len(valid_driver_rows)
    if valid_driver_rows:
        for name, end in (("v4l2_to_mpp_ms", "mpp_complete_ns"),
                          ("driver_timestamp_to_dqbuf_ms", "dqbuf_ns")):
            result[name] = summarize([
                (int(r[end]) - int(r["v4l2_timestamp_ns"])) / 1e6
                for r in valid_driver_rows])
    return result


def load_run(path, expected, warmup):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected:
        raise ValueError(f"{path}: expected {expected} records, got {len(rows)}")
    indices = [int(row["frame_index"]) for row in rows]
    if sorted(indices) != list(range(expected)):
        raise ValueError(f"{path}: missing, duplicate or unexpected application frame indices")
    for row in rows:
        if row["mpp_success"] != "1":
            raise ValueError(f"{path}: encoding failed at frame {row['frame_index']}")
        timestamps = [int(row[name]) for name in TIMESTAMPS]
        if timestamps[0] < 0 or timestamps != sorted(timestamps):
            raise ValueError(f"{path}: invalid timestamp order at frame {row['frame_index']}")
        if row["in_statistics"] != str(int(int(row["frame_index"]) >= warmup)):
            raise ValueError(f"{path}: trace warmup does not match --warmup {warmup}")
        if row["v4l2_timestamp_valid"] == "1":
            driver_time = int(row["v4l2_timestamp_ns"])
            if driver_time < 0 or driver_time > timestamps[0]:
                raise ValueError(f"{path}: invalid driver timestamp at frame {row['frame_index']}")
    selected = [r for r in rows if r["in_statistics"] == "1"]
    return selected, {"file": str(path), "records": len(rows),
                      "warmup": warmup, "successful_samples": len(selected),
                      "metrics": metrics(selected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--expected", type=int, default=1000, help="total frames per run")
    parser.add_argument("--warmup", type=int, default=100, help="excluded initial frames")
    args = parser.parse_args()
    if not 0 <= args.warmup < args.expected:
        parser.error("require expected > warmup >= 0")
    try:
        samples, runs = [], []
        for path in args.csv:
            selected, run = load_run(path, args.expected, args.warmup)
            samples.extend(selected)
            runs.append(run)
        print(json.dumps({"runs": runs, "pooled": metrics(samples)},
                         ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
