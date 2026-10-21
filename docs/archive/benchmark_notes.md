# RK3588 YOLOv5s Thread Benchmark Notes

## Scope

Input: `/home/orangepi/streamer_codev5.0/Desktop/video.mp4`

- Resolution: 1920x1080
- FPS: 25
- Frames: 341
- Duration: 13.64 s

Build config:

- `ENABLE_RTMP=OFF`
- Mode: full pipeline (`VideoCapture -> RKNN inference -> draw -> output.avi -> MPP path`)
- Output: `/home/orangepi/streamer_codev5.0/Desktop/output.avi`

## Code Changes In This Round

- Fixed thread-unsafe label loading in `post_process.cpp` with `std::call_once`.
- Added bounds checks for `labels[id]` and fixed label string termination.
- Added `rknn_outputs_release()` after post-process in `yolov5s.cpp`.
- Removed duplicate drawing: inference no longer draws; `thread_poll.cpp` draws once after inference.
- Added `BenchmarkStats` for stage timing and CSV summary.
- Added runtime thread selection with `./app --threads N`.

## Full Pipeline Results

| Threads | Frames | Elapsed ms | End-to-end FPS | Preprocess avg ms | RKNN avg ms | Postprocess avg ms | Draw avg ms | AVI write avg ms | MPP path avg ms | Avg CPU % | NPU available |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 341 | 21444 | 15.902 | 11.180 | 31.480 | 6.220 | 2.137 | 42.862 | 11.269 | 88.8 | 0 |
| 6 | 341 | 21634 | 15.762 | 16.085 | 52.311 | 7.502 | 2.316 | 42.713 | 11.768 | 88.6 | 0 |
| 9 | 341 | 21894 | 15.575 | 19.584 | 73.364 | 6.351 | 2.139 | 43.126 | 11.554 | 87.7 | 0 |
| 12 | 341 | 22036 | 15.475 | 21.842 | 85.947 | 6.061 | 2.041 | 43.868 | 11.396 | 87.4 | 0 |

## Findings

1. Full-pipeline FPS is currently limited by the output path, especially OpenCV `VideoWriter::write()` at about 43 ms/frame.
2. Increasing worker count from 3 to 12 does not improve end-to-end FPS in this mode. FPS slightly decreases from 15.9 to 15.5.
3. Per-frame RKNN latency increases as thread count rises, which is expected when more RKNN contexts contend for the same three NPU cores.
4. CPU utilization is already high, around 87-89%, so additional threads mainly add contention.
5. NPU utilization could not be measured by the current user because `/sys/kernel/debug/rknpu/load` returns permission denied. `PerfMonitor` recorded `npu_available=0`.

## Conclusion

For the current complete video output pipeline, 3 worker threads are the best measured setting. The 12-thread configuration does not help because the benchmark includes frame drawing, AVI writing, and the MPP conversion/encoding path.

The resume claim `12 threads, 150+ FPS, NPU utilization >=95%` should not be validated with this full-pipeline mode. It needs a separate `infer-only` or `rknn-only` benchmark that disables video writing, MPP encoding, most logs, and possibly drawing.

## Recommended Next Experiments

1. Add `--mode infer-only`: read frames, run preprocess + RKNN + postprocess, skip AVI write and MPP.
2. Add `--mode rknn-only`: reuse one preprocessed input buffer and repeatedly call RKNN to measure pure NPU throughput.
3. Disable per-frame logs during benchmark builds.
4. Run NPU utilization collection with permission to read `/sys/kernel/debug/rknpu/load`.
5. Re-test 3/6/9/12 threads in `infer-only` and `rknn-only` modes.
