# Sudo NPU Utilization Supplement

The previous benchmark could not read `/sys/kernel/debug/rknpu/load` as the normal `orangepi` user. The same full-pipeline benchmark was rerun with:

```bash
sudo env LD_LIBRARY_PATH=/home/orangepi/streamer_codev5.0/Desktop/3rdparty/librknn_api/aarch64:/home/orangepi/streamer_codev5.0/Desktop/3rdparty/rga/RK3588/lib/Linux/aarch64:$LD_LIBRARY_PATH ./app --threads N
```

## Full Pipeline Results With NPU Monitor

| Threads | FPS | Preprocess avg ms | RKNN avg ms | Postprocess avg ms | Draw avg ms | AVI write avg ms | MPP path avg ms | Avg CPU % | Avg NPU0 % | Avg NPU1 % | Avg NPU2 % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 16.227 | 10.182 | 31.078 | 5.699 | 1.969 | 41.834 | 11.290 | 90.8 | 13.5 | 12.9 | 13.2 |
| 6 | 15.529 | 15.570 | 50.107 | 6.649 | 2.419 | 43.566 | 11.434 | 88.2 | 13.0 | 12.9 | 12.9 |
| 9 | 15.445 | 16.341 | 71.911 | 7.940 | 2.193 | 42.279 | 11.865 | 85.8 | 12.7 | 13.5 | 12.4 |
| 12 | 15.596 | 19.713 | 82.974 | 6.775 | 2.193 | 42.914 | 11.850 | 87.7 | 12.7 | 13.0 | 13.2 |

## Interpretation

- NPU utilization is readable under sudo and `npu_available_ratio=1.00` for all runs.
- In the current full pipeline, all three NPU cores average only about 13% utilization.
- CPU utilization is high, around 86-91%.
- OpenCV AVI writing costs about 42-44 ms/frame and dominates the end-to-end frame rate.
- Increasing threads from 3 to 12 does not increase full-pipeline FPS and does not raise average NPU utilization.

## Conclusion

The measured full-pipeline workload is CPU/output-path limited, not NPU limited. To evaluate a resume claim like `12 threads, 150+ FPS, NPU utilization >=95%`, the benchmark must be changed to `infer-only` or `rknn-only` so video writing, MPP encoding, drawing, and excessive logs do not dominate the measurement.
