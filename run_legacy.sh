#!/bin/bash
cd /home/orangepi/streamer_codev5.0/Desktop/build
export LD_LIBRARY_PATH=/home/orangepi/streamer_codev5.0/Desktop/3rdparty/librknn_api/aarch64:$LD_LIBRARY_PATH
echo '[legacy] model=../model/yolov5s.rknn, runtime=1.5.3'
exec ./app --model-path ../model/yolov5s.rknn --threads 6 --warmup 50 "$@"
