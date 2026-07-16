#!/bin/bash
cd /home/orangepi/streamer_codev5.0/Desktop/build
export LD_LIBRARY_PATH=/home/orangepi/RKNN_demo/rknn_yolov5_demo_Linux/lib:$LD_LIBRARY_PATH
echo '[int8_normal] model=../model/yolov5s_relu_INT8_Normal.rknn, runtime=2.3.2, skip-sigmoid'
exec ./app --model-path ../model/yolov5s_relu_INT8_Normal.rknn --threads 6 --warmup 50 --skip-sigmoid "$@"
