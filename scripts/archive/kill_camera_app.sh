#!/usr/bin/env bash
# 杀掉所有占用摄像头的 app 进程，释放 /dev/video0
echo "=== Killing camera app processes ==="
ps aux | grep -E "build/app|app_neon" | grep -v grep
fuser -k /dev/video0 2>/dev/null && echo "Released /dev/video0"
pkill -9 -f "build/app" 2>/dev/null && echo "Killed app processes"
sleep 1
fuser /dev/video0 2>&1 || echo "/dev/video0 is free"
