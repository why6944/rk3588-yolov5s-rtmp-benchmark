#ifndef _STREAMER_H_
#define _STREAMER_H_

// RTMP开关宏定义：1启用RTMP推送，0禁用（仅保留MPP编码）
#ifndef ENABLE_RTMP
#define ENABLE_RTMP 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

// 初始化流媒体推送器
// width: 视频宽度
// height: 视频高度
// fps: 帧率
// bitrate: 比特率
// rtmp_url: RTMP推流地址
int init_streamer(int width, int height, int fps, int bitrate, const char *rtmp_url);

// 处理一帧图像数据（copy路径）
// frame_data: 图像数据
// frame_size: 图像数据大小
int process_frame(uint8_t *frame_data, int frame_size);

// 处理一帧外部DMA-BUF图像（fd路径）
// dma_fd: NV12 DMA-BUF fd
// frame_size: 图像数据大小
int process_frame_fd(int dma_fd, int frame_size);

// 释放当前缓存的外部DMA-BUF输入帧，必须在关闭fd前调用
void release_frame_fd();

// 关闭流媒体推送器
void close_streamer();

#ifdef __cplusplus
}
#endif

#endif /* _STREAMER_H_ */ 