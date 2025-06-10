#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpp.h"
#include "rtmp.h"

typedef struct {
    MppContext *mpp_ctx;
    SpsHeader sps_header;
    int is_initialized;
} StreamerContext;

static StreamerContext g_streamer_ctx = {0};

int init_streamer(int width, int height, int fps, int bitrate, const char *rtmp_url) {
    if (g_streamer_ctx.is_initialized) {
        return 0;
    }

    // 初始化MPP编码器
    g_streamer_ctx.mpp_ctx = alloc_mpp_context();
    if (!g_streamer_ctx.mpp_ctx) {
        printf("Failed to allocate MPP context\n");
        return -1;
    }

    // 配置MPP编码器参数
    g_streamer_ctx.mpp_ctx->width = width;
    g_streamer_ctx.mpp_ctx->height = height;

    g_streamer_ctx.mpp_ctx->fps_in_flex = 0;  // 使用固定帧率模式
    g_streamer_ctx.mpp_ctx->fps_in_num = fps;
    g_streamer_ctx.mpp_ctx->fps_in_den = 1;

    g_streamer_ctx.mpp_ctx->fps_in_flex = 0;  // 使用固定帧率模式
    g_streamer_ctx.mpp_ctx->fps_out_num = fps;
    g_streamer_ctx.mpp_ctx->fps_out_den = 1;
    
    g_streamer_ctx.mpp_ctx->bps = bitrate;
    g_streamer_ctx.mpp_ctx->gop_len = fps * 2;  // GOP长度为帧率的2倍
    g_streamer_ctx.mpp_ctx->write_frame = write_frame;
    g_streamer_ctx.mpp_ctx->type = MPP_VIDEO_CodingAVC;  // 设置H.264编码
    g_streamer_ctx.mpp_ctx->fmt = MPP_FMT_YUV420SP;  // 设置YUV420P格式
    g_streamer_ctx.mpp_ctx->rc_mode = MPP_ENC_RC_MODE_CBR;

    printf("初始化流媒体推送器...\n");
    printf("视频参数: %dx%d, %d fps, %d bps\n", width, height, fps, bitrate);
    printf("RTMP地址: %s\n", rtmp_url);
    
    // 初始化MPP
    int ret = g_streamer_ctx.mpp_ctx->init_mpp(g_streamer_ctx.mpp_ctx);
    if(ret != 0)
    {
        printf("mpp init fail!\n");
    }
    else
    {
        printf("mpp init success!\n");
    }

    // 获取SPS/PPS信息
    if (!g_streamer_ctx.mpp_ctx->get_header(g_streamer_ctx.mpp_ctx, &g_streamer_ctx.sps_header)) {
        printf("Failed to get SPS/PPS header\n");
        return -1;
    }

    printf("初始化RTMP...\n");
    // 初始化RTMP
    if (init_rtmp_streamer((char*)rtmp_url, g_streamer_ctx.sps_header.data, g_streamer_ctx.sps_header.size) < 0) {
        printf("Failed to initialize RTMP streamer\n");
        return -1;
    }
    printf("初始化RTMP成功\n");

    g_streamer_ctx.is_initialized = 1;
    return 0;
}

int process_frame(uint8_t *frame_data, int frame_size) {
    if (!g_streamer_ctx.is_initialized || !g_streamer_ctx.mpp_ctx) {
        return -1;
    }

    // 打印输入帧信息
    printf("输入帧信息: 大小=%d bytes, 格式=nv12\n", frame_size);
    

    // 使用MPP进行编码
    if (!g_streamer_ctx.mpp_ctx->process_image(frame_data, frame_size, g_streamer_ctx.mpp_ctx)) {
        printf("Failed to process frame\n");
        return -1;
    }

    return 0;
}

void close_streamer() {
    if (g_streamer_ctx.mpp_ctx) {
        g_streamer_ctx.mpp_ctx->close(g_streamer_ctx.mpp_ctx);
        g_streamer_ctx.mpp_ctx = NULL;
    }
    
    if (g_streamer_ctx.sps_header.data) {
        free(g_streamer_ctx.sps_header.data);
        g_streamer_ctx.sps_header.data = NULL;
    }
    
    g_streamer_ctx.is_initialized = 0;
} 