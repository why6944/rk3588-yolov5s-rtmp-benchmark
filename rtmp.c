#include <libavformat/avformat.h>
#include <libavutil/mathematics.h>
#include <libavutil/time.h>
#include <libavcodec/avcodec.h>
#include <string.h>
#include <stdlib.h>
#include "rtmp.h"
#include <unistd.h>  // 用于sleep函数
                
static AVPacket pkt;
static AVFormatContext *ofmt_ctx = NULL;
static int frame_index=0;

int write_frame(uint8_t*data,int size)
{
        static int64_t pts = 0;
        static int64_t dts = 0;
        static const int64_t frame_duration = 1000/30;  // 30fps对应的帧间隔

        // 打印编码后的帧信息
        printf("RTMP发送帧: 大小=%d bytes, PTS=%ld, DTS=%ld\n", 
               size, pts, dts);

        pkt.size = size;
        pkt.data = data;
        pkt.flags = 0x01;
        pkt.stream_index = 0;
        pkt.pts = pts;
        pkt.dts = dts;
        pkt.duration = frame_duration;

        pts += frame_duration;
        dts += frame_duration;

        if (av_write_frame(ofmt_ctx, &pkt) < 0) {
                printf("Error muxing packet\n");
                return -1;
        }
        return 0;
}

int init_rtmp_streamer(char* stream,uint8_t *data,uint32_t size)
{       
        printf("开始初始化RTMP流媒体推送...\n");
        printf("RTMP地址: %s\n", stream);
        printf("SPS/PPS数据大小: %d bytes\n", size);

        int ret;
        if((ret = avformat_network_init()) < 0)
        {
                fprintf(stderr, "avformat_network_init failed!");
                return -1;
        }

        printf("网络初始化成功\n");
        
        // 使用FLV格式 (RTMP通常封装为FLV)
        avformat_alloc_output_context2(&ofmt_ctx,NULL,"flv",stream);
        if(!ofmt_ctx)
        {
                fprintf(stderr, "Could not create output context\n");
                return -1;
        }

        printf("创建输出上下文成功\n");

        AVStream *out_stream = avformat_new_stream(ofmt_ctx,NULL);
        if(! out_stream)
        {
                printf("Failed allocating output stream!\n");
                goto end;
        }

        out_stream->time_base = av_make_q(1, 30);

        // 创建编码器上下文
        AVCodecContext *o_codec_ctx = avcodec_alloc_context3(NULL);
        if (!o_codec_ctx) {
                printf("Failed to allocate codec context\n");
                goto end;
        }

        // 设置编码器参数
        o_codec_ctx->codec_id = AV_CODEC_ID_H264;
        o_codec_ctx->codec_type = AVMEDIA_TYPE_VIDEO;
        o_codec_ctx->codec_tag = 0;
        o_codec_ctx->pix_fmt = AV_PIX_FMT_NV12;  // 修改为YUV420P格式
        o_codec_ctx->width = 1920;
        o_codec_ctx->height = 1080;
        o_codec_ctx->time_base = av_make_q(1, 25);  // 设置时间基准
        o_codec_ctx->framerate = av_make_q(25, 1);  // 设置帧率
        o_codec_ctx->gop_size = 50;  // 设置GOP大小
        o_codec_ctx->max_b_frames = 0;  // 禁用B帧
        o_codec_ctx->profile = FF_PROFILE_H264_HIGH;  // 设置H.264 profile
        o_codec_ctx->level = 31;  // 设置H.264 level
        o_codec_ctx->extradata = data;
        o_codec_ctx->extradata_size = size;

        printf("  输出格式: %s\n", ofmt_ctx->oformat->name);
        printf("  帧率: %d/%d\n", o_codec_ctx->framerate.num, o_codec_ctx->framerate.den);

        printf("设置编码器参数: 分辨率 %dx%d\n", o_codec_ctx->width, o_codec_ctx->height);
        
        if (!out_stream->codecpar) {
                printf("Error: out_stream->codecpar is NULL\n");
                goto end;
        }

        printf("开始复制编码器参数到流中...\n");
        ret = avcodec_parameters_from_context(out_stream->codecpar, o_codec_ctx);
        if (ret < 0) {
                printf("Failed to copy codec parameters from encoder context.\n");
                goto end;
        }
        else
        {
                printf("ok to copy codec parameters from encoder context.\n");
        }
        out_stream->codecpar->codec_tag = 0; // some FLV encoders want this to be 0

        // av_dump_format(ofmt_ctx,0,stream,1);

        printf("打开RTMP URL %s\n", stream);
        // 打开输出URL（Open output URL）
        if (!(ofmt_ctx->oformat->flags & AVFMT_NOFILE)) {
                int retry_count = 0;
                while (1) {
                        ret = avio_open(&ofmt_ctx->pb, stream, AVIO_FLAG_WRITE);
                        if (ret >= 0) {
                                break;
                        }
                        retry_count++;
                        printf("无法连接到RTMP服务器 '%s'，5秒后重试... (第%d次尝试)\n", stream, retry_count);
                        sleep(5);  // 等待5秒
                }
        }
        printf("打开输出URL成功\n");
        //写文件头（Write file header）
        ret = avformat_write_header(ofmt_ctx, NULL);
        if (ret < 0) {
                printf( "Error occurred when opening output URL\n");
                goto end;
        }
        printf("写入文件头成功\n");
        // free(data);
        // 清理编码器上下文
        avcodec_free_context(&o_codec_ctx);

        printf("创建输出流成功\n");
        printf("创建编码器上下文成功\n");
        printf("复制编码器参数到流中成功\n");
        printf("格式信息打印完成\n");
        printf("RTMP流媒体推送初始化完成\n");
        printf("RTMP流媒体推送器关键参数:\n");

        return 0;
        end:
        if (ofmt_ctx && !(ofmt_ctx->oformat->flags & AVFMT_NOFILE))
                avio_close(ofmt_ctx->pb);
        avformat_free_context(ofmt_ctx);
        if (ret < 0 && ret != AVERROR_EOF) {
                printf( "Error occurred.\n");
                printf("初始化失败，开始清理资源...\n");
                return -1;
        }
}
