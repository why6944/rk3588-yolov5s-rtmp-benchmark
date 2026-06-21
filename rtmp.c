#include <libavformat/avformat.h>
#include <libavutil/mathematics.h>
#include <libavutil/time.h>
#include <libavutil/mem.h>
#include <libavcodec/avcodec.h>
#include <string.h>
#include <stdlib.h>
#include "rtmp.h"
#include <unistd.h>

static AVPacket pkt;
static AVFormatContext *ofmt_ctx = NULL;
static AVStream *video_stream = NULL;
static int frame_index = 0;
static int64_t open_deadline_us = 0;
static int stream_fps = 25;

static int rtmp_open_interrupt_cb(void *opaque)
{
        (void)opaque;
        return open_deadline_us > 0 && av_gettime_relative() > open_deadline_us;
}

int write_frame(uint8_t *data, int size)
{
        if (!ofmt_ctx || !video_stream) return -1;

        av_init_packet(&pkt);
        pkt.size = size;
        pkt.data = data;
        pkt.stream_index = video_stream->index;
        pkt.pts = av_rescale_q(frame_index, (AVRational){1, stream_fps}, video_stream->time_base);
        pkt.dts = pkt.pts;
        pkt.duration = av_rescale_q(1, (AVRational){1, stream_fps}, video_stream->time_base);
        pkt.flags = AV_PKT_FLAG_KEY;
        frame_index++;

        if (av_write_frame(ofmt_ctx, &pkt) < 0) {
                printf("Error muxing packet\n");
                return -1;
        }
        return 0;
}

int init_rtmp_streamer(char *stream, RtmpContext *config)
{
        printf("开始初始化RTMP流媒体推送...\n");
        printf("RTMP地址: %s\n", stream);
        printf("SPS/PPS数据大小: %d bytes\n", config->extradata_size);

        int ret;
        int url_opened = 0;
        stream_fps = config->fps > 0 ? config->fps : 25;
        frame_index = 0;

        if ((ret = avformat_network_init()) < 0) {
                fprintf(stderr, "avformat_network_init failed!\n");
                return -1;
        }

        printf("网络初始化成功\n");

        avformat_alloc_output_context2(&ofmt_ctx, NULL, "flv", stream);
        if (!ofmt_ctx) {
                fprintf(stderr, "Could not create output context\n");
                return -1;
        }
        ofmt_ctx->interrupt_callback.callback = rtmp_open_interrupt_cb;
        ofmt_ctx->interrupt_callback.opaque = NULL;

        printf("创建输出上下文成功\n");

        video_stream = avformat_new_stream(ofmt_ctx, NULL);
        if (!video_stream) {
                printf("Failed allocating output stream!\n");
                goto end;
        }

        video_stream->time_base = (AVRational){1, 1000};
        video_stream->avg_frame_rate = (AVRational){stream_fps, 1};
        video_stream->r_frame_rate = (AVRational){stream_fps, 1};

        AVCodecParameters *par = video_stream->codecpar;
        if (!par) {
                printf("Warning: stream codecpar is NULL, allocating manually...\n");
                par = avcodec_parameters_alloc();
                if (!par) {
                        printf("Error: manual codecpar allocation failed\n");
                        goto end;
                }
                video_stream->codecpar = par;
        }

        par->codec_type = AVMEDIA_TYPE_VIDEO;
        par->codec_id = config->codec_id;
        par->codec_tag = 0;
        par->format = config->pix_fmt;
        par->width = config->width;
        par->height = config->height;
        par->profile = config->profile;
        par->level = config->level;

        if (config->extradata && config->extradata_size > 0) {
                par->extradata = av_mallocz(config->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
                if (!par->extradata) {
                        printf("Failed to allocate codec extradata\n");
                        goto end;
                }
                memcpy(par->extradata, config->extradata, config->extradata_size);
                par->extradata_size = config->extradata_size;
        }

        printf("  输出格式: %s\n", ofmt_ctx->oformat->name);
        printf("  codec_type=%d codec_id=%d width=%d height=%d\n",
               par->codec_type, par->codec_id, par->width, par->height);

        printf("打开RTMP URL %s\n", stream);
        if (!(ofmt_ctx->oformat->flags & AVFMT_NOFILE)) {
                int retry_count = 0;
                const int max_retries = 3;
                const int64_t open_timeout_us = 3000000;
                ret = -1;
                while (retry_count < max_retries) {
                        open_deadline_us = av_gettime_relative() + open_timeout_us;
                        ret = avio_open2(&ofmt_ctx->pb, stream, AVIO_FLAG_WRITE, &ofmt_ctx->interrupt_callback, NULL);
                        open_deadline_us = 0;
                        if (ret >= 0) {
                                url_opened = 1;
                                break;
                        }
                        ofmt_ctx->pb = NULL;
                        retry_count++;
                        if (retry_count < max_retries) {
                                printf("无法连接到RTMP服务器, 1秒后重试... (第%d/%d次)\n", retry_count, max_retries);
                                sleep(1);
                        }
                }
                if (ret < 0) {
                        printf("无法连接到RTMP服务器，已达到最大重试次数%d\n", max_retries);
                        goto end;
                }
        }
        printf("打开输出URL成功\n");

        ret = avformat_write_header(ofmt_ctx, NULL);
        if (ret < 0) {
                printf("Error occurred when opening output URL, ret=%d\n", ret);
                goto end;
        }
        printf("写入文件头成功\n");
        printf("RTMP流媒体推送初始化完成\n");
        return 0;

end:
        fflush(stdout);
        (void)url_opened;
        ofmt_ctx = NULL;
        video_stream = NULL;
        printf("RTMP初始化失败，已清理资源\n");
        return -1;
}
