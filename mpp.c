#include "mpp.h"
static void mpp_close(MppContext* ctx);
static int init_mpp(MppContext *mpp_enc_data);
static _Bool write_header(MppContext *mpp_enc_data, SpsHeader *sps_header);
static _Bool process_image(uint8_t *p, int size, MppContext *mpp_enc_data);

MppContext * alloc_mpp_context()
{
        MppContext *ctx = (MppContext *)malloc(sizeof(MppContext));
        ctx->init_mpp = init_mpp;
        ctx->close = mpp_close;
        ctx->write_header = write_header;
        ctx->process_image = process_image;
        return ctx;
}

static void mpp_close(MppContext* ctx)
{
        MPP_RET ret = MPP_OK;
        ret = ctx->mpi->reset(ctx->ctx);
        if (ret)
        {
                printf("mpi->reset failed\n");
        }

        if (ctx->ctx)
        {
                mpp_destroy(ctx->ctx);
                ctx->ctx = NULL;
        }

        if (ctx->frm_buf)
        {
                mpp_buffer_put(ctx->frm_buf);
                ctx->frm_buf = NULL;
        }
        free(ctx);
    
}

//这里写死RK3588，根据实际需求更改
static int mpp_get_soc_type()
{
	return ROCKCHIP_SOC_RK3588;
}

static int init_mpp(MppContext *mpp_enc_data)
{
    MPP_RET ret = MPP_OK;
    MppPollType timeout = MPP_POLL_BLOCK;

	printf("start to init mpp...\n ");
    // 设置基本参数
    mpp_enc_data->type = MPP_VIDEO_CodingAVC;
    mpp_enc_data->fmt = MPP_FMT_YUV420SP;
    mpp_enc_data->hor_stride = MPP_ALIGN(mpp_enc_data->width, 16);
    mpp_enc_data->ver_stride = MPP_ALIGN(mpp_enc_data->height, 16);
    mpp_enc_data->frame_size = mpp_enc_data->hor_stride * mpp_enc_data->ver_stride * 3 / 2 ;


    // // 计算motion info size
    // RockchipSocType soc_type = mpp_get_soc_type();
    // RK_U32 w = mpp_enc_data->hor_stride, h = mpp_enc_data->ver_stride;
    // if (soc_type == ROCKCHIP_SOC_RK3588) {
    //     mpp_enc_data->mdinfo_size = (MPP_ALIGN(w, 64) >> 6) * (MPP_ALIGN(h, 64) >> 6) * 32;
    // } else {
    //     mpp_enc_data->mdinfo_size = (MPP_VIDEO_CodingHEVC == mpp_enc_data->type) ?
    //               (MPP_ALIGN(w, 32) >> 5) * (MPP_ALIGN(h, 32) >> 5) * 16 :
    //               (MPP_ALIGN(w, 64) >> 6) * (MPP_ALIGN(h, 16) >> 4) * 16;
    // }

    // 初始化buffer group
    ret = mpp_buffer_group_get_internal(&mpp_enc_data->buf_grp, MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CACHABLE);
    if (ret) {
        printf("failed to get mpp buffer group ret %d\n", ret);
        goto MPP_INIT_OUT;
    }
    else
    {
        printf("get mpp buffer group\n");
    }

    // 分配frame buffer
    ret = mpp_buffer_get(mpp_enc_data->buf_grp, &mpp_enc_data->frm_buf, mpp_enc_data->frame_size);
    if (ret) {
        printf("failed to get buffer for input frame ret %d\n", ret);
        goto MPP_INIT_OUT;
    }

    // 分配packet buffer
    ret = mpp_buffer_get(mpp_enc_data->buf_grp, &mpp_enc_data->pkt_buf, mpp_enc_data->frame_size);
    if (ret) {
        printf("failed to get buffer for output packet ret %d\n", ret);
        goto MPP_INIT_OUT;
    }

    // // 分配motion info buffer
    // ret = mpp_buffer_get(mpp_enc_data->buf_grp, &mpp_enc_data->md_info, mpp_enc_data->mdinfo_size);
    // if (ret) {
    //     printf("failed to get buffer for motion info ret %d\n", ret);
    //     goto MPP_INIT_OUT;
    // }

    // 创建MPP上下文
    ret = mpp_create(&mpp_enc_data->ctx, &mpp_enc_data->mpi);
    if (ret) {
        printf("mpp_create failed ret %d\n", ret);
        goto MPP_INIT_OUT;
    }

    // 设置输出超时
    ret = mpp_enc_data->mpi->control(mpp_enc_data->ctx, MPP_SET_OUTPUT_TIMEOUT, &timeout);
    if (ret) {
        printf("mpi control set output timeout failed ret %d\n", ret);
        goto MPP_INIT_OUT;
    }

    // 初始化编码器
    ret = mpp_init(mpp_enc_data->ctx, MPP_CTX_ENC, mpp_enc_data->type);
    if (ret) {
        printf("mpp_init failed ret %d\n", ret);
        goto MPP_INIT_OUT;
    }

    // 初始化编码器配置
    ret = mpp_enc_cfg_init(&mpp_enc_data->cfg);
    if (ret) {
        printf("mpp_enc_cfg_init failed ret %d\n", ret);
        goto MPP_INIT_OUT;
    }

    // 获取默认配置
    ret = mpp_enc_data->mpi->control(mpp_enc_data->ctx, MPP_ENC_GET_CFG, mpp_enc_data->cfg);
    if (ret) {
        printf("get enc cfg failed ret %d\n", ret);
        goto MPP_INIT_OUT;
    }

    // 设置编码器参数
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "prep:width", mpp_enc_data->width);
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "prep:height", mpp_enc_data->height);
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "prep:hor_stride", mpp_enc_data->hor_stride);
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "prep:ver_stride", mpp_enc_data->ver_stride);
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "prep:format", mpp_enc_data->fmt);

    // 设置码率控制参数
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:mode", mpp_enc_data->rc_mode);
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:bps_target", mpp_enc_data->bps);
    mpp_enc_cfg_set_u32(mpp_enc_data->cfg, "rc:max_reenc_times", 0);
    mpp_enc_cfg_set_u32(mpp_enc_data->cfg, "rc:super_mode", 0);
    //设置的MPP_ENC_RC_MODE_CBR模式
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:bps_max", mpp_enc_data->bps_max ? mpp_enc_data->bps_max : mpp_enc_data->bps * 17 / 16);
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:bps_min", mpp_enc_data->bps_min ? mpp_enc_data->bps_min : mpp_enc_data->bps * 15 / 16);

    // 设置帧率参数
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:fps_in_flex", mpp_enc_data->fps_in_flex);
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:fps_in_num", mpp_enc_data->fps_in_num);
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:fps_in_denom", mpp_enc_data->fps_in_den);
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:fps_out_flex", mpp_enc_data->fps_out_flex);
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:fps_out_num", mpp_enc_data->fps_out_num);
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:fps_out_denom", mpp_enc_data->fps_out_den);

    // 设置GOP参数
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:gop", mpp_enc_data->gop_len);

    // 设置编码器类型
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "codec:type", mpp_enc_data->type);
    RK_U32 constraint_set;
    /*
        * H.264 profile_idc parameter
        * 66  - Baseline profile
        * 77  - Main profile
        * 100 - High profile
        */
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "h264:profile", 100);
    /*
        * H.264 level_idc parameter
        * 10 / 11 / 12 / 13    - qcif@15fps / cif@7.5fps / cif@15fps / cif@30fps
        * 20 / 21 / 22         - cif@30fps / half-D1@@25fps / D1@12.5fps
        * 30 / 31 / 32         - D1@25fps / 720p@30fps / 720p@60fps
        * 40 / 41 / 42         - 1080p@30fps / 1080p@30fps / 1080p@60fps
        * 50 / 51 / 52         - 4K@30fps
        */
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "h264:level", 31);
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "h264:cabac_en", 1);
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "h264:cabac_idc", 0);
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "h264:trans8x8", 1);

    
    // 应用配置
    ret = mpp_enc_data->mpi->control(mpp_enc_data->ctx, MPP_ENC_SET_CFG, mpp_enc_data->cfg);
    if (ret) {
        printf("mpi control enc set cfg failed ret %d\n", ret);
        goto MPP_INIT_OUT;
    }

    // 设置SEI模式
    mpp_enc_data->sei_mode = MPP_ENC_SEI_MODE_ONE_FRAME;
    ret = mpp_enc_data->mpi->control(mpp_enc_data->ctx, MPP_ENC_SET_SEI_CFG, &mpp_enc_data->sei_mode);
    if (ret) {
        printf("mpi control enc set sei cfg failed ret %d\n", ret);
        goto MPP_INIT_OUT;
    }

    printf("MPP编码器配置参数:\n");
    printf("  分辨率: %dx%d\n", mpp_enc_data->width, mpp_enc_data->height);
    printf("  输入帧率: %d/%d\n", mpp_enc_data->fps_in_num, mpp_enc_data->fps_in_den);
    printf("  输出帧率: %d/%d\n", mpp_enc_data->fps_out_num, mpp_enc_data->fps_out_den);
    printf("  码率: %d bps\n", mpp_enc_data->bps);
    printf("  GOP长度: %d\n", mpp_enc_data->gop_len);
    printf("  编码类型: %s\n", mpp_enc_data->type == MPP_VIDEO_CodingAVC ? "H.264" : "H.265");
    printf("  像素格式: %s\n", mpp_enc_data->fmt == MPP_FMT_BGR888 ? "BGR888" : "YUV420P");
    printf("  码率控制模式: %s\n", mpp_enc_data->rc_mode == MPP_ENC_RC_MODE_CBR ? "CBR" : "VBR");
    printf("  H.264 Profile: High\n");
    printf("  H.264 Level: 31 (720p@30fps)\n");

    return 0;

MPP_INIT_OUT:
    if (mpp_enc_data->ctx) {
        mpp_destroy(mpp_enc_data->ctx);
        mpp_enc_data->ctx = NULL;
    }

    if (mpp_enc_data->frm_buf) {
        mpp_buffer_put(mpp_enc_data->frm_buf);
        mpp_enc_data->frm_buf = NULL;
    }

    if (mpp_enc_data->pkt_buf) {
        mpp_buffer_put(mpp_enc_data->pkt_buf);
        mpp_enc_data->pkt_buf = NULL;
    }

    // if (mpp_enc_data->md_info) {
    //     mpp_buffer_put(mpp_enc_data->md_info);
    //     mpp_enc_data->md_info = NULL;
    // }

    if (mpp_enc_data->cfg) {
        mpp_enc_cfg_deinit(mpp_enc_data->cfg);
        mpp_enc_data->cfg = NULL;
    }

    printf("init mpp failed!\n");
    return ret;
}

static _Bool write_header(MppContext *mpp_enc_data, SpsHeader *sps_header)
{
    MPP_RET ret = MPP_OK;
    MppPacket packet = NULL;

    printf("开始获取编码器header信息...\n");
    if (mpp_enc_data->type == MPP_VIDEO_CodingAVC || mpp_enc_data->type == MPP_VIDEO_CodingHEVC) {
        // 初始化packet
        mpp_packet_init_with_buffer(&packet, mpp_enc_data->pkt_buf);
        mpp_packet_set_length(packet, 0);

        // 获取header信息
        ret = mpp_enc_data->mpi->control(mpp_enc_data->ctx, MPP_ENC_GET_HDR_SYNC, packet);
        if (ret) {
            printf("mpi control enc get extra info failed\n");
            return 1;
        }

        // 获取并保存header数据
        if (packet) {
            void *ptr = mpp_packet_get_pos(packet);
            size_t len = mpp_packet_get_length(packet);
            
            if (sps_header) {
                sps_header->data = (uint8_t*)malloc(len);
                if (!sps_header->data) {
                    printf("failed to allocate memory for sps header\n");
                    mpp_packet_deinit(&packet);
                    return 1;
                }
                sps_header->size = len;
                memcpy(sps_header->data, ptr, len);
            }
        }

        mpp_packet_deinit(&packet);
        printf("开始获取编码器header信息成功\n");
    }

    return 1;
}

static _Bool process_image(uint8_t *p, int size, MppContext *mpp_enc_data)
{   
    MPP_RET ret = MPP_OK;
    MppFrame frame = NULL;
    MppPacket packet = NULL;
    MppMeta meta = NULL;
    void *buf = mpp_buffer_get_ptr(mpp_enc_data->frm_buf);
    RK_U32 eoi = 1;
    static int save_count = 0;  // 用于限制保存的图片数量

    // 复制输入数据到frame buffer
    memcpy(buf, p, size);

    // 初始化frame
    ret = mpp_frame_init(&frame);
    if (ret) {
        printf("mpp_frame_init failed\n");
        return 1;
    }

    // 设置frame参数
    mpp_frame_set_width(frame, mpp_enc_data->width);
    mpp_frame_set_height(frame, mpp_enc_data->height);
    mpp_frame_set_hor_stride(frame, mpp_enc_data->hor_stride);
    mpp_frame_set_ver_stride(frame, mpp_enc_data->ver_stride);
    mpp_frame_set_fmt(frame, mpp_enc_data->fmt);
    mpp_frame_set_buffer(frame, mpp_enc_data->frm_buf);
    mpp_frame_set_eos(frame, mpp_enc_data->frm_eos);

    // 设置metadata
    meta = mpp_frame_get_meta(frame);
    mpp_packet_init_with_buffer(&packet, mpp_enc_data->pkt_buf);
    mpp_packet_set_length(packet, 0);
    mpp_meta_set_packet(meta, KEY_OUTPUT_PACKET, packet);

    // 发送frame到编码器
    ret = mpp_enc_data->mpi->encode_put_frame(mpp_enc_data->ctx, frame);
    if (ret) {
        printf("mpp encode put frame failed\n");
        mpp_frame_deinit(&frame);
        return 1;
    }

    mpp_frame_deinit(&frame);

    do {
        // 获取编码后的packet
        ret = mpp_enc_data->mpi->encode_get_packet(mpp_enc_data->ctx, &packet);
        if (ret) {
            printf("mpp encode get packet failed\n");
            return 1;
        }

        if (packet) {
            void *ptr = mpp_packet_get_pos(packet);
            size_t len = mpp_packet_get_length(packet);
            char log_buf[256];
            RK_S32 log_size = sizeof(log_buf) - 1;
            RK_S32 log_len = 0;

            if (!mpp_enc_data->first_pkt)
                mpp_enc_data->first_pkt = mpp_time();

            mpp_enc_data->pkt_eos = mpp_packet_get_eos(packet);

            // 保存编码后的图片（仅保存前5帧）
            if (save_count < 5) {
                char filename[64];
                snprintf(filename, sizeof(filename), "encoded_frame_%d.h264", save_count);
                FILE *fp = fopen(filename, "wb");
                if (fp) {
                    fwrite(ptr, 1, len, fp);
                    fclose(fp);
                    printf("已保存编码后的帧到文件: %s, 大小: %zu bytes\n", filename, len);
                    save_count++;
                }
            }

            // 调用回调函数处理编码后的数据
            if (mpp_enc_data->write_frame)
                if (!(mpp_enc_data->write_frame)(ptr, len))
                    printf("------------sendok!\n");

            log_len += snprintf(log_buf + log_len, log_size - log_len,
                              "encoded frame %-4d", mpp_enc_data->frame_count);

            // 处理分区编码
            if (mpp_packet_is_partition(packet)) {
                eoi = mpp_packet_is_eoi(packet);
                log_len += snprintf(log_buf + log_len, log_size - log_len,
                                  " pkt %d", mpp_enc_data->frm_pkt_cnt);
                mpp_enc_data->frm_pkt_cnt = (eoi) ? (0) : (mpp_enc_data->frm_pkt_cnt + 1);
            }

            log_len += snprintf(log_buf + log_len, log_size - log_len,
                              " size %-7zu", len);

            // 处理metadata信息
            if (mpp_packet_has_meta(packet)) {
                meta = mpp_packet_get_meta(packet);
                RK_S32 temporal_id = 0;
                RK_S32 lt_idx = -1;
                RK_S32 avg_qp = -1;
                RK_S32 bps_rt = -1;

                if (MPP_OK == mpp_meta_get_s32(meta, KEY_TEMPORAL_ID, &temporal_id))
                    log_len += snprintf(log_buf + log_len, log_size - log_len,
                                      " tid %d", temporal_id);

                if (MPP_OK == mpp_meta_get_s32(meta, KEY_LONG_REF_IDX, &lt_idx))
                    log_len += snprintf(log_buf + log_len, log_size - log_len,
                                      " lt %d", lt_idx);

                if (MPP_OK == mpp_meta_get_s32(meta, KEY_ENC_AVERAGE_QP, &avg_qp))
                    log_len += snprintf(log_buf + log_len, log_size - log_len,
                                      " qp %2d", avg_qp);

                if (MPP_OK == mpp_meta_get_s32(meta, KEY_ENC_BPS_RT, &bps_rt))
                    log_len += snprintf(log_buf + log_len, log_size - log_len,
                                      " bps_rt %d", bps_rt);
            }

            printf("%s\n", log_buf);

            mpp_packet_deinit(&packet);
            mpp_enc_data->stream_size += len;
            mpp_enc_data->frame_count += eoi;

            if (mpp_enc_data->pkt_eos) {
                printf("found last packet\n");
            }
        }
    } while (!eoi);

    if (mpp_enc_data->frame_num > 0 && mpp_enc_data->frame_count >= mpp_enc_data->frame_num) {
        printf("encode max %d frames", mpp_enc_data->frame_count);
        return 0;
    }

    if (mpp_enc_data->frm_eos && mpp_enc_data->pkt_eos)
        return 0;

    return 1;
}