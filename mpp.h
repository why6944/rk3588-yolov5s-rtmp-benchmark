#ifndef _MPP_H
#define _MPP_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <string.h>
#include <errno.h>
#include <rockchip/rk_mpi.h>

#ifdef __cplusplus
        extern "C"
        {
#endif

#define MPP_ALIGN(x, a)         (((x)+(a)-1)&~((a)-1))

//rk系列芯片枚举
typedef enum RockchipSocType_e {
    ROCKCHIP_SOC_AUTO,
    ROCKCHIP_SOC_RK3036,
    ROCKCHIP_SOC_RK3066,
    ROCKCHIP_SOC_RK3188,
    ROCKCHIP_SOC_RK3288,
    ROCKCHIP_SOC_RK312X,
    ROCKCHIP_SOC_RK3368,
    ROCKCHIP_SOC_RK3399,
    ROCKCHIP_SOC_RK3228H,
    ROCKCHIP_SOC_RK3328,
    ROCKCHIP_SOC_RK3228,
    ROCKCHIP_SOC_RK3229,
    ROCKCHIP_SOC_RV1108,
    ROCKCHIP_SOC_RV1109,
    ROCKCHIP_SOC_RV1126,
    ROCKCHIP_SOC_RK3326,
    ROCKCHIP_SOC_RK3128H,
    ROCKCHIP_SOC_PX30,
    ROCKCHIP_SOC_RK1808,
    ROCKCHIP_SOC_RK3566,
    ROCKCHIP_SOC_RK3567,
    ROCKCHIP_SOC_RK3568,
    ROCKCHIP_SOC_RK3588,
    ROCKCHIP_SOC_RK3528,
    ROCKCHIP_SOC_RK3562,
    ROCKCHIP_SOC_RK3576,
    ROCKCHIP_SOC_RV1126B,
    ROCKCHIP_SOC_BUTT,
} RockchipSocType;

typedef struct {
        uint8_t *data;
        uint32_t size;
} SpsHeader;

typedef struct {
        // base flow context
        MppCtx ctx;
        MppApi *mpi;
        RK_S32 chn;

        // global flow control flag
        RK_U32 frm_eos;
        RK_U32 pkt_eos;
        RK_U32 frm_pkt_cnt;
        RK_S32 frame_num;
        RK_S32 frame_count;
        RK_U64 stream_size;
        volatile RK_U32 loop_end;

        // encoder config set
        MppEncCfg       cfg;
        MppEncPrepCfg   prep_cfg;
        MppEncRcCfg     rc_cfg;
        MppEncCodecCfg  codec_cfg;
        MppEncSliceSplit split_cfg;
        MppEncOSDPltCfg osd_plt_cfg;
        MppEncOSDPlt    osd_plt;
        MppEncOSDData   osd_data;
        MppEncROICfg    roi_cfg;

        // input / output
        MppBufferGroup buf_grp;
        MppBuffer frm_buf;
        MppBuffer pkt_buf;
        MppBuffer md_info;
        MppEncSeiMode sei_mode;
        MppEncHeaderMode header_mode;

        // paramter for resource malloc
        RK_U32 width;
        RK_U32 height;
        RK_U32 hor_stride;
        RK_U32 ver_stride;
        MppFrameFormat fmt;
        MppCodingType type;
        RK_S32 loop_times;

        // resources
        size_t header_size;
        size_t frame_size;
        size_t mdinfo_size;
        size_t packet_size;

        // rate control runtime parameter
        RK_S32 fps_in_flex;
        RK_S32 fps_in_den;
        RK_S32 fps_in_num;
        RK_S32 fps_out_flex;
        RK_S32 fps_out_den;
        RK_S32 fps_out_num;
        RK_S32 bps;
        RK_S32 bps_max;
        RK_S32 bps_min;
        RK_S32 rc_mode;
        RK_S32 gop_mode;
        RK_S32 gop_len;
        RK_S32 vi_len;
        RK_S32 scene_mode;
        RK_S32 cu_qp_delta_depth;
        RK_S32 anti_flicker_str;
        RK_S32 atr_str_i;
        RK_S32 atr_str_p;
        RK_S32 atl_str;
        RK_S32 sao_str_i;
        RK_S32 sao_str_p;
        RK_S64 first_frm;
        RK_S64 first_pkt;

        //call back function
        int (*write_frame)(uint8_t*data,int size);
        //function pointer
        int (*init_mpp)(void *mpp_enc_data);
        _Bool (*process_image)(uint8_t *p, int size,void *mpp_enc_data);
        _Bool (*write_header)(void *mpp_enc_data,SpsHeader *sps_header);
        void (*close)(void* ctx);

} MppContext;

MppContext* alloc_mpp_context();

#ifdef __cplusplus
        }
#endif

#endif /* !_MPP_H */