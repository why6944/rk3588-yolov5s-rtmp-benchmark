#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <string>
#include <vector>

#include <rockchip/rk_mpi.h>
#include <rockchip/mpp_buffer.h>
#include <rockchip/mpp_frame.h>
#include <rockchip/mpp_packet.h>

#include "rga.h"
#include "im2d.h"
#include "RgaUtils.h"

#ifndef DMA_HEAP_IOC_MAGIC
#define DMA_HEAP_IOC_MAGIC 'H'
#endif

struct dma_heap_allocation_data_local {
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;
    uint64_t heap_flags;
};

#ifndef DMA_HEAP_IOCTL_ALLOC
#define DMA_HEAP_IOCTL_ALLOC _IOWR(DMA_HEAP_IOC_MAGIC, 0x0, struct dma_heap_allocation_data_local)
#endif

#ifndef MPP_ALIGN
#define MPP_ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))
#endif

static int allocate_dma_heap_fd(size_t size, std::string &heap_path) {
    const char *heap_paths[] = {
        "/dev/dma_heap/system-uncached",
        "/dev/dma_heap/cma-uncached",
        "/dev/dma_heap/system",
        "/dev/dma_heap/cma"
    };

    for (const char *path : heap_paths) {
        int heap_fd = open(path, O_RDWR | O_CLOEXEC);
        if (heap_fd < 0) continue;

        dma_heap_allocation_data_local data;
        memset(&data, 0, sizeof(data));
        data.len = size;
        data.fd_flags = O_RDWR | O_CLOEXEC;
        data.heap_flags = 0;
        if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &data) == 0) {
            close(heap_fd);
            heap_path = path;
            return static_cast<int>(data.fd);
        }
        close(heap_fd);
    }
    return -1;
}

static void fill_rgb_pattern(std::vector<uint8_t> &rgb, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            size_t off = (static_cast<size_t>(y) * width + x) * 3;
            rgb[off + 0] = static_cast<uint8_t>((x * 255) / width);
            rgb[off + 1] = static_cast<uint8_t>((y * 255) / height);
            rgb[off + 2] = static_cast<uint8_t>(((x + y) * 255) / (width + height));
        }
    }
}

static int init_encoder(MppCtx *ctx, MppApi **mpi, MppEncCfg *cfg,
                        MppBufferGroup *buf_grp, MppBuffer *pkt_buf,
                        int width, int height, int hor_stride, int ver_stride,
                        size_t frame_size, int fps, int bps) {
    MPP_RET ret = MPP_OK;
    MppPollType timeout = MPP_POLL_BLOCK;

    ret = mpp_buffer_group_get_internal(buf_grp, MPP_BUFFER_TYPE_DRM);
    if (ret) { printf("[MPP] mpp_buffer_group_get_internal failed ret=%d\n", ret); return -1; }

    ret = mpp_buffer_get(*buf_grp, pkt_buf, frame_size);
    if (ret) { printf("[MPP] pkt buffer get failed ret=%d\n", ret); return -1; }

    ret = mpp_create(ctx, mpi);
    if (ret) { printf("[MPP] mpp_create failed ret=%d\n", ret); return -1; }

    ret = (*mpi)->control(*ctx, MPP_SET_OUTPUT_TIMEOUT, &timeout);
    if (ret) { printf("[MPP] set output timeout failed ret=%d\n", ret); return -1; }

    ret = mpp_init(*ctx, MPP_CTX_ENC, MPP_VIDEO_CodingAVC);
    if (ret) { printf("[MPP] mpp_init failed ret=%d\n", ret); return -1; }

    ret = mpp_enc_cfg_init(cfg);
    if (ret) { printf("[MPP] mpp_enc_cfg_init failed ret=%d\n", ret); return -1; }

    ret = (*mpi)->control(*ctx, MPP_ENC_GET_CFG, *cfg);
    if (ret) { printf("[MPP] get cfg failed ret=%d\n", ret); return -1; }

    mpp_enc_cfg_set_s32(*cfg, "prep:width", width);
    mpp_enc_cfg_set_s32(*cfg, "prep:height", height);
    mpp_enc_cfg_set_s32(*cfg, "prep:hor_stride", hor_stride);
    mpp_enc_cfg_set_s32(*cfg, "prep:ver_stride", ver_stride);
    mpp_enc_cfg_set_s32(*cfg, "prep:format", MPP_FMT_YUV420SP);

    mpp_enc_cfg_set_s32(*cfg, "rc:mode", MPP_ENC_RC_MODE_CBR);
    mpp_enc_cfg_set_s32(*cfg, "rc:bps_target", bps);
    mpp_enc_cfg_set_s32(*cfg, "rc:bps_max", bps * 17 / 16);
    mpp_enc_cfg_set_s32(*cfg, "rc:bps_min", bps * 15 / 16);
    mpp_enc_cfg_set_s32(*cfg, "rc:fps_in_flex", 0);
    mpp_enc_cfg_set_s32(*cfg, "rc:fps_in_num", fps);
    mpp_enc_cfg_set_s32(*cfg, "rc:fps_in_denom", 1);
    mpp_enc_cfg_set_s32(*cfg, "rc:fps_out_flex", 0);
    mpp_enc_cfg_set_s32(*cfg, "rc:fps_out_num", fps);
    mpp_enc_cfg_set_s32(*cfg, "rc:fps_out_denom", 1);
    mpp_enc_cfg_set_s32(*cfg, "rc:gop", fps * 2);

    mpp_enc_cfg_set_s32(*cfg, "codec:type", MPP_VIDEO_CodingAVC);
    mpp_enc_cfg_set_s32(*cfg, "h264:profile", 100);
    mpp_enc_cfg_set_s32(*cfg, "h264:level", 31);
    mpp_enc_cfg_set_s32(*cfg, "h264:cabac_en", 1);
    mpp_enc_cfg_set_s32(*cfg, "h264:cabac_idc", 0);
    mpp_enc_cfg_set_s32(*cfg, "h264:trans8x8", 1);

    ret = (*mpi)->control(*ctx, MPP_ENC_SET_CFG, *cfg);
    if (ret) { printf("[MPP] set cfg failed ret=%d\n", ret); return -1; }

    MppEncSeiMode sei_mode = MPP_ENC_SEI_MODE_ONE_FRAME;
    ret = (*mpi)->control(*ctx, MPP_ENC_SET_SEI_CFG, &sei_mode);
    if (ret) { printf("[MPP] set sei failed ret=%d\n", ret); return -1; }

    printf("[MPP] init OK width=%d height=%d hor_stride=%d ver_stride=%d frame_size=%zu\n",
           width, height, hor_stride, ver_stride, frame_size);
    return 0;
}

static int write_header(FILE *fp, MppCtx ctx, MppApi *mpi, MppBuffer pkt_buf) {
    MppPacket packet = NULL;
    MPP_RET ret = mpp_packet_init_with_buffer(&packet, pkt_buf);
    if (ret) { printf("[MPP] header packet init failed ret=%d\n", ret); return -1; }
    mpp_packet_set_length(packet, 0);
    ret = mpi->control(ctx, MPP_ENC_GET_HDR_SYNC, packet);
    if (ret) {
        printf("[MPP] get header failed ret=%d\n", ret);
        mpp_packet_deinit(&packet);
        return -1;
    }
    void *ptr = mpp_packet_get_pos(packet);
    size_t len = mpp_packet_get_length(packet);
    if (len > 0) fwrite(ptr, 1, len, fp);
    printf("[MPP] header bytes=%zu\n", len);
    mpp_packet_deinit(&packet);
    return 0;
}

static int encode_one_frame(FILE *fp, MppCtx ctx, MppApi *mpi, MppBuffer pkt_buf,
                            MppBuffer external_frame_buf,
                            int width, int height, int hor_stride, int ver_stride) {
    MPP_RET ret = MPP_OK;
    MppFrame frame = NULL;
    MppPacket packet = NULL;

    ret = mpp_frame_init(&frame);
    if (ret) { printf("[MPP] mpp_frame_init failed ret=%d\n", ret); return -1; }

    mpp_frame_set_width(frame, width);
    mpp_frame_set_height(frame, height);
    mpp_frame_set_hor_stride(frame, hor_stride);
    mpp_frame_set_ver_stride(frame, ver_stride);
    mpp_frame_set_fmt(frame, MPP_FMT_YUV420SP);
    mpp_frame_set_buffer(frame, external_frame_buf);
    mpp_frame_set_eos(frame, 0);

    MppMeta meta = mpp_frame_get_meta(frame);
    ret = mpp_packet_init_with_buffer(&packet, pkt_buf);
    if (ret) { printf("[MPP] output packet init failed ret=%d\n", ret); mpp_frame_deinit(&frame); return -1; }
    mpp_packet_set_length(packet, 0);
    mpp_meta_set_packet(meta, KEY_OUTPUT_PACKET, packet);

    ret = mpi->encode_put_frame(ctx, frame);
    if (ret) { printf("[MPP] encode_put_frame failed ret=%d\n", ret); return -1; }
    mpp_frame_deinit(&frame);

    ret = mpi->encode_get_packet(ctx, &packet);
    if (ret) { printf("[MPP] encode_get_packet failed ret=%d\n", ret); return -1; }

    if (!packet) {
        printf("[MPP] packet is NULL\n");
        return -1;
    }

    void *ptr = mpp_packet_get_pos(packet);
    size_t len = mpp_packet_get_length(packet);
    if (len > 0) fwrite(ptr, 1, len, fp);
    printf("[MPP] encoded packet bytes=%zu eos=%d\n", len, mpp_packet_get_eos(packet));
    mpp_packet_deinit(&packet);
    return len > 0 ? 0 : -1;
}

int main(int argc, char **argv) {
    const int width = argc > 1 ? atoi(argv[1]) : 640;
    const int height = argc > 2 ? atoi(argv[2]) : 480;
    const std::string out_path = argc > 3 ? argv[3] : "../debug_records/mpp_fd_probe.h264";
    const int fps = 30;
    const int bps = 2000000;

    const int hor_stride = MPP_ALIGN(width, 16);
    const int ver_stride = MPP_ALIGN(height, 16);
    const size_t frame_size = static_cast<size_t>(MPP_ALIGN(hor_stride, 64)) * MPP_ALIGN(ver_stride, 64) * 3 / 2;
    const size_t rgb_size = static_cast<size_t>(width) * height * 3;

    printf("[Probe] RGA RGB -> NV12 DMA-BUF -> MPP external frame buffer\n");
    printf("[Probe] width=%d height=%d hor_stride=%d ver_stride=%d frame_size=%zu\n",
           width, height, hor_stride, ver_stride, frame_size);

    std::vector<uint8_t> rgb(rgb_size);
    fill_rgb_pattern(rgb, width, height);

    std::string heap_path;
    int nv12_fd = allocate_dma_heap_fd(frame_size, heap_path);
    if (nv12_fd < 0) {
        printf("[DMA] allocate dma_heap fd failed: %s\n", strerror(errno));
        return 1;
    }
    printf("[DMA] fd=%d heap=%s size=%zu\n", nv12_fd, heap_path.c_str(), frame_size);

    void *nv12_map = mmap(NULL, frame_size, PROT_READ | PROT_WRITE, MAP_SHARED, nv12_fd, 0);
    if (nv12_map == MAP_FAILED) {
        printf("[DMA] mmap failed: %s\n", strerror(errno));
        close(nv12_fd);
        return 1;
    }
    memset(nv12_map, 0, frame_size);

    rga_buffer_handle_t rgb_handle = importbuffer_virtualaddr(rgb.data(), rgb_size);
    rga_buffer_handle_t nv12_handle = importbuffer_fd(nv12_fd, frame_size);
    if (!rgb_handle || !nv12_handle) {
        printf("[RGA] import failed rgb_handle=%p nv12_handle=%p\n", rgb_handle, nv12_handle);
        return 1;
    }

    rga_buffer_t src = wrapbuffer_handle(rgb_handle, width, height, RK_FORMAT_RGB_888);
    rga_buffer_t dst = wrapbuffer_handle(nv12_handle, hor_stride, ver_stride, RK_FORMAT_YCrCb_420_SP);
    int check = imcheck(src, dst, {}, {});
    printf("[RGA] imcheck=%d %s\n", check, imStrError((IM_STATUS)check));
    if (check != IM_STATUS_NOERROR) return 1;
    int rga_ret = imcvtcolor(src, dst, RK_FORMAT_RGB_888, RK_FORMAT_YCrCb_420_SP);
    printf("[RGA] imcvtcolor ret=%d %s\n", rga_ret, imStrError((IM_STATUS)rga_ret));
    if (rga_ret != IM_STATUS_SUCCESS) return 1;

    MppBuffer imported_frame = NULL;
    MppBufferInfo info;
    memset(&info, 0, sizeof(info));
    info.type = MPP_BUFFER_TYPE_EXT_DMA;
    info.fd = nv12_fd;
    info.size = frame_size;
    MPP_RET ret = mpp_buffer_import(&imported_frame, &info);
    if (ret || !imported_frame) {
        printf("[MPP] mpp_buffer_import external DMA fd failed ret=%d buffer=%p\n", ret, imported_frame);
        return 1;
    }
    printf("[MPP] mpp_buffer_import external DMA fd OK, buffer=%p\n", imported_frame);

    MppCtx ctx = NULL;
    MppApi *mpi = NULL;
    MppEncCfg cfg = NULL;
    MppBufferGroup buf_grp = NULL;
    MppBuffer pkt_buf = NULL;

    if (init_encoder(&ctx, &mpi, &cfg, &buf_grp, &pkt_buf, width, height, hor_stride, ver_stride, frame_size, fps, bps) != 0) {
        return 1;
    }

    FILE *fp = fopen(out_path.c_str(), "wb");
    if (!fp) {
        printf("[OUT] open %s failed: %s\n", out_path.c_str(), strerror(errno));
        return 1;
    }

    int ok = 0;
    if (write_header(fp, ctx, mpi, pkt_buf) != 0) ok = -1;
    if (ok == 0 && encode_one_frame(fp, ctx, mpi, pkt_buf, imported_frame, width, height, hor_stride, ver_stride) != 0) ok = -1;
    fclose(fp);

    if (pkt_buf) mpp_buffer_put(pkt_buf);
    if (imported_frame) mpp_buffer_put(imported_frame);
    if (cfg) mpp_enc_cfg_deinit(cfg);
    if (ctx && mpi) mpi->reset(ctx);
    if (ctx) mpp_destroy(ctx);
    if (buf_grp) mpp_buffer_group_put(buf_grp);
    if (rgb_handle) releasebuffer_handle(rgb_handle);
    if (nv12_handle) releasebuffer_handle(nv12_handle);
    if (nv12_map != MAP_FAILED) munmap(nv12_map, frame_size);
    close(nv12_fd);

    if (ok == 0) {
        printf("[RESULT] mpp_external_dmabuf_input=OK output=%s\n", out_path.c_str());
        return 0;
    }
    printf("[RESULT] mpp_external_dmabuf_input=FAIL\n");
    return 1;
}
