#include "yolov5s.h"
#include "post_process.h"
#include "benchmark_stats.h"
#include "debug_log.h"

#include <chrono>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdint.h>
#include <linux/videodev2.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>

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

static int allocate_dma_heap_fd(size_t size)
{
    const char *heap_paths[] = {
        "/dev/dma_heap/system-uncached",
        "/dev/dma_heap/cma-uncached",
        "/dev/dma_heap/system",
        "/dev/dma_heap/cma"
    };

    for(const char *path : heap_paths)
    {
        int heap_fd = open(path, O_RDWR | O_CLOEXEC);
        if(heap_fd < 0)
            continue;

        dma_heap_allocation_data_local data;
        memset(&data, 0, sizeof(data));
        data.len = size;
        data.fd_flags = O_RDWR | O_CLOEXEC;
        data.heap_flags = 0;
        if(ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &data) == 0)
        {
            close(heap_fd);
            return static_cast<int>(data.fd);
        }
        close(heap_fd);
    }
    return -1;
}

static bool int8_dump_enabled()
{
    const char *value = std::getenv("RKNN_INT8_DUMP");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

static void print_tensor_attr(rknn_tensor_attr *attr)
{
    string shape_str = attr->n_dims < 1 ? "" : to_string(attr->dims[0]);
    for(int i = 1; i < attr->n_dims; i++)
    {
        string current_str = to_string(attr->dims[i]);
        shape_str += "," + current_str;
    }
}

Yolov5s::Yolov5s(const char* model_path, int npu_index)
    : Yolov5s(model_path, npu_index, nullptr)
{
}

Yolov5s::Yolov5s(const char* model_path, int npu_index, rknn_context* shared_context)
{
    int ret = -1;
    const bool can_share_context = (shared_context != nullptr && *shared_context != 0);

    if(can_share_context)
    {
        ret = rknn_dup_context(shared_context, &this->context);
        if(ret != 0) this->context = 0;
        if(ret != 0)
            printf("rknn dup context failed! error code: %d, fallback to full model load.\n", ret);
        else
            LOG_DEBUG("yolo %d duplicated RKNN context successfully.\n", npu_index);
    }

    if(!can_share_context || ret != 0)
    {
        model_data = load_model(model_path, this->model_size);
        if(!model_data)
        {
            printf("load model failed, skip rknn init.\n");
            return;
        }

        ret = rknn_init(&this->context, model_data, this->model_size, RKNN_FLAG_PRIOR_HIGH, NULL);
        if (ret != 0)
        {
            printf("rknn init failed! error code: %d\n", ret);
            this->context = 0;
            return;
        }
        else
            LOG_DEBUG("yolo %d initialized RKNN context by full model load.\n", npu_index);
    }

    if(npu_index % 3 == 0)      { ret = rknn_set_core_mask(this->context, RKNN_NPU_CORE_0); }
    else if(npu_index % 3 == 1) { ret = rknn_set_core_mask(this->context, RKNN_NPU_CORE_1); }
    else                        { ret = rknn_set_core_mask(this->context, RKNN_NPU_CORE_2); }
    if (ret != 0)
    {
        printf("npu set failed! error code: %d\n", ret);
        return;
    }

    ret = rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &this->num_tensors, sizeof(this->num_tensors));
    if (ret != 0 || num_tensors.n_input == 0 || num_tensors.n_output == 0)
    {
        printf("rknn_query tensor count failed! error code: %d, inputs=%u, outputs=%u\n",
               ret, num_tensors.n_input, num_tensors.n_output);
        return;
    }

    input_attrs.resize(num_tensors.n_input);
    output_attrs.resize(num_tensors.n_output);

    for(int i = 0; i < num_tensors.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(context, RKNN_QUERY_INPUT_ATTR, &(this->input_attrs[i]), sizeof(this->input_attrs[i]));
        if (ret != 0)
        {
            printf("rknn_query input_attrs failed! error code: %d\n", ret);
            return;
        }
        LOG_DEBUG("输入的tensor%d属性为：\n", i);
        print_tensor_attr(&(this->input_attrs[i]));
    }

    for(int i = 0; i < num_tensors.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(context, RKNN_QUERY_OUTPUT_ATTR, &(this->output_attrs[i]), sizeof(this->output_attrs[i]));
        if (ret != 0)
        {
            printf("rknn_query output_attrs failed! error code: %d\n", ret);
            return;
        }
        print_tensor_attr(&(this->output_attrs[i]));
    }

    if(g_rknn_output_mem_mode)
        init_rknn_output_mem_buffers();

    if(input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        model_channel = input_attrs[0].dims[1];
        model_height  = input_attrs[0].dims[2];
        model_width   = input_attrs[0].dims[3];
    }
    else if(input_attrs[0].fmt == RKNN_TENSOR_NHWC)
    {
        model_height  = input_attrs[0].dims[1];
        model_width   = input_attrs[0].dims[2];
        model_channel = input_attrs[0].dims[3];
    }

    if(model_width <= 0 || model_height <= 0 || model_channel <= 0)
    {
        printf("[Yolov5s] invalid input tensor shape or format.\n");
        return;
    }

    // 预分配 dst_buf（模型输入尺寸在 rknn_query 后已确定，只分配一次）
    int dst_size = model_height * model_width * model_channel;
    rga_dst_size_ = dst_size;

    if(g_rknn_input_fd_mode && init_rknn_input_fd_buffer(dst_size))
    {
        LOG_DEBUG("[yolo %d] rknn input fd buffer enabled, size=%d bytes\n", npu_index, dst_size);
        initialized_ = true;
        return;
    }

    rga_dst_buf_ = (char *)malloc(dst_size);
    if (!rga_dst_buf_)
    {
        printf("[yolo %d] dst_buf malloc 失败！\n", npu_index);
        return;
    }
    memset(rga_dst_buf_, 0x00, dst_size);
    rga_dst_handle_ = importbuffer_virtualaddr(rga_dst_buf_, dst_size);
    if (rga_dst_handle_ == 0)
    {
        printf("[yolo %d] dst_buf importbuffer 失败！\n", npu_index);
        free(rga_dst_buf_);
        rga_dst_buf_ = nullptr;
        return;
    }
    else
        LOG_DEBUG("[yolo %d] dst_buf 预分配成功，大小=%d bytes\n", npu_index, dst_size);

    initialized_ = true;
}

rknn_context* Yolov5s::get_context_ptr()
{
    return &context;
}

bool Yolov5s::isInitialized() const
{
    return initialized_;
}

bool Yolov5s::init_rknn_input_fd_buffer(int dst_size)
{
    rga_dst_dma_fd_ = allocate_dma_heap_fd(dst_size);
    if(rga_dst_dma_fd_ < 0)
    {
        printf("[Yolov5s] dma_heap alloc failed, fallback to rknn_inputs_set copy path.\n");
        return false;
    }

    void *addr = mmap(NULL, dst_size, PROT_READ | PROT_WRITE, MAP_SHARED, rga_dst_dma_fd_, 0);
    if(addr == MAP_FAILED)
    {
        printf("[Yolov5s] dma_heap mmap failed, fallback to copy path.\n");
        close(rga_dst_dma_fd_);
        rga_dst_dma_fd_ = -1;
        return false;
    }

    rga_dst_buf_ = static_cast<char *>(addr);
    memset(rga_dst_buf_, 0x00, dst_size);
    rga_dst_handle_ = importbuffer_fd(rga_dst_dma_fd_, dst_size);
    if(rga_dst_handle_ == 0)
    {
        printf("[Yolov5s] importbuffer_fd failed, fallback to copy path.\n");
        munmap(rga_dst_buf_, dst_size);
        rga_dst_buf_ = nullptr;
        close(rga_dst_dma_fd_);
        rga_dst_dma_fd_ = -1;
        return false;
    }

    rknn_input_mem_ = rknn_create_mem_from_fd(context, rga_dst_dma_fd_, rga_dst_buf_, dst_size, 0);
    if(!rknn_input_mem_)
    {
        printf("[Yolov5s] rknn_create_mem_from_fd failed, fallback to copy path.\n");
        releasebuffer_handle(rga_dst_handle_);
        rga_dst_handle_ = 0;
        munmap(rga_dst_buf_, dst_size);
        rga_dst_buf_ = nullptr;
        close(rga_dst_dma_fd_);
        rga_dst_dma_fd_ = -1;
        return false;
    }

    rknn_input_mem_attr_ = input_attrs[0];
    rknn_input_mem_attr_.index = 0;
    rknn_input_mem_attr_.type = RKNN_TENSOR_UINT8;
    rknn_input_mem_attr_.fmt = RKNN_TENSOR_NHWC;
    rknn_input_mem_attr_.size = dst_size;
    rknn_input_mem_attr_.pass_through = 0;
    rknn_input_mem_attr_.w_stride = model_width;
    rknn_input_mem_attr_.h_stride = model_height;

    int ret = rknn_set_io_mem(context, rknn_input_mem_, &rknn_input_mem_attr_);
    if(ret != 0)
    {
        printf("[Yolov5s] rknn_set_io_mem failed: %d, fallback to copy path.\n", ret);
        rknn_destroy_mem(context, rknn_input_mem_);
        rknn_input_mem_ = nullptr;
        releasebuffer_handle(rga_dst_handle_);
        rga_dst_handle_ = 0;
        munmap(rga_dst_buf_, dst_size);
        rga_dst_buf_ = nullptr;
        close(rga_dst_dma_fd_);
        rga_dst_dma_fd_ = -1;
        return false;
    }

    rknn_input_fd_enabled_ = true;
    return true;
}

bool Yolov5s::init_rknn_output_mem_buffers()
{
    release_rknn_output_mem_buffers();
    rknn_output_prealloc_bufs_.resize(num_tensors.n_output, nullptr);
    rknn_output_prealloc_sizes_.resize(num_tensors.n_output, 0);

    for(uint32_t i = 0; i < num_tensors.n_output; ++i)
    {
        uint32_t mem_size = std::max(output_attrs[i].size, output_attrs[i].size_with_stride);
        if(mem_size == 0)
        {
            printf("[Yolov5s] output prealloc size is 0 for tensor %u, fallback to runtime output allocation.\n", i);
            release_rknn_output_mem_buffers();
            return false;
        }

        void *buf = malloc(mem_size);
        if(!buf)
        {
            printf("[Yolov5s] output prealloc malloc %u failed, fallback to runtime output allocation.\n", i);
            release_rknn_output_mem_buffers();
            return false;
        }
        memset(buf, 0, mem_size);
        rknn_output_prealloc_bufs_[i] = buf;
        rknn_output_prealloc_sizes_[i] = mem_size;
    }

    rknn_output_mem_enabled_ = true;
    LOG_DEBUG("[Yolov5s] rknn output prealloc enabled, outputs=%d\n", num_tensors.n_output);
    return true;
}

void Yolov5s::release_rknn_output_mem_buffers()
{
    for(auto *buf : rknn_output_prealloc_bufs_)
    {
        if(buf)
            free(buf);
    }
    rknn_output_prealloc_bufs_.clear();
    rknn_output_prealloc_sizes_.clear();
    rknn_output_mem_enabled_ = false;
}

Yolov5s::~Yolov5s()
{
    release_rga_buffers();
    if (context)
        rknn_destroy(context);
    free(this->model_data);
}

void Yolov5s::ensure_src_buffers(int img_w, int img_h, int img_c)
{
    if (img_w == rga_cached_img_w_ && img_h == rga_cached_img_h_)
        return;

    if (rga_src_handle_)     { releasebuffer_handle(rga_src_handle_);     rga_src_handle_     = 0; }
    if (rga_src_cvt_handle_) { releasebuffer_handle(rga_src_cvt_handle_); rga_src_cvt_handle_ = 0; }
    if (rga_src_buf_)        { free(rga_src_buf_);     rga_src_buf_     = nullptr; }
    if (rga_src_cvt_buf_)    { free(rga_src_cvt_buf_); rga_src_cvt_buf_ = nullptr; }

    int src_size     = img_h * img_w * img_c;
    rga_src_buf_     = (char *)malloc(src_size);
    rga_src_cvt_buf_ = (char *)malloc(src_size);
    if (!rga_src_buf_ || !rga_src_cvt_buf_)
    {
        printf("[Yolov5s] src_buf malloc 失败！\n");
        return;
    }

    rga_src_handle_     = importbuffer_virtualaddr(rga_src_buf_,     src_size);
    rga_src_cvt_handle_ = importbuffer_virtualaddr(rga_src_cvt_buf_, src_size);
    if (rga_src_handle_ == 0 || rga_src_cvt_handle_ == 0)
        printf("[Yolov5s] src_buf importbuffer 失败！\n");

    rga_cached_img_w_ = img_w;
    rga_cached_img_h_ = img_h;
    LOG_DEBUG("[Yolov5s] src_buf 分配完成，尺寸 %dx%dx%d\n", img_w, img_h, img_c);
}

void Yolov5s::release_rga_buffers()
{
    release_rknn_output_mem_buffers();
    if (rga_src_handle_)     { releasebuffer_handle(rga_src_handle_);     rga_src_handle_     = 0; }
    if (rga_src_cvt_handle_) { releasebuffer_handle(rga_src_cvt_handle_); rga_src_cvt_handle_ = 0; }
    if (rknn_input_mem_)     { rknn_destroy_mem(context, rknn_input_mem_); rknn_input_mem_ = nullptr; }
    if (rga_dst_handle_)     { releasebuffer_handle(rga_dst_handle_);     rga_dst_handle_     = 0; }
    if (rga_src_buf_)        { free(rga_src_buf_);     rga_src_buf_     = nullptr; }
    if (rga_src_cvt_buf_)    { free(rga_src_cvt_buf_); rga_src_cvt_buf_ = nullptr; }
    if (rga_dst_dma_fd_ >= 0)
    {
        if (rga_dst_buf_) munmap(rga_dst_buf_, rga_dst_size_);
        close(rga_dst_dma_fd_);
        rga_dst_dma_fd_ = -1;
        rga_dst_buf_ = nullptr;
    }
    else if (rga_dst_buf_)
    {
        free(rga_dst_buf_);
        rga_dst_buf_ = nullptr;
    }
    rknn_input_fd_enabled_ = false;
}

unsigned char* Yolov5s::load_model(const char* model_path, unsigned int &model_size)
{
    FILE *fp = fopen(model_path, "rb");
    if(fp == NULL) { printf("open model failed!\n"); return nullptr; }

    int ret = fseek(fp, 0, SEEK_END);
    if(ret) printf("fseek err : %d\n", ret);
    model_size = ftell(fp);

    unsigned char* data = (unsigned char*)malloc(model_size);
    ret = fseek(fp, 0, SEEK_SET);
    if(ret) printf("fseek err : %d\n", ret);
    ret = fread(data, 1, model_size, fp);
    if(ret < 0) printf("read model failed! err: %d\n", ret);

    fclose(fp);
    return data;
}


int Yolov5s::inference_frame(FrameData &frame_data, detect_result_group_t &result_group)
{
    if(frame_data.hasCameraBuffer() && frame_data.dmabuf->fourcc == V4L2_PIX_FMT_YUYV)
    {
        int ret = inference_camera_dmabuf_yuyv(*frame_data.dmabuf, result_group);
        if(ret == 0)
            return ret;
        LOG_DEBUG("[Yolov5s] direct camera preprocess failed, fallback to Mat path, frame=%d\n", frame_data.index);
    }

    if(!frame_data.hasMat())
        return -1;
    return inference_image(frame_data.frame, result_group);
}

int Yolov5s::inference_camera_dmabuf_yuyv(const DmaBufFrameRef &frame_ref, detect_result_group_t &result_group)
{
    if(frame_ref.rga_handle == 0 || frame_ref.width <= 0 || frame_ref.height <= 0)
        return -1;

    int ret = 0;
    float nms_threshold      = g_nms_threshold;
    float box_conf_threshold = g_box_threshold;

    this->img_channel = 3;
    this->img_width = frame_ref.width;
    this->img_height = frame_ref.height;

    int resize_height  = this->model_height;
    int resize_width   = this->model_width;

    auto start = std::chrono::high_resolution_clock::now();
    long long pre_copy_us = 0;
    long long pre_color_us = 0;
    long long pre_resize_us = 0;

    ensure_src_buffers(img_width, img_height, img_channel);
    if(rga_src_cvt_handle_ == 0 || rga_dst_handle_ == 0)
        return -1;

    rga_buffer_t src     = wrapbuffer_handle(frame_ref.rga_handle, img_width, img_height, RK_FORMAT_YUYV_422);
    rga_buffer_t src_cvt = wrapbuffer_handle(rga_src_cvt_handle_, img_width, img_height, RK_FORMAT_RGB_888);
    rga_buffer_t dst     = wrapbuffer_handle(rga_dst_handle_, resize_width, resize_height, RK_FORMAT_RGB_888);

    ret = imcheck(src, src_cvt, {}, {});
    if(ret != IM_STATUS_NOERROR)
    {
        printf("%d, camera fd imcheck error! %s\n", __LINE__, imStrError((IM_STATUS)ret));
        return -1;
    }

    auto color_start = std::chrono::high_resolution_clock::now();
    ret = imcvtcolor(src, src_cvt, RK_FORMAT_YUYV_422, RK_FORMAT_RGB_888);
    auto color_end = std::chrono::high_resolution_clock::now();
    pre_color_us = std::chrono::duration_cast<std::chrono::microseconds>(color_end - color_start).count();
    if(ret != IM_STATUS_SUCCESS)
    {
        printf("%d, camera fd cvtColor error! %s\n", __LINE__, imStrError((IM_STATUS)ret));
        return -1;
    }

    auto resize_start = std::chrono::high_resolution_clock::now();
    ret = imresize(src_cvt, dst);
    auto resize_end = std::chrono::high_resolution_clock::now();
    pre_resize_us = std::chrono::duration_cast<std::chrono::microseconds>(resize_end - resize_start).count();
    if(ret != IM_STATUS_SUCCESS)
    {
        printf("%d, camera fd resize error! %s\n", __LINE__, imStrError((IM_STATUS)ret));
        return -1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    long long preprocess_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    start = std::chrono::high_resolution_clock::now();

    int inputs_num = num_tensors.n_input;
    rknn_input inputs[inputs_num];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index        = 0;
    inputs[0].type         = RKNN_TENSOR_UINT8;
    inputs[0].size         = model_height * model_width * model_channel;
    inputs[0].pass_through = false;
    inputs[0].fmt          = RKNN_TENSOR_NHWC;
    inputs[0].buf          = rga_dst_buf_;

    auto input_set_start = std::chrono::high_resolution_clock::now();
    if(!rknn_input_fd_enabled_)
        ret = rknn_inputs_set(context, inputs_num, inputs);
    else
        ret = 0;
    auto input_set_end = std::chrono::high_resolution_clock::now();
    if(ret != 0)
        printf("rknn_inputs_set failed! error code: %d\n", ret);

    int outputs_num = num_tensors.n_output;
    std::vector<rknn_output> outputs(outputs_num);
    std::vector<void*> output_bufs(outputs_num, nullptr);

    auto run_start = std::chrono::high_resolution_clock::now();
    ret = rknn_run(context, NULL);
    auto run_end = std::chrono::high_resolution_clock::now();
    if(ret != 0)
        printf("rknn_run failed! error code: %d\n", ret);

    auto outputs_get_start = std::chrono::high_resolution_clock::now();
    memset(outputs.data(), 0, sizeof(rknn_output) * outputs_num);
    for (int i = 0; i < outputs_num; i++)
    {
        outputs[i].want_float = 0;
        outputs[i].index = i;
        if(rknn_output_mem_enabled_)
        {
            outputs[i].is_prealloc = 1;
            outputs[i].buf = rknn_output_prealloc_bufs_[i];
            outputs[i].size = rknn_output_prealloc_sizes_[i];
        }
    }
    ret = rknn_outputs_get(context, outputs_num, outputs.data(), NULL);
    for(int i = 0; i < outputs_num; i++)
        output_bufs[i] = outputs[i].buf;
    auto outputs_get_end = std::chrono::high_resolution_clock::now();
    if(ret != 0)
        printf("rknn_outputs_get/output_mem failed! error code: %d\n", ret);

    end = std::chrono::high_resolution_clock::now();
    long long input_set_us = std::chrono::duration_cast<std::chrono::microseconds>(input_set_end - input_set_start).count();
    long long rknn_run_us = std::chrono::duration_cast<std::chrono::microseconds>(run_end - run_start).count();
    long long outputs_get_us = std::chrono::duration_cast<std::chrono::microseconds>(outputs_get_end - outputs_get_start).count();
    long long rknn_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    auto post_start = std::chrono::high_resolution_clock::now();
    float scale_w = (float)model_width  / img_width;
    float scale_h = (float)model_height / img_height;

    vector<int32_t> qnt_zps;
    vector<float> qnt_scales;
    for (int i = 0; i < outputs_num; i++)
    {
        qnt_zps.emplace_back(output_attrs[i].zp);
        qnt_scales.emplace_back(output_attrs[i].scale);
    }

    post_process_timing_t post_timing;
    // ---- INT8 dump for post-processing benchmark ----
    {
        static int dump_frame_count = 0;
        if (int8_dump_enabled() && dump_frame_count < 30) {
            char fname[128];
            snprintf(fname, sizeof(fname), "int8_dumps/frame_%04d.bin", dump_frame_count);
            FILE *fp = fopen(fname, "wb");
            if (fp) {
                uint32_t sizes[3] = {
                    (uint32_t)output_attrs[0].size,
                    (uint32_t)output_attrs[1].size,
                    (uint32_t)output_attrs[2].size
                };
                int32_t zps[3] = {(int32_t)qnt_zps[0], (int32_t)qnt_zps[1], (int32_t)qnt_zps[2]};
                float scales[3] = {qnt_scales[0], qnt_scales[1], qnt_scales[2]};
                fwrite(sizes, sizeof(sizes), 1, fp);
                fwrite(zps, sizeof(zps), 1, fp);
                fwrite(scales, sizeof(scales), 1, fp);
                fwrite(output_bufs[0], 1, sizes[0], fp);
                fwrite(output_bufs[1], 1, sizes[1], fp);
                fwrite(output_bufs[2], 1, sizes[2], fp);
                fclose(fp);
                printf("[INT8_DUMP] Saved %s (%u+%u+%u bytes)\n", fname, sizes[0], sizes[1], sizes[2]);
            } else {
                printf("[INT8_DUMP] Failed to open %s\n", fname);
            }
            dump_frame_count++;
        }
    }
    post_process((int8_t *)output_bufs[0], (int8_t *)output_bufs[1], (int8_t *)output_bufs[2],
                 model_height, model_width, box_conf_threshold, nms_threshold,
                 scale_w, scale_h, qnt_zps, qnt_scales, result_group, &post_timing);
    auto post_end = std::chrono::high_resolution_clock::now();
    long long postprocess_us = std::chrono::duration_cast<std::chrono::microseconds>(post_end - post_start).count();

    rknn_outputs_release(context, outputs_num, outputs.data());
    BenchmarkStats::instance().record_inference(preprocess_us, rknn_us, postprocess_us);
    BenchmarkStats::instance().record_preprocess_detail(pre_copy_us, pre_color_us, pre_resize_us);
    BenchmarkStats::instance().record_rknn_detail(input_set_us, rknn_run_us, outputs_get_us);
    BenchmarkStats::instance().record_postprocess_detail(
        post_timing.decode_us, post_timing.sort_us, post_timing.nms_us, post_timing.result_us,
        post_timing.valid_count, post_timing.result_count);

    return 0;
}


int Yolov5s::inference_image(const Mat& orig_img, detect_result_group_t &result_group)
{
    int ret = 0;

    float nms_threshold      = g_nms_threshold;
    float box_conf_threshold = g_box_threshold;

    this->img_channel = orig_img.channels();

    int resize_height  = this->model_height;
    int resize_width   = this->model_width;
    int resize_channel = this->model_channel;

    auto start = std::chrono::high_resolution_clock::now();
    long long pre_copy_us = 0;
    long long pre_color_us = 0;
    long long pre_resize_us = 0;

    // -------------------------------------------------------
    // [修改] 统一处理 padding 和非 padding 两种情况
    //
    // 原来的流程（非 padding 情况）：
    //   bkg = orig_img.clone()        → 第一次拷贝：整帧数据（约 6MB）
    //   memcpy(src_buf, bkg.data, ...) → 第二次拷贝：同样的整帧数据
    //   共拷贝两次，第一次完全多余
    //
    // 新的流程（非 padding 情况）：
    //   memcpy(rga_src_buf_, orig_img.data, ...) → 只拷贝一次
    //
    // padding 情况因为需要补零边框，仍然需要一个临时 bkg，
    // 但把 bkg.data 直接写进 rga_src_buf_，避免二次拷贝。
    // -------------------------------------------------------

    auto copy_start = std::chrono::high_resolution_clock::now();
    if(orig_img.cols % 16 != 0 || orig_img.rows % 16 != 0)
    {
        // padding 情况：图像不是 16 的倍数，需要补零
        int bkg_width  = (orig_img.cols + 15) / 16 * 16;
        int bkg_height = (orig_img.rows + 15) / 16 * 16;

        this->img_width  = bkg_width;
        this->img_height = bkg_height;

        // 确保 src 缓冲区已分配（首帧或尺寸变化时）
        ensure_src_buffers(img_width, img_height, img_channel);

        // 先把 rga_src_buf_ 清零，再把原图拷入左上角，实现 padding
        // 不再创建 Mat bkg，直接写入预分配缓冲区，减少一次内存分配
        memset(rga_src_buf_, 0x00, img_height * img_width * img_channel);
        for(int row = 0; row < orig_img.rows; row++)
        {
            memcpy(rga_src_buf_ + row * img_width * img_channel,
                   orig_img.data + row * orig_img.cols * img_channel,
                   orig_img.cols * img_channel);
        }
    }
    else
    {
        // 非 padding 情况（绝大多数标准分辨率视频走这里）
        this->img_width  = orig_img.cols;
        this->img_height = orig_img.rows;

        // 确保 src 缓冲区已分配
        ensure_src_buffers(img_width, img_height, img_channel);

        // [修改] 直接从 orig_img.data 拷一次，不再 clone()
        // 原来：bkg = orig_img.clone()（拷贝1）; memcpy(src_buf, bkg.data)（拷贝2）
        // 现在：memcpy(rga_src_buf_, orig_img.data)（只拷贝1次）
        memcpy(rga_src_buf_, orig_img.data, img_height * img_width * img_channel);
    }
    auto copy_end = std::chrono::high_resolution_clock::now();
    pre_copy_us = std::chrono::duration_cast<std::chrono::microseconds>(copy_end - copy_start).count();

    // [修改] 删除了 memset(rga_src_cvt_buf_) 和 memset(rga_dst_buf_)
    // 原因：RGA 的 imcvtcolor 和 imresize 会完整覆盖这两块缓冲区，
    // 提前清零没有任何作用，只是白白浪费 CPU 时间（约 7MB 的 memset）

    // 用预分配的句柄包装缓冲区描述符（无系统调用，只填结构体）
    rga_buffer_t src     = wrapbuffer_handle(rga_src_handle_,     img_width,    img_height,    RK_FORMAT_BGR_888);
    rga_buffer_t src_cvt = wrapbuffer_handle(rga_src_cvt_handle_, img_width,    img_height,    RK_FORMAT_RGB_888);
    rga_buffer_t dst     = wrapbuffer_handle(rga_dst_handle_,     resize_width, resize_height, RK_FORMAT_RGB_888);

    ret = imcheck(src, dst, {}, {});
    if(ret != IM_STATUS_NOERROR)
    {
        printf("%d, imcheck error! %s\n", __LINE__, imStrError((IM_STATUS)ret));
        ret = -1;
    }

    auto color_start = std::chrono::high_resolution_clock::now();
    ret = imcvtcolor(src, src_cvt, RK_FORMAT_BGR_888, RK_FORMAT_RGB_888);
    auto color_end = std::chrono::high_resolution_clock::now();
    pre_color_us = std::chrono::duration_cast<std::chrono::microseconds>(color_end - color_start).count();
    if(ret != IM_STATUS_SUCCESS)
    {
        printf("%d, cvtColor error! %s\n", __LINE__, imStrError((IM_STATUS)ret));
        ret = -1;
    }

    auto resize_start = std::chrono::high_resolution_clock::now();
    ret = imresize(src_cvt, dst);
    auto resize_end = std::chrono::high_resolution_clock::now();
    pre_resize_us = std::chrono::duration_cast<std::chrono::microseconds>(resize_end - resize_start).count();
    if(ret != IM_STATUS_SUCCESS)
    {
        printf("%d, resize error! %s\n", __LINE__, imStrError((IM_STATUS)ret));
        ret = -1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    long long preprocess_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // ------- RKNN 推理 -------
    start = std::chrono::high_resolution_clock::now();

    int inputs_num = num_tensors.n_input;
    rknn_input inputs[inputs_num];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index       = 0;
    inputs[0].type        = RKNN_TENSOR_UINT8;
    inputs[0].size        = model_height * model_width * model_channel;
    inputs[0].pass_through = false;
    inputs[0].fmt         = RKNN_TENSOR_NHWC;
    inputs[0].buf         = rga_dst_buf_;
    auto input_set_start = std::chrono::high_resolution_clock::now();
    if(!rknn_input_fd_enabled_)
        ret = rknn_inputs_set(context, inputs_num, inputs);
    else
        ret = 0;
    auto input_set_end = std::chrono::high_resolution_clock::now();

    int outputs_num = num_tensors.n_output;
    std::vector<rknn_output> outputs(outputs_num);
    std::vector<void*> output_bufs(outputs_num, nullptr);

    auto run_start = std::chrono::high_resolution_clock::now();
    ret = rknn_run(context, NULL);
    auto run_end = std::chrono::high_resolution_clock::now();
    if(ret != 0)
        printf("rknn_run failed! error code: %d\n", ret);

    auto outputs_get_start = std::chrono::high_resolution_clock::now();
    memset(outputs.data(), 0, sizeof(rknn_output) * outputs_num);
    for (int i = 0; i < outputs_num; i++)
    {
        outputs[i].want_float = 0;
        outputs[i].index = i;
        if(rknn_output_mem_enabled_)
        {
            outputs[i].is_prealloc = 1;
            outputs[i].buf = rknn_output_prealloc_bufs_[i];
            outputs[i].size = rknn_output_prealloc_sizes_[i];
        }
    }
    ret = rknn_outputs_get(context, outputs_num, outputs.data(), NULL);
    for(int i = 0; i < outputs_num; i++)
        output_bufs[i] = outputs[i].buf;
    auto outputs_get_end = std::chrono::high_resolution_clock::now();
    if(ret != 0)
        printf("rknn_outputs_get/output_mem failed! error code: %d\n", ret);

    end = std::chrono::high_resolution_clock::now();
    long long input_set_us = std::chrono::duration_cast<std::chrono::microseconds>(input_set_end - input_set_start).count();
    long long rknn_run_us = std::chrono::duration_cast<std::chrono::microseconds>(run_end - run_start).count();
    long long outputs_get_us = std::chrono::duration_cast<std::chrono::microseconds>(outputs_get_end - outputs_get_start).count();
    long long rknn_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // ------- 后处理 -------
    auto post_start = std::chrono::high_resolution_clock::now();
    float scale_w = (float)model_width  / img_width;
    float scale_h = (float)model_height / img_height;

    vector<int32_t> qnt_zps;
    vector<float>   qnt_scales;
    for (int i = 0; i < outputs_num; i++)
    {
        qnt_zps.emplace_back(output_attrs[i].zp);
        qnt_scales.emplace_back(output_attrs[i].scale);
    }

    // ---- INT8 dump for post-processing benchmark ----
    {
        static int dump_frame_count = 0;
        if (int8_dump_enabled() && dump_frame_count < 30) {
            char fname[128];
            snprintf(fname, sizeof(fname), "int8_dumps/frame_%04d.bin", dump_frame_count);
            FILE *fp = fopen(fname, "wb");
            if (fp) {
                uint32_t sizes[3] = {
                    (uint32_t)output_attrs[0].size,
                    (uint32_t)output_attrs[1].size,
                    (uint32_t)output_attrs[2].size
                };
                int32_t zps[3] = {(int32_t)qnt_zps[0], (int32_t)qnt_zps[1], (int32_t)qnt_zps[2]};
                float scales[3] = {qnt_scales[0], qnt_scales[1], qnt_scales[2]};
                fwrite(sizes, sizeof(sizes), 1, fp);
                fwrite(zps, sizeof(zps), 1, fp);
                fwrite(scales, sizeof(scales), 1, fp);
                fwrite(output_bufs[0], 1, sizes[0], fp);
                fwrite(output_bufs[1], 1, sizes[1], fp);
                fwrite(output_bufs[2], 1, sizes[2], fp);
                fclose(fp);
                printf("[INT8_DUMP] Saved %s\n", fname);
            }
            dump_frame_count++;
        }
    }
    post_process_timing_t post_timing;
    post_process((int8_t *)output_bufs[0], (int8_t *)output_bufs[1], (int8_t *)output_bufs[2],
                 model_height, model_width, box_conf_threshold, nms_threshold,
                 scale_w, scale_h, qnt_zps, qnt_scales, result_group, &post_timing);
    auto post_end = std::chrono::high_resolution_clock::now();
    long long postprocess_us = std::chrono::duration_cast<std::chrono::microseconds>(post_end - post_start).count();

    rknn_outputs_release(context, outputs_num, outputs.data());
    BenchmarkStats::instance().record_inference(preprocess_us, rknn_us, postprocess_us);
    BenchmarkStats::instance().record_preprocess_detail(pre_copy_us, pre_color_us, pre_resize_us);
    BenchmarkStats::instance().record_rknn_detail(input_set_us, rknn_run_us, outputs_get_us);
    BenchmarkStats::instance().record_postprocess_detail(
        post_timing.decode_us, post_timing.sort_us, post_timing.nms_us, post_timing.result_us,
        post_timing.valid_count, post_timing.result_count);

    ret = 0;
    return ret;
}


int Yolov5s::benchmark_rknn_only(const Mat& orig_img, int loops)
{
    if(loops <= 0) return 0;

    // Prepare the RKNN input tensor once, then repeatedly run the NPU.
    this->img_channel = orig_img.channels();
    int resize_height  = this->model_height;
    int resize_width   = this->model_width;

    if(orig_img.cols % 16 != 0 || orig_img.rows % 16 != 0)
    {
        int bkg_width  = (orig_img.cols + 15) / 16 * 16;
        int bkg_height = (orig_img.rows + 15) / 16 * 16;

        this->img_width  = bkg_width;
        this->img_height = bkg_height;
        ensure_src_buffers(img_width, img_height, img_channel);

        memset(rga_src_buf_, 0x00, img_height * img_width * img_channel);
        for(int row = 0; row < orig_img.rows; row++)
        {
            memcpy(rga_src_buf_ + row * img_width * img_channel,
                   orig_img.data + row * orig_img.cols * img_channel,
                   orig_img.cols * img_channel);
        }
    }
    else
    {
        this->img_width  = orig_img.cols;
        this->img_height = orig_img.rows;
        ensure_src_buffers(img_width, img_height, img_channel);
        memcpy(rga_src_buf_, orig_img.data, img_height * img_width * img_channel);
    }

    rga_buffer_t src     = wrapbuffer_handle(rga_src_handle_,     img_width,    img_height,    RK_FORMAT_BGR_888);
    rga_buffer_t src_cvt = wrapbuffer_handle(rga_src_cvt_handle_, img_width,    img_height,    RK_FORMAT_RGB_888);
    rga_buffer_t dst     = wrapbuffer_handle(rga_dst_handle_,     resize_width, resize_height, RK_FORMAT_RGB_888);

    int ret = imcheck(src, dst, {}, {});
    if(ret != IM_STATUS_NOERROR)
    {
        printf("%d, imcheck error! %s\n", __LINE__, imStrError((IM_STATUS)ret));
        return -1;
    }

    ret = imcvtcolor(src, src_cvt, RK_FORMAT_BGR_888, RK_FORMAT_RGB_888);
    if(ret != IM_STATUS_SUCCESS)
    {
        printf("%d, cvtColor error! %s\n", __LINE__, imStrError((IM_STATUS)ret));
        return -1;
    }

    ret = imresize(src_cvt, dst);
    if(ret != IM_STATUS_SUCCESS)
    {
        printf("%d, resize error! %s\n", __LINE__, imStrError((IM_STATUS)ret));
        return -1;
    }

    int inputs_num = num_tensors.n_input;
    rknn_input inputs[inputs_num];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index        = 0;
    inputs[0].type         = RKNN_TENSOR_UINT8;
    inputs[0].size         = model_height * model_width * model_channel;
    inputs[0].pass_through = false;
    inputs[0].fmt          = RKNN_TENSOR_NHWC;
    inputs[0].buf          = rga_dst_buf_;

    int outputs_num = num_tensors.n_output;

    for(int i = 0; i < loops; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        if(!rknn_input_fd_enabled_)
            ret = rknn_inputs_set(context, inputs_num, inputs);
        else
            ret = 0;
        if(ret != 0) printf("rknn_inputs_set failed! error code: %d\n", ret);

        std::vector<rknn_output> outputs(outputs_num);

        ret = rknn_run(context, NULL);
        if(ret != 0) printf("rknn_run failed! error code: %d\n", ret);
        memset(outputs.data(), 0, sizeof(rknn_output) * outputs_num);
        for(int j = 0; j < outputs_num; j++)
        {
            outputs[j].want_float = 0;
            outputs[j].index = j;
            if(rknn_output_mem_enabled_)
            {
                outputs[j].is_prealloc = 1;
                outputs[j].buf = rknn_output_prealloc_bufs_[j];
                outputs[j].size = rknn_output_prealloc_sizes_[j];
            }
        }
        ret = rknn_outputs_get(context, outputs_num, outputs.data(), NULL);
        if(ret != 0) printf("rknn_outputs_get failed! error code: %d\n", ret);
        rknn_outputs_release(context, outputs_num, outputs.data());

        auto end = std::chrono::high_resolution_clock::now();
        BenchmarkStats::instance().record_inference(
            0,
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(),
            0);
    }

    return 0;
}

int Yolov5s::draw_result(cv::Mat &orig_img, detect_result_group_t& result_group)
{
    auto draw_start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < result_group.box_count; i++)
    {
        int xmin = result_group.result[i].box.xmin;
        int ymin = result_group.result[i].box.ymin;
        int xmax = result_group.result[i].box.xmax;
        int ymax = result_group.result[i].box.ymax;

        cv::rectangle(orig_img, cv::Point(xmin, ymin), cv::Point(xmax, ymax),
                      cv::Scalar(255, 0, 0, 255), 3);

        std::stringstream ss;
        ss << std::fixed << std::setprecision(2)
           << result_group.result[i].label << ":"
           << result_group.result[i].box_conf * 100 << " %";
        std::string img_label = ss.str();

        cv::putText(orig_img, img_label, cv::Point(xmin, ymin - 15),
                    FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 1, cv::LINE_8, false);
    }
    auto draw_end = std::chrono::high_resolution_clock::now();
    BenchmarkStats::instance().record_draw(
        std::chrono::duration_cast<std::chrono::microseconds>(draw_end - draw_start).count());
    return 0;
}
