#include "yolov5s.h"
#include "post_process.h"
#include "benchmark_stats.h"
#include "debug_log.h"

#include <chrono>

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
{
    int ret;
    model_data = load_model(model_path, this->model_size);

    ret = rknn_init(&this->context, model_data, this->model_size, RKNN_FLAG_PRIOR_HIGH, NULL);
    if (ret != 0)
        printf("rknn init failed! error code: %d\n", ret);
    else
        LOG_DEBUG("yolo %d初始化成功！\n", npu_index);

    if(npu_index % 4 == 0)      { ret = rknn_set_core_mask(this->context, RKNN_NPU_CORE_0); }
    else if(npu_index % 4 == 1) { ret = rknn_set_core_mask(this->context, RKNN_NPU_CORE_1); }
    else                        { ret = rknn_set_core_mask(this->context, RKNN_NPU_CORE_2); }
    if (ret != 0) printf("npu set failed! error code: %d\n", ret);

    ret = rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &this->num_tensors, sizeof(this->num_tensors));
    if (ret != 0) printf("rknn_query failed! error code: %d\n", ret);

    input_attrs.resize(num_tensors.n_input);
    output_attrs.resize(num_tensors.n_output);

    for(int i = 0; i < num_tensors.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(context, RKNN_QUERY_INPUT_ATTR, &(this->input_attrs[i]), sizeof(this->input_attrs[i]));
        if (ret != 0) printf("rknn_query input_attrs failed! error code: %d\n", ret);
        LOG_DEBUG("输入的tensor%d属性为：\n", i);
        print_tensor_attr(&(this->input_attrs[i]));
    }

    for(int i = 0; i < num_tensors.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(context, RKNN_QUERY_OUTPUT_ATTR, &(this->output_attrs[i]), sizeof(this->output_attrs[i]));
        if (ret != 0) printf("rknn_query output_attrs failed! error code: %d\n", ret);
        print_tensor_attr(&(this->output_attrs[i]));
    }

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

    // 预分配 dst_buf（模型输入尺寸在 rknn_query 后已确定，只分配一次）
    int dst_size = model_height * model_width * model_channel;
    rga_dst_buf_ = (char *)malloc(dst_size);
    if (!rga_dst_buf_)
    {
        printf("[yolo %d] dst_buf malloc 失败！\n", npu_index);
        return;
    }
    memset(rga_dst_buf_, 0x00, dst_size);
    rga_dst_handle_ = importbuffer_virtualaddr(rga_dst_buf_, dst_size);
    if (rga_dst_handle_ == 0)
        printf("[yolo %d] dst_buf importbuffer 失败！\n", npu_index);
    else
        LOG_DEBUG("[yolo %d] dst_buf 预分配成功，大小=%d bytes\n", npu_index, dst_size);
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
    if (rga_src_handle_)     { releasebuffer_handle(rga_src_handle_);     rga_src_handle_     = 0; }
    if (rga_src_cvt_handle_) { releasebuffer_handle(rga_src_cvt_handle_); rga_src_cvt_handle_ = 0; }
    if (rga_dst_handle_)     { releasebuffer_handle(rga_dst_handle_);     rga_dst_handle_     = 0; }
    if (rga_src_buf_)        { free(rga_src_buf_);     rga_src_buf_     = nullptr; }
    if (rga_src_cvt_buf_)    { free(rga_src_cvt_buf_); rga_src_cvt_buf_ = nullptr; }
    if (rga_dst_buf_)        { free(rga_dst_buf_);     rga_dst_buf_     = nullptr; }
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

    ret = imcvtcolor(src, src_cvt, RK_FORMAT_BGR_888, RK_FORMAT_RGB_888);
    if(ret != IM_STATUS_SUCCESS)
    {
        printf("%d, cvtColor error! %s\n", __LINE__, imStrError((IM_STATUS)ret));
        ret = -1;
    }

    ret = imresize(src_cvt, dst);
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
    ret = rknn_inputs_set(context, inputs_num, inputs);
    auto input_set_end = std::chrono::high_resolution_clock::now();

    int outputs_num = num_tensors.n_output;
    rknn_output outputs[outputs_num];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < outputs_num; i++)
        outputs[i].want_float = 0;

    auto run_start = std::chrono::high_resolution_clock::now();
    ret = rknn_run(context, NULL);
    auto run_end = std::chrono::high_resolution_clock::now();
    auto outputs_get_start = std::chrono::high_resolution_clock::now();
    ret = rknn_outputs_get(context, outputs_num, outputs, NULL);
    auto outputs_get_end = std::chrono::high_resolution_clock::now();

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

    post_process_timing_t post_timing;
    post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf,
                 model_height, model_width, box_conf_threshold, nms_threshold,
                 scale_w, scale_h, qnt_zps, qnt_scales, result_group, &post_timing);
    auto post_end = std::chrono::high_resolution_clock::now();
    long long postprocess_us = std::chrono::duration_cast<std::chrono::microseconds>(post_end - post_start).count();

    rknn_outputs_release(context, outputs_num, outputs);
    BenchmarkStats::instance().record_inference(preprocess_us, rknn_us, postprocess_us);
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
        ret = rknn_inputs_set(context, inputs_num, inputs);
        if(ret != 0) printf("rknn_inputs_set failed! error code: %d\n", ret);

        rknn_output outputs[outputs_num];
        memset(outputs, 0, sizeof(outputs));
        for(int j = 0; j < outputs_num; j++)
            outputs[j].want_float = 0;

        ret = rknn_run(context, NULL);
        if(ret != 0) printf("rknn_run failed! error code: %d\n", ret);
        ret = rknn_outputs_get(context, outputs_num, outputs, NULL);
        if(ret != 0) printf("rknn_outputs_get failed! error code: %d\n", ret);
        rknn_outputs_release(context, outputs_num, outputs);

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