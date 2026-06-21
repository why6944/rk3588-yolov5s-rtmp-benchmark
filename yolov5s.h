#ifndef YOLOV5S_H
#define YOLOV5S_H

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <string.h>
#include <vector>
#include "3rdparty/librknn_api/include/rknn_api.h"

#include "RgaUtils.h"
#include "im2d.h"
#include "rga.h"

#include "post_process.h"

using namespace std;
using namespace cv;

class Yolov5s
{
private:
    rknn_context context;
    unsigned int model_size;

    rknn_tensor_attr input_tensor;
    rknn_tensor_attr output_tensor;
    rknn_input_output_num num_tensors;

    vector<rknn_tensor_attr> input_attrs;
    vector<rknn_tensor_attr> output_attrs;

    unsigned char *model_data;
    unsigned char *load_model(const char* model_path, unsigned int &model_size);

    // -------------------------------------------------------
    // [修改] 预分配 RGA 缓冲区
    //
    // 原来的设计：每次调用 inference_image 都 malloc 三块内存，
    // 用 importbuffer_virtualaddr 注册给内核，用完再注销再释放。
    // 每帧都做一次，开销很大。
    //
    // 新的设计：在构造函数里分配一次，析构时才释放。
    // 每帧直接复用，不再有 malloc/free 和内核注册的开销。
    //
    // dst_buf（模型输入尺寸，构造时已知）：构造函数里直接分配。
    // src_buf / src_cvt_buf（原图尺寸，推理时才知道）：
    //     第一帧调用时分配，之后一直复用。
    // -------------------------------------------------------
    char               *rga_src_buf_      = nullptr;  // 原图数据缓冲区
    char               *rga_src_cvt_buf_  = nullptr;  // 颜色转换后的缓冲区（BGR→RGB）
    char               *rga_dst_buf_      = nullptr;  // 缩放后的缓冲区（模型输入用）
    rga_buffer_handle_t rga_src_handle_     = 0;      // 原图的 RGA 内核句柄
    rga_buffer_handle_t rga_src_cvt_handle_ = 0;      // 颜色转换后的 RGA 内核句柄
    rga_buffer_handle_t rga_dst_handle_     = 0;      // 缩放后的 RGA 内核句柄
    int                 rga_cached_img_w_   = 0;      // 已分配的原图宽（用于判断是否要重新分配）
    int                 rga_cached_img_h_   = 0;      // 已分配的原图高

    // 首帧或分辨率变化时调用，分配 src 相关缓冲区
    void ensure_src_buffers(int img_w, int img_h, int img_c);

    // 析构时统一释放所有 RGA 缓冲区，集中管理避免遗漏
    void release_rga_buffers();

public:
    Yolov5s(const char* model_path, int npu_index);
    ~Yolov5s();

    int model_height;
    int model_width;
    int model_channel;

    int img_height;
    int img_width;
    int img_channel;

    int inference_image(const Mat &origin_img, detect_result_group_t &result_group);
    int benchmark_rknn_only(const Mat &origin_img, int loops);
    int draw_result(cv::Mat &orig_img, detect_result_group_t &group);
};

#endif