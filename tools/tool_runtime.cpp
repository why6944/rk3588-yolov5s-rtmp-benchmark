#include "post_process.h"

// Standalone inference tools do not link the application's main.cpp.
extern "C" {
int g_verbose_log = 0;
float g_box_threshold = BOX_THRESHOLD;
float g_nms_threshold = NMS_THRESHOLD;
int g_skip_sigmoid = 0;
int g_rknn_input_fd_mode = 0;
int g_rknn_output_mem_mode = 0;
int g_mpp_input_fd_mode = 0;
int g_cpu_preprocess_mode = 0;
}
