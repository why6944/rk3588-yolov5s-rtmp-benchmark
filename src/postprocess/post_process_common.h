#ifndef POST_PROCESS_COMMON_H
#define POST_PROCESS_COMMON_H
#include "post_process.h"

#include <stdint.h>
#include <vector>
#include <string>
#include <cstring>
#include <cstdio>

#define OBJ_CLASS_NUM       80
#define BOX_NUM_SIZE        85
#define MAX_OBJ_BOXS        64
#define LABEL_NAME_MAX_SIZE 32

struct ProbArray { float conf; int index; };

// Function pointer type for the process() step
typedef int (*process_func_t)(int8_t *input, float *anchor, int grid_h, int grid_w,
                               int model_height, int model_width, int stride,
                               std::vector<float> &boxes, std::vector<float> &objProbs,
                               std::vector<int> &classID, float box_threshold,
                               int32_t zp, float scale);

// Math helpers
float deqnt_int8_to_f32_cmn(int8_t int_num, int32_t zp, float scale);
int8_t qnt_f32_to_int8_cmn(float float_num, int32_t zp, float scale);
float sigmoid_cmn(float x);
float unsigmoid_cmn(float y);
void set_post_process_common_skip_sigmoid(bool skip);
bool post_process_common_skip_sigmoid();
float apply_activation_cmn(float x);
int clamp_cmn(float val, int min, int max);

// IOU + NMS
float calculate_iou_cmn(float xmin0, float ymin0, float xmax0, float ymax0,
                         float xmin1, float ymin1, float xmax1, float ymax1);
int nms_cmn(int validCount, std::vector<float> &boxes, std::vector<int> &classID,
             std::vector<int> &indexArray, int currentClass, float nms_threshold);

// Sort
int sort_descending_cmn(std::vector<struct ProbArray> &p_arr);

// Label loading
int load_label_names_cmn(const char *filepath, std::vector<std::string> &labels);

// Common orchestrator
int post_process_common(int8_t *output0, int8_t *output1, int8_t *output2,
                         int model_height, int model_width, float box_threshold,
                         float nms_threshold, float scale_w, float scale_h,
                         std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                         detect_result_group_t &result_group, post_process_timing_t *timing,
                         process_func_t process_fn);

#endif
