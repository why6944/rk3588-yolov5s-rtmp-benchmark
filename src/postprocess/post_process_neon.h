#ifndef POST_PROCESS_NEON_H
#define POST_PROCESS_NEON_H
#include "post_process_common.h"
int process_neon(int8_t *input, float *anchor, int grid_h, int grid_w,
                  int model_height, int model_width, int stride,
                  std::vector<float> &boxes, std::vector<float> &objProbs,
                  std::vector<int> &classID, float box_threshold,
                  int32_t zp, float scale);
#endif
