#include "post_process_common.h"
#include <cmath>

int process_auto(int8_t *input, float *anchor, int grid_h, int grid_w,
                  int model_height, int model_width, int stride,
                  std::vector<float> &boxes, std::vector<float> &objProbs,
                  std::vector<int> &classID, float box_threshold,
                  int32_t zp, float scale) {
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    float thres = unsigmoid_cmn(box_threshold);
    int8_t thres_i8 = qnt_f32_to_int8_cmn(thres, zp, scale);

    for (int a = 0; a < 3; a++) {
        for (int i = 0; i < grid_h; i++) {
            for (int j = 0; j < grid_w; j++) {
                int8_t box_anchor_conf = input[(a * BOX_NUM_SIZE + 4) * grid_len + i * grid_w + j];
                if (box_anchor_conf > thres_i8) {
                    int box_offt = (a * BOX_NUM_SIZE) * grid_len + i * grid_w + j;
                    int8_t *box_p = input + box_offt;

                    float bx = deqnt_int8_to_f32_cmn(*box_p, zp, scale); float box_x = sigmoid_cmn(bx) * 2 - 0.5;
                    float by = deqnt_int8_to_f32_cmn(*(box_p + 1 * grid_len), zp, scale); float box_y = sigmoid_cmn(by) * 2 - 0.5;
                    float bw = deqnt_int8_to_f32_cmn(*(box_p + 2 * grid_len), zp, scale); float box_w = sigmoid_cmn(bw) * 2.0;
                    float bh = deqnt_int8_to_f32_cmn(*(box_p + 3 * grid_len), zp, scale); float box_h = sigmoid_cmn(bh) * 2.0;

                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = box_w * box_w * (float)anchor[a * 2];
                    box_h = box_h * box_h * (float)anchor[a * 2 + 1];
                    box_x -= box_w / 2.0f;
                    box_y -= box_h / 2.0f;

                    boxes.emplace_back(box_x);
                    boxes.emplace_back(box_y);
                    boxes.emplace_back(box_w);
                    boxes.emplace_back(box_h);

                    int8_t maxClassProb = *(box_p + 5 * grid_len);
                    int maxClassId = 0;
                    for (int k = 1; k < OBJ_CLASS_NUM; k++) {
                        int8_t prob = *(box_p + (5 + k) * grid_len);
                        if (prob > maxClassProb) { maxClassProb = prob; maxClassId = k; }
                    }
                    objProbs.emplace_back(sigmoid_cmn(deqnt_int8_to_f32_cmn(maxClassProb, zp, scale)));
                    classID.emplace_back(maxClassId);
                    validCount++;
                }
            }
        }
    }
    return validCount;
}
