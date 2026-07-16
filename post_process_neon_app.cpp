#include <arm_neon.h>
#include <cmath>
#include <vector>
#include <stdint.h>

#define OBJ_CLASS_NUM 80
#define BOX_NUM_SIZE 85

extern int g_skip_sigmoid;

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

static float deqnt_i8_to_f32(int8_t v, int32_t zp, float s) { return (float)(v - zp) * s; }

// NEON-accelerated process() - same signature as original
int process_neon_app(int8_t *input, float *anchor, int grid_h, int grid_w,
                      int model_height, int model_width, int stride,
                      std::vector<float> &boxes, std::vector<float> &objProbs,
                      std::vector<int> &classID, float box_threshold,
                      int32_t zp, float scale, int8_t box_int8)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8x16_t vthresh = vdupq_n_s8(box_int8);

    for (int a = 0; a < 3; a++) {
        int8_t *obj_ptr = input + (a * BOX_NUM_SIZE + 4) * grid_len;
        int groups = grid_len / 16;

        for (int g = 0; g < groups; g++) {
            int8x16_t confs = vld1q_s8(obj_ptr + g * 16);
            if (vmaxvq_u8(vcgeq_s8(confs, vthresh)) == 0) continue;

            for (int k = 0; k < 16; k++) {
                int cell = g * 16 + k;
                if (obj_ptr[cell] <= box_int8) continue;
                int i = cell / grid_w;
                int j = cell % grid_w;
                int box_offt = (a * BOX_NUM_SIZE) * grid_len + cell;
                int8_t *box_p = input + box_offt;

                float bx = deqnt_i8_to_f32(*box_p, zp, scale); if (!g_skip_sigmoid) bx = sigmoid(bx); float box_x = bx * 2 - 0.5;
                float by = deqnt_i8_to_f32(*(box_p + 1 * grid_len), zp, scale); if (!g_skip_sigmoid) by = sigmoid(by); float box_y = by * 2 - 0.5;
                float bw = deqnt_i8_to_f32(*(box_p + 2 * grid_len), zp, scale); if (!g_skip_sigmoid) bw = sigmoid(bw); float box_w = bw * 2.0;
                float bh = deqnt_i8_to_f32(*(box_p + 3 * grid_len), zp, scale); if (!g_skip_sigmoid) bh = sigmoid(bh); float box_h = bh * 2.0;

                box_x = (box_x + j) * (float)stride;
                box_y = (box_y + i) * (float)stride;
                box_w = box_w * box_w * (float)anchor[a*2];
                box_h = box_h * box_h * (float)anchor[a*2 + 1];
                box_x -= box_w / 2.0;
                box_y -= box_h / 2.0;

                boxes.emplace_back(box_x); boxes.emplace_back(box_y);
                boxes.emplace_back(box_w); boxes.emplace_back(box_h);

                int8_t maxProb = *(box_p + 5 * grid_len);
                int maxId = 0;
                for (int c = 1; c < OBJ_CLASS_NUM; c++) {
                    int8_t prob = *(box_p + (5 + c) * grid_len);
                    if (prob > maxProb) { maxProb = prob; maxId = c; }
                }
                float mp = deqnt_i8_to_f32(maxProb, zp, scale); if (!g_skip_sigmoid) mp = sigmoid(mp);
                objProbs.emplace_back(mp);
                classID.emplace_back(maxId);
                validCount++;
            }
        }
    }
    return validCount;
}
