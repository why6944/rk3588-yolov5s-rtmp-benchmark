#include <arm_neon.h>
#include "post_process.h"
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <algorithm>
#include <mutex>
#include <chrono>
#include <set>

using namespace std;
extern int g_skip_sigmoid;

static float anchor0[6] = {10, 13, 16, 30, 33, 23};
static float anchor1[6] = {30, 61, 62, 45, 59, 119};
static float anchor2[6] = {116, 90, 156, 198, 373, 326};

struct ProbArray { float conf; int index; };
static vector<string> labels;

static float sigmoid(float x) { return 1 / (1 + expf(-x)); }
static float unsigmoid(float y) { return -1.0f * logf(1.0f / y - 1); }

static int sort_descending(vector<ProbArray>& p_arr) {
    sort(p_arr.begin(), p_arr.end(), [](const ProbArray& a, const ProbArray& b) { return a.conf > b.conf; });
    return 0;
}

static float calculateIOU(float xmin0, float ymin0, float xmax0, float ymax0,
                          float xmin1, float ymin1, float xmax1, float ymax1) {
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

// NMS with BOTH bugs fixed:
// 1. classID[indexArray[i]] instead of classID[i]
// 2. indexArray[j] = -1 instead of indexArray[i] = -1
static int nms(int validCount, vector<float> &boxes, vector<int> &classID,
                vector<int>& indexArray, int currentClass, float nms_threshold) {
    for(int i = 0;i <validCount; i++) {
        if(indexArray[i] == -1 || classID[indexArray[i]] != currentClass) continue;
        int n = indexArray[i];
        for(int j = i+1; j < validCount; j++) {
            int m = indexArray[j];
            if(m == -1 || classID[m] != currentClass) continue;
            float xmin0 = boxes[n*4], ymin0 = boxes[n*4+1], xmax0 = boxes[n*4+2] + xmin0, ymax0 = boxes[n*4+3] + ymin0;
            float xmin1 = boxes[m*4], ymin1 = boxes[m*4+1], xmax1 = boxes[m*4+2] + xmin1, ymax1 = boxes[m*4+3] + ymin1;
            float iou = calculateIOU(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);
            if(iou > nms_threshold) indexArray[j] = -1;  // suppress lower-confidence box (j), not i
        }
    }
    return 0;
}

static int readLines(const char * LablePath, vector<string> &lable_vector, int maxLines) {
    ifstream file(LablePath);
    if (!file.is_open()) return 0;
    lable_vector.clear();
    string line;
    while (lable_vector.size() < static_cast<size_t>(maxLines) && getline(file, line))
        lable_vector.emplace_back(line);
    return lable_vector.size();
}

static int LoadLableName(const char * filepath, vector<string> &lable_vector, int num_labels) {
    return readLines(filepath, lable_vector, num_labels);
}

static float deqnt_int8_to_f32(int int_num, int32_t zp, float scale) { return (float)(int_num - zp) * scale; }

inline static int32_t __limit_num(float val, float min, float max) {
    float f = val <= min ? min : (val >= max ? max : val);
    return static_cast<int32_t>(f);
}

static int8_t qnt_f32_to_int8(float float_num, int32_t zp, float scale) {
    float float_qnt_num = (float_num / scale) + zp;
    int8_t int_num = static_cast<int8_t>(__limit_num(float_qnt_num, -128, 127));
    return int_num;
}

// NEON-accelerated process(): scan objectness in groups of 16 int8_t
static int process(int8_t *input, float *anchor, int grid_h, int grid_w, int model_height, int model_width, int stride,
            vector<float> &boxes, vector<float> &objProbs, vector<int> &classID, float box_threshold, int32_t zp, float scale) {
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    float box_thresh_val = g_skip_sigmoid ? box_threshold : unsigmoid(box_threshold);
    int8_t box_int8 = qnt_f32_to_int8(box_thresh_val, zp, scale);
    int8x16_t vthresh = vdupq_n_s8(box_int8);

    for (int a = 0; a < 3; a++) {
        int8_t *obj_ptr = input + (a * BOX_NUM_SIZE + 4) * grid_len;
        int groups = grid_len / 16;

        // NEON fast scan: 16 cells at a time
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

                float bx = deqnt_int8_to_f32(*box_p, zp,scale); if (!g_skip_sigmoid) bx = sigmoid(bx); float box_x = bx * 2 - 0.5;
                float by = deqnt_int8_to_f32(*(box_p + 1 * grid_len), zp,scale); if (!g_skip_sigmoid) by = sigmoid(by); float box_y = by * 2 - 0.5;
                float bw = deqnt_int8_to_f32(*(box_p + 2 * grid_len), zp,scale); if (!g_skip_sigmoid) bw = sigmoid(bw); float box_w = bw * 2.0;
                float bh = deqnt_int8_to_f32(*(box_p + 3 * grid_len), zp,scale); if (!g_skip_sigmoid) bh = sigmoid(bh); float box_h = bh * 2.0;

                box_x = (box_x + j) * (float)stride;
                box_y = (box_y + i) * (float)stride;
                box_w = box_w * box_w * (float)anchor[a*2];
                box_h = box_h * box_h * (float)anchor[a*2 + 1];
                box_x = box_x - (box_w / 2.0);
                box_y = box_y - (box_h / 2.0);

                boxes.emplace_back(box_x); boxes.emplace_back(box_y);
                boxes.emplace_back(box_w); boxes.emplace_back(box_h);

                // 80-class search (scalar - cross-stride access)
                int8_t maxClassProb = *(box_p + 5 * grid_len);
                int maxClassId = 0;
                for (int c = 1; c < OBJ_CLASS_NUM; c++) {
                    int8_t prob = *(box_p + (5 + c) * grid_len);
                    if (prob > maxClassProb) { maxClassProb = prob; maxClassId = c; }
                }
                float mp = deqnt_int8_to_f32(maxClassProb, zp, scale); if (!g_skip_sigmoid) mp = sigmoid(mp);
                objProbs.emplace_back(mp);
                classID.emplace_back(maxClassId);
                validCount++;
            }
        }

        // Scalar tail (empty for 640x640: grid_len%16==0)
        for (int t = groups * 16; t < grid_len; t++) {
            if (obj_ptr[t] <= box_int8) continue;
            int i = t / grid_w, j = t % grid_w;
            int box_offt = (a * BOX_NUM_SIZE) * grid_len + t;
            int8_t *box_p = input + box_offt;
            float bx = deqnt_int8_to_f32(*box_p, zp,scale); if (!g_skip_sigmoid) bx = sigmoid(bx); float box_x = bx * 2 - 0.5;
            float by = deqnt_int8_to_f32(*(box_p + 1 * grid_len), zp,scale); if (!g_skip_sigmoid) by = sigmoid(by); float box_y = by * 2 - 0.5;
            float bw = deqnt_int8_to_f32(*(box_p + 2 * grid_len), zp,scale); if (!g_skip_sigmoid) bw = sigmoid(bw); float box_w = bw * 2.0;
            float bh = deqnt_int8_to_f32(*(box_p + 3 * grid_len), zp,scale); if (!g_skip_sigmoid) bh = sigmoid(bh); float box_h = bh * 2.0;
            box_x = (box_x + j) * (float)stride; box_y = (box_y + i) * (float)stride;
            box_w = box_w * box_w * (float)anchor[a*2]; box_h = box_h * box_h * (float)anchor[a*2 + 1];
            box_x -= box_w / 2.0; box_y -= box_h / 2.0;
            boxes.emplace_back(box_x); boxes.emplace_back(box_y);
            boxes.emplace_back(box_w); boxes.emplace_back(box_h);
            int8_t maxClassProb = *(box_p + 5 * grid_len);
            int maxClassId = 0;
            for (int c = 1; c < OBJ_CLASS_NUM; c++) {
                int8_t prob = *(box_p + (5 + c) * grid_len);
                if (prob > maxClassProb) { maxClassProb = prob; maxClassId = c; }
            }
            float mp = deqnt_int8_to_f32(maxClassProb, zp, scale); if (!g_skip_sigmoid) mp = sigmoid(mp);
            objProbs.emplace_back(mp); classID.emplace_back(maxClassId);
            validCount++;
        }
    }
    return validCount;
}

inline static int clamp(float val, int min, int max) { return val > min? (val < max? val : max) : min; }

// Same post_process() signature as post_process.cpp — drop-in replacement
int post_process(int8_t *output0, int8_t *output1, int8_t *output2,
                 int model_height, int model_width, float box_threshold,
                 float nms_threshold, float scale_w, float scale_h,
                 std::vector<int32_t>& qnt_zps, std::vector<float>& qnt_scales,
                 detect_result_group_t &result_group, post_process_timing_t *timing)
{
    static std::once_flag labels_once;
    std::call_once(labels_once, []() {
        LoadLableName(LABLE_PATH, labels, OBJ_CLASS_NUM);
    });

    vector<float> detect_boxes;
    vector<float> objProbs;
    vector<int> classID;

    auto decode_start = std::chrono::high_resolution_clock::now();

    int stride0 = 8, grid_h0 = model_height / stride0, grid_w0 = model_width / stride0;
    int validCount0 = process(output0, anchor0, grid_h0, grid_w0, model_height, model_width, stride0,
                              detect_boxes, objProbs, classID, box_threshold, qnt_zps[0], qnt_scales[0]);
    int stride1 = 16, grid_h1 = model_height / stride1, grid_w1 = model_width / stride1;
    int validCount1 = process(output1, anchor1, grid_h1, grid_w1, model_height, model_width, stride1,
                              detect_boxes, objProbs, classID, box_threshold, qnt_zps[1], qnt_scales[1]);
    int stride2 = 32, grid_h2 = model_height / stride2, grid_w2 = model_width / stride2;
    int validCount2 = process(output2, anchor2, grid_h2, grid_w2, model_height, model_width, stride2,
                              detect_boxes, objProbs, classID, box_threshold, qnt_zps[2], qnt_scales[2]);

    int validCount = validCount0 + validCount1 + validCount2;
    auto decode_end = std::chrono::high_resolution_clock::now();
    if (timing) { timing->decode_us = std::chrono::duration_cast<std::chrono::microseconds>(decode_end - decode_start).count(); timing->valid_count = validCount; }
    if (validCount <= 0) { result_group.box_count = 0; return 0; }

    auto sort_start = std::chrono::high_resolution_clock::now();
    std::vector<ProbArray> prob_arr; prob_arr.reserve(validCount);
    for(int i = 0; i<validCount; i++) prob_arr.push_back({objProbs[i], i});
    sort_descending(prob_arr);
    std::vector<int> indexArray; indexArray.reserve(validCount);
    objProbs.clear(); objProbs.reserve(validCount);
    for (int i = 0; i < validCount; i++) { objProbs.emplace_back(prob_arr[i].conf); indexArray.emplace_back(prob_arr[i].index); }
    auto sort_end = std::chrono::high_resolution_clock::now();
    if (timing) timing->sort_us = std::chrono::duration_cast<std::chrono::microseconds>(sort_end - sort_start).count();

    auto nms_start = std::chrono::high_resolution_clock::now();
    std::set<int> class_set(begin(classID),end(classID));
    for(const int& id : class_set)
        nms(validCount, detect_boxes, classID, indexArray, id, nms_threshold);
    auto nms_end = std::chrono::high_resolution_clock::now();
    if (timing) timing->nms_us = std::chrono::duration_cast<std::chrono::microseconds>(nms_end - nms_start).count();

    auto result_start = std::chrono::high_resolution_clock::now();
    int count = 0;
    result_group.box_count = 0;
    for(int i = 0; i < validCount && count < MAX_OBJ_BOXS; i++) {
        if(indexArray[i] == -1) continue;
        int n = indexArray[i];
        float xmin = detect_boxes[4*n + 0], ymin = detect_boxes[4*n + 1];
        float xmax = detect_boxes[4*n + 2] + xmin, ymax = detect_boxes[4*n + 3] + ymin;
        float box_conf = objProbs[i];
        int id = classID[n];
        result_group.result[count].box.xmin = (int)(clamp(xmin, 0, model_width) / scale_w);
        result_group.result[count].box.ymin = (int)(clamp(ymin, 0, model_height) / scale_h);
        result_group.result[count].box.xmax = (int)(clamp(xmax, 0, model_width) / scale_w);
        result_group.result[count].box.ymax = (int)(clamp(ymax, 0, model_height) / scale_h);
        result_group.result[count].box_conf = box_conf;
        const char *label_temp = (id >= 0 && id < static_cast<int>(labels.size())) ? labels[id].c_str() : "unknown";
        strncpy(result_group.result[count].label, label_temp, sizeof(result_group.result[count].label) - 1);
        result_group.result[count].label[sizeof(result_group.result[count].label) - 1] = '\0';
        count++;
    }
    result_group.box_count = count;
    auto result_end = std::chrono::high_resolution_clock::now();
    if (timing) { timing->result_us = std::chrono::duration_cast<std::chrono::microseconds>(result_end - result_start).count(); timing->result_count = count; }
    return 0;
}
