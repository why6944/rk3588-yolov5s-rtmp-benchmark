#include "post_process_common.h"
#include <cmath>
#include <algorithm>
#include <set>
#include <fstream>
#include <iostream>
#include <mutex>
#include <chrono>

float anchor0_cmn[6] = {10, 13, 16, 30, 33, 23};
float anchor1_cmn[6] = {30, 61, 62, 45, 59, 119};
float anchor2_cmn[6] = {116, 90, 156, 198, 373, 326};

float deqnt_int8_to_f32_cmn(int8_t int_num, int32_t zp, float scale) {
    return (float)(int_num - zp) * scale;
}

int8_t qnt_f32_to_int8_cmn(float float_num, int32_t zp, float scale) {
    float f = (float_num / scale) + zp;
    if (f < -128) f = -128;
    if (f > 127) f = 127;
    return (int8_t)f;
}

float sigmoid_cmn(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float unsigmoid_cmn(float y) {
    return -1.0f * logf(1.0f / y - 1);
}

int clamp_cmn(float val, int min, int max) {
    return val > min ? (val < max ? val : max) : min;
}

int sort_descending_cmn(std::vector<ProbArray> &p_arr) {
    std::sort(p_arr.begin(), p_arr.end(),
        [](const ProbArray &a, const ProbArray &b) { return a.conf > b.conf; });
    return 0;
}

float calculate_iou_cmn(float xmin0, float ymin0, float xmax0, float ymax0,
                         float xmin1, float ymin1, float xmax1, float ymax1) {
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) +
              (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

int nms_cmn(int validCount, std::vector<float> &boxes, std::vector<int> &classID,
             std::vector<int> &indexArray, int currentClass, float nms_threshold) {
    for (int i = 0; i < validCount; i++) {
        if (indexArray[i] == -1 || classID[indexArray[i]] != currentClass)
            continue;
        int n = indexArray[i];
        for (int j = i + 1; j < validCount; j++) {
            int m = indexArray[j];
            if (m == -1 || classID[m] != currentClass)
                continue;
            float xmin0 = boxes[n * 4];
            float ymin0 = boxes[n * 4 + 1];
            float xmax0 = boxes[n * 4 + 2] + xmin0;
            float ymax0 = boxes[n * 4 + 3] + ymin0;
            float xmin1 = boxes[m * 4];
            float ymin1 = boxes[m * 4 + 1];
            float xmax1 = boxes[m * 4 + 2] + xmin1;
            float ymax1 = boxes[m * 4 + 3] + ymin1;
            float iou = calculate_iou_cmn(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);
            if (iou > nms_threshold) {
                indexArray[j] = -1;  // suppress lower-confidence box (j)
            }
        }
    }
    return 0;
}

int load_label_names_cmn(const char *filepath, std::vector<std::string> &labels) {
    std::ifstream file(filepath);
    if (!file.is_open()) return 0;
    labels.clear();
    std::string line;
    while (labels.size() < OBJ_CLASS_NUM && std::getline(file, line))
        labels.emplace_back(line);
    return labels.size();
}

int post_process_common(int8_t *output0, int8_t *output1, int8_t *output2,
                         int model_height, int model_width, float box_threshold,
                         float nms_threshold, float scale_w, float scale_h,
                         std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                         detect_result_group_t &result_group, post_process_timing_t *timing,
                         process_func_t process_fn) {
    static std::vector<std::string> labels;
    static std::once_flag once;
    std::call_once(once, []() {
        load_label_names_cmn("../model/coco_80_labels_list.txt", labels);
    });

    std::vector<float> detect_boxes;
    std::vector<float> objProbs;
    std::vector<int> classID;

    auto t0 = std::chrono::high_resolution_clock::now();

    int stride0 = 8, grid_h0 = model_height / 8, grid_w0 = model_width / 8;
    int valid0 = process_fn(output0, anchor0_cmn, grid_h0, grid_w0, model_height, model_width,
                            stride0, detect_boxes, objProbs, classID, box_threshold, qnt_zps[0], qnt_scales[0]);
    int stride1 = 16, grid_h1 = model_height / 16, grid_w1 = model_width / 16;
    int valid1 = process_fn(output1, anchor1_cmn, grid_h1, grid_w1, model_height, model_width,
                            stride1, detect_boxes, objProbs, classID, box_threshold, qnt_zps[1], qnt_scales[1]);
    int stride2 = 32, grid_h2 = model_height / 32, grid_w2 = model_width / 32;
    int valid2 = process_fn(output2, anchor2_cmn, grid_h2, grid_w2, model_height, model_width,
                            stride2, detect_boxes, objProbs, classID, box_threshold, qnt_zps[2], qnt_scales[2]);

    int validCount = valid0 + valid1 + valid2;
    auto t1 = std::chrono::high_resolution_clock::now();
    if (timing) { timing->decode_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(); timing->valid_count = validCount; }
    if (validCount <= 0) { result_group.box_count = 0; return 0; }

    // Sort
    std::vector<ProbArray> prob_arr; prob_arr.reserve(validCount);
    for (int i = 0; i < validCount; i++) prob_arr.push_back({objProbs[i], i});
    sort_descending_cmn(prob_arr);
    std::vector<int> indexArray; indexArray.reserve(validCount);
    objProbs.clear(); objProbs.reserve(validCount);
    for (int i = 0; i < validCount; i++) { objProbs.push_back(prob_arr[i].conf); indexArray.push_back(prob_arr[i].index); }
    auto t2 = std::chrono::high_resolution_clock::now();
    if (timing) timing->sort_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    // NMS per class
    std::set<int> class_set(classID.begin(), classID.end());
    for (int id : class_set)
        nms_cmn(validCount, detect_boxes, classID, indexArray, id, nms_threshold);
    auto t3 = std::chrono::high_resolution_clock::now();
    if (timing) timing->nms_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    // Build results
    int count = 0;
    for (int i = 0; i < validCount && count < MAX_OBJ_BOXS; i++) {
        if (indexArray[i] == -1) continue;
        int n = indexArray[i];
        float xmin = detect_boxes[4 * n + 0], ymin = detect_boxes[4 * n + 1];
        float xmax = detect_boxes[4 * n + 2] + xmin, ymax = detect_boxes[4 * n + 3] + ymin;
        int id = classID[n];
        result_group.result[count].box.xmin = clamp_cmn(xmin, 0, model_width) / scale_w;
        result_group.result[count].box.ymin = clamp_cmn(ymin, 0, model_height) / scale_h;
        result_group.result[count].box.xmax = clamp_cmn(xmax, 0, model_width) / scale_w;
        result_group.result[count].box.ymax = clamp_cmn(ymax, 0, model_height) / scale_h;
        result_group.result[count].box_conf = objProbs[i];
        const char *lbl = (id >= 0 && id < (int)labels.size()) ? labels[id].c_str() : "unknown";
        strncpy(result_group.result[count].label, lbl, sizeof(result_group.result[count].label) - 1);
        result_group.result[count].label[sizeof(result_group.result[count].label) - 1] = '\0';
        count++;
    }
    result_group.box_count = count;
    auto t4 = std::chrono::high_resolution_clock::now();
    if (timing) { timing->result_us = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count(); timing->result_count = count; }
    return 0;
}
