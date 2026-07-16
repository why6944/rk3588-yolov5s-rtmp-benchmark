#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include "yolov5s.h"

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <image_path> [model_path]\n", argv[0]);
        return 1;
    }
    const char *img_path = argv[1];
    const char *model_path = (argc >= 3) ? argv[2] : "../model/yolov5s.rknn";

    printf("Loading model: %s\n", model_path);
    Yolov5s yolo(model_path, 0);
    printf("Model input: %dx%dx%d\n", yolo.model_width, yolo.model_height, yolo.model_channel);

    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        fprintf(stderr, "Failed to read image: %s\n", img_path);
        return 1;
    }
    printf("Image: %dx%d\n", img.cols, img.rows);

    // Run inference 30 times to dump INT8 outputs
    detect_result_group_t result;
    for (int i = 0; i < 30; i++) {
        printf("Frame %d...\n", i);
        int ret = yolo.inference_image(img, result);
        if (ret != 0) {
            fprintf(stderr, "Inference failed at frame %d\n", i);
            return 1;
        }
    }

    printf("Done. Check int8_dumps/ directory.\n");
    return 0;
}
