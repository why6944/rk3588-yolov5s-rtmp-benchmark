#ifndef CAMERA_DMABUF_H
#define CAMERA_DMABUF_H

#include <opencv2/opencv.hpp>

#include <linux/videodev2.h>
#include <stddef.h>
#include <stdint.h>
#include <string>
#include <vector>

#include "rga.h"
#include "im2d.h"
#include "frame_data.h"

struct CameraDmaBufOptions
{
    int camera_id = 0;
    int width = 640;
    int height = 480;
    int fps = 30;
    std::string format = "YUYV";
    bool export_dmabuf = true;
};

class CameraDmaBufCapture
{
public:
    CameraDmaBufCapture() = default;
    ~CameraDmaBufCapture();

    bool open(const CameraDmaBufOptions &options);
    bool read(cv::Mat &frame);
    bool readFrameData(FrameData &frame_data, int frame_index);
    void close();

    bool isOpened() const { return fd_ >= 0 && streaming_; }
    int width() const { return width_; }
    int height() const { return height_; }
    double fps() const { return fps_; }
    const std::string& actualFormat() const { return actual_format_; }

private:
    struct Buffer
    {
        void *start = nullptr;
        size_t length = 0;
        int dmabuf_fd = -1;
        rga_buffer_handle_t rga_handle = 0;
    };

    static int xioctl(int fd, unsigned long request, void *arg);
    static uint32_t fourccFromString(const std::string &fmt);
    static std::string fourccName(uint32_t fourcc);

    bool setupFormat(const CameraDmaBufOptions &options);
    bool setupBuffers();
    bool startStream();
    void stopStream();
    bool dequeueBuffer(v4l2_buffer &buf);
    bool queueBuffer(uint32_t index);

    int fd_ = -1;
    int width_ = 0;
    int height_ = 0;
    double fps_ = 0.0;
    uint32_t pixfmt_ = 0;
    int bytesperline_ = 0;
    std::string actual_format_;
    std::vector<Buffer> buffers_;
    std::vector<uint8_t> bgr_buf_;
    rga_buffer_handle_t bgr_handle_ = 0;
    bool streaming_ = false;
    bool export_dmabuf_ = true;
};

#endif
