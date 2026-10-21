#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "rga.h"
#include "im2d.h"
#include "RgaUtils.h"

struct Buffer {
    void *start = nullptr;
    size_t length = 0;
    int dmabuf_fd = -1;
};

static int xioctl(int fd, unsigned long request, void *arg) {
    int r;
    do {
        r = ioctl(fd, request, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

static uint32_t fourcc_from_string(const std::string &s) {
    if (s == "MJPG") return V4L2_PIX_FMT_MJPEG;
    if (s == "YUYV") return V4L2_PIX_FMT_YUYV;
    return v4l2_fourcc(s[0], s[1], s[2], s[3]);
}

static const char *fourcc_name(uint32_t f) {
    static char name[5];
    name[0] = f & 0xff;
    name[1] = (f >> 8) & 0xff;
    name[2] = (f >> 16) & 0xff;
    name[3] = (f >> 24) & 0xff;
    name[4] = 0;
    return name;
}

int main(int argc, char **argv) {
    std::string dev = "/dev/video0";
    std::string fmt_name = argc > 1 ? argv[1] : "YUYV";
    int width = argc > 2 ? atoi(argv[2]) : 640;
    int height = argc > 3 ? atoi(argv[3]) : 480;
    int fps = argc > 4 ? atoi(argv[4]) : 30;
    std::string out_path = argc > 5 ? argv[5] : "../debug_records/dma_probe_rga.png";
    bool try_rga = fmt_name == "YUYV";

    int fd = open(dev.c_str(), O_RDWR | O_NONBLOCK, 0);
    if (fd < 0) {
        std::cerr << "open " << dev << " failed: " << strerror(errno) << "\n";
        return 1;
    }

    v4l2_capability cap{};
    if (xioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) {
        std::cerr << "VIDIOC_QUERYCAP failed: " << strerror(errno) << "\n";
        close(fd);
        return 1;
    }
    std::cout << "[CAP] driver=" << cap.driver << ", card=" << cap.card
              << ", caps=0x" << std::hex << cap.capabilities << std::dec << "\n";

    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = fourcc_from_string(fmt_name);
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if (xioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
        std::cerr << "VIDIOC_S_FMT " << fmt_name << " failed: " << strerror(errno) << "\n";
        close(fd);
        return 1;
    }
    width = fmt.fmt.pix.width;
    height = fmt.fmt.pix.height;
    uint32_t actual_fourcc = fmt.fmt.pix.pixelformat;
    std::cout << "[FMT] actual=" << width << "x" << height
              << ", fourcc=" << fourcc_name(actual_fourcc)
              << ", sizeimage=" << fmt.fmt.pix.sizeimage
              << ", bytesperline=" << fmt.fmt.pix.bytesperline << "\n";

    v4l2_streamparm parm{};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = fps;
    if (xioctl(fd, VIDIOC_S_PARM, &parm) == 0) {
        std::cout << "[FPS] requested=" << fps << ", actual="
                  << parm.parm.capture.timeperframe.denominator << "/"
                  << parm.parm.capture.timeperframe.numerator << "\n";
    } else {
        std::cout << "[FPS] VIDIOC_S_PARM failed: " << strerror(errno) << "\n";
    }

    v4l2_requestbuffers req{};
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
        std::cerr << "VIDIOC_REQBUFS(MMAP) failed: " << strerror(errno) << "\n";
        close(fd);
        return 1;
    }
    std::cout << "[REQBUFS] count=" << req.count << "\n";

    std::vector<Buffer> buffers(req.count);
    bool expbuf_all_ok = true;
    for (uint32_t i = 0; i < req.count; ++i) {
        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (xioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
            std::cerr << "VIDIOC_QUERYBUF index=" << i << " failed: " << strerror(errno) << "\n";
            return 1;
        }
        buffers[i].length = buf.length;
        buffers[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
        if (buffers[i].start == MAP_FAILED) {
            std::cerr << "mmap index=" << i << " failed: " << strerror(errno) << "\n";
            return 1;
        }

        v4l2_exportbuffer exp{};
        exp.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        exp.index = i;
        exp.flags = O_CLOEXEC;
        if (xioctl(fd, VIDIOC_EXPBUF, &exp) < 0) {
            expbuf_all_ok = false;
            std::cout << "[EXPBUF] index=" << i << " failed: " << strerror(errno) << "\n";
        } else {
            buffers[i].dmabuf_fd = exp.fd;
            std::cout << "[EXPBUF] index=" << i << ", dmabuf_fd=" << exp.fd
                      << ", length=" << buffers[i].length << "\n";
        }
    }

    if (!expbuf_all_ok) {
        std::cout << "[RESULT] camera_dmabuf_export=FAIL\n";
        for (auto &b : buffers) {
            if (b.dmabuf_fd >= 0) close(b.dmabuf_fd);
            if (b.start && b.start != MAP_FAILED) munmap(b.start, b.length);
        }
        close(fd);
        return 2;
    }
    std::cout << "[RESULT] camera_dmabuf_export=OK\n";

    for (uint32_t i = 0; i < req.count; ++i) {
        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (xioctl(fd, VIDIOC_QBUF, &buf) < 0) {
            std::cerr << "VIDIOC_QBUF index=" << i << " failed: " << strerror(errno) << "\n";
            return 1;
        }
    }

    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd, VIDIOC_STREAMON, &type) < 0) {
        std::cerr << "VIDIOC_STREAMON failed: " << strerror(errno) << "\n";
        return 1;
    }

    v4l2_buffer dq{};
    dq.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    dq.memory = V4L2_MEMORY_MMAP;
    for (;;) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(fd, &fds);
        timeval tv{};
        tv.tv_sec = 2;
        int r = select(fd + 1, &fds, NULL, NULL, &tv);
        if (r <= 0) {
            std::cerr << "select timeout/error while waiting frame: " << strerror(errno) << "\n";
            xioctl(fd, VIDIOC_STREAMOFF, &type);
            return 1;
        }
        if (xioctl(fd, VIDIOC_DQBUF, &dq) == 0) break;
        if (errno != EAGAIN) {
            std::cerr << "VIDIOC_DQBUF failed: " << strerror(errno) << "\n";
            xioctl(fd, VIDIOC_STREAMOFF, &type);
            return 1;
        }
    }
    std::cout << "[DQBUF] index=" << dq.index << ", bytesused=" << dq.bytesused
              << ", dmabuf_fd=" << buffers[dq.index].dmabuf_fd << "\n";

    if (!try_rga) {
        std::cout << "[RGA] skipped: format " << fmt_name << " is compressed or unsupported for direct RGA read\n";
        std::cout << "[RESULT] rga_read_camera_fd=SKIP\n";
    } else {
        int src_fd = buffers[dq.index].dmabuf_fd;
        int out_size = width * height * 3;
        char *rgb = (char *)malloc(out_size);
        memset(rgb, 0, out_size);
        rga_buffer_handle_t src_handle = importbuffer_fd(src_fd, buffers[dq.index].length);
        rga_buffer_handle_t dst_handle = importbuffer_virtualaddr(rgb, out_size);
        if (!src_handle || !dst_handle) {
            std::cout << "[RGA] importbuffer failed, src_handle=" << src_handle
                      << ", dst_handle=" << dst_handle << "\n";
            std::cout << "[RESULT] rga_read_camera_fd=FAIL\n";
        } else {
            rga_buffer_t src = wrapbuffer_handle(src_handle, width, height, RK_FORMAT_YUYV_422);
            rga_buffer_t dst = wrapbuffer_handle(dst_handle, width, height, RK_FORMAT_RGB_888);
            int check = imcheck(src, dst, {}, {});
            std::cout << "[RGA] imcheck=" << check << " " << imStrError((IM_STATUS)check) << "\n";
            int ret = check == IM_STATUS_NOERROR ? imcvtcolor(src, dst, RK_FORMAT_YUYV_422, RK_FORMAT_RGB_888) : check;
            std::cout << "[RGA] imcvtcolor ret=" << ret << " " << imStrError((IM_STATUS)ret) << "\n";
            if (ret == IM_STATUS_SUCCESS) {
                cv::Mat rgb_mat(height, width, CV_8UC3, rgb);
                cv::Mat bgr_mat;
                cv::cvtColor(rgb_mat, bgr_mat, cv::COLOR_RGB2BGR);
                bool ok = cv::imwrite(out_path, bgr_mat);
                std::cout << "[RGA] output=" << out_path << ", write=" << (ok ? "OK" : "FAIL") << "\n";
                std::cout << "[RESULT] rga_read_camera_fd=" << (ok ? "OK" : "FAIL") << "\n";
            } else {
                std::cout << "[RESULT] rga_read_camera_fd=FAIL\n";
            }
        }
        if (src_handle) releasebuffer_handle(src_handle);
        if (dst_handle) releasebuffer_handle(dst_handle);
        free(rgb);
    }

    xioctl(fd, VIDIOC_STREAMOFF, &type);
    for (auto &b : buffers) {
        if (b.dmabuf_fd >= 0) close(b.dmabuf_fd);
        if (b.start && b.start != MAP_FAILED) munmap(b.start, b.length);
    }
    close(fd);
    return 0;
}
