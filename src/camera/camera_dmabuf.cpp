#include "camera_dmabuf.h"

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <unistd.h>

#include <cstdlib>
#include <iostream>
#include <opencv2/imgproc.hpp>

#include "RgaUtils.h"

#ifndef DMA_HEAP_IOC_MAGIC
#define DMA_HEAP_IOC_MAGIC 'H'
#endif

struct dma_heap_allocation_data_local {
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;
    uint64_t heap_flags;
};

#ifndef DMA_HEAP_IOCTL_ALLOC
#define DMA_HEAP_IOCTL_ALLOC _IOWR(DMA_HEAP_IOC_MAGIC, 0x0, struct dma_heap_allocation_data_local)
#endif

static int allocateCameraDmaHeapFd(size_t size)
{
    const char *heap_paths[] = {
        "/dev/dma_heap/system-uncached",
        "/dev/dma_heap/cma-uncached",
        "/dev/dma_heap/system",
        "/dev/dma_heap/cma"
    };

    for (const char *path : heap_paths) {
        int heap_fd = ::open(path, O_RDWR | O_CLOEXEC);
        if (heap_fd < 0)
            continue;

        dma_heap_allocation_data_local data{};
        data.len = size;
        data.fd_flags = O_RDWR | O_CLOEXEC;
        if (::ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &data) == 0) {
            ::close(heap_fd);
            return static_cast<int>(data.fd);
        }
        ::close(heap_fd);
    }
    return -1;
}

CameraDmaBufCapture::~CameraDmaBufCapture()
{
    close();
}

int CameraDmaBufCapture::xioctl(int fd, unsigned long request, void *arg)
{
    int r;
    do {
        r = ioctl(fd, request, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

uint32_t CameraDmaBufCapture::fourccFromString(const std::string &fmt)
{
    if (fmt == "MJPG") return V4L2_PIX_FMT_MJPEG;
    if (fmt == "YUYV") return V4L2_PIX_FMT_YUYV;
    if (fmt.size() >= 4) return v4l2_fourcc(fmt[0], fmt[1], fmt[2], fmt[3]);
    return V4L2_PIX_FMT_YUYV;
}

std::string CameraDmaBufCapture::fourccName(uint32_t fourcc)
{
    char name[5];
    name[0] = fourcc & 0xff;
    name[1] = (fourcc >> 8) & 0xff;
    name[2] = (fourcc >> 16) & 0xff;
    name[3] = (fourcc >> 24) & 0xff;
    name[4] = 0;
    return std::string(name);
}

bool CameraDmaBufCapture::open(const CameraDmaBufOptions &options)
{
    close();
    export_dmabuf_ = options.export_dmabuf;
    copy_to_mat_ = options.copy_to_mat;

    std::string dev = "/dev/video" + std::to_string(options.camera_id);
    fd_ = ::open(dev.c_str(), O_RDWR | O_NONBLOCK, 0);
    if (fd_ < 0) {
        std::cerr << "[CameraDmaBuf] open " << dev << " failed: " << strerror(errno) << "\n";
        return false;
    }

    v4l2_capability cap{};
    if (xioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0) {
        std::cerr << "[CameraDmaBuf] VIDIOC_QUERYCAP failed: " << strerror(errno) << "\n";
        close();
        return false;
    }

    if (!setupFormat(options) || !setupBuffers() || !startStream()) {
        close();
        return false;
    }

    std::cout << "[CameraDmaBuf] opened " << dev
              << ", actual=" << width_ << "x" << height_
              << "@" << fps_ << "fps"
              << ", format=" << actual_format_
              << ", buffers=" << buffers_.size()
              << ", transport=" << (export_dmabuf_ ? "dmabuf-fd" : "mmap-virtualaddr")
              << std::endl;
    return true;
}

bool CameraDmaBufCapture::setupFormat(const CameraDmaBufOptions &options)
{
    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = options.width;
    fmt.fmt.pix.height = options.height;
    fmt.fmt.pix.pixelformat = fourccFromString(options.format);
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (xioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
        std::cerr << "[CameraDmaBuf] VIDIOC_S_FMT " << options.format
                  << " failed: " << strerror(errno) << "\n";
        return false;
    }

    width_ = fmt.fmt.pix.width;
    height_ = fmt.fmt.pix.height;
    pixfmt_ = fmt.fmt.pix.pixelformat;
    bytesperline_ = fmt.fmt.pix.bytesperline;
    actual_format_ = fourccName(pixfmt_);

    if (pixfmt_ != V4L2_PIX_FMT_YUYV) {
        std::cerr << "[CameraDmaBuf] only YUYV supports direct RGA read in this backend, actual="
                  << actual_format_ << "\n";
        return false;
    }

    v4l2_streamparm parm{};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = options.fps;
    if (xioctl(fd_, VIDIOC_S_PARM, &parm) == 0 && parm.parm.capture.timeperframe.numerator != 0) {
        fps_ = static_cast<double>(parm.parm.capture.timeperframe.denominator) /
               static_cast<double>(parm.parm.capture.timeperframe.numerator);
    } else {
        fps_ = options.fps;
    }

    return true;
}

bool CameraDmaBufCapture::setupBuffers()
{
    v4l2_requestbuffers req{};
    uint32_t requested_count = 8;
    if (const char *env_count = std::getenv("V4L2_BUFFER_COUNT")) {
        int parsed = std::atoi(env_count);
        if (parsed >= 2 && parsed <= 32)
            requested_count = static_cast<uint32_t>(parsed);
        else
            std::cerr << "[CameraDmaBuf] ignore invalid V4L2_BUFFER_COUNT=" << env_count
                      << ", valid range is 2..32\n";
    }
    req.count = requested_count;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (xioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
        std::cerr << "[CameraDmaBuf] VIDIOC_REQBUFS failed: " << strerror(errno) << "\n";
        return false;
    }
    if (req.count < 2) {
        std::cerr << "[CameraDmaBuf] insufficient V4L2 buffers: " << req.count << "\n";
        return false;
    }

    buffers_.resize(req.count);
    {
        std::lock_guard<std::mutex> lock(buffer_state_mutex_);
        client_holds_buffer_.assign(req.count, false);
        outstanding_buffers_ = 0;
        // Keep one buffer queued in the driver, so the UVC producer never runs dry.
        max_outstanding_buffers_ = req.count - 1;
    }
    for (uint32_t i = 0; i < req.count; ++i) {
        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (xioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
            std::cerr << "[CameraDmaBuf] VIDIOC_QUERYBUF " << i << " failed: " << strerror(errno) << "\n";
            return false;
        }

        buffers_[i].length = buf.length;
        buffers_[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
        if (buffers_[i].start == MAP_FAILED) {
            std::cerr << "[CameraDmaBuf] mmap " << i << " failed: " << strerror(errno) << "\n";
            buffers_[i].start = nullptr;
            return false;
        }

        if (export_dmabuf_) {
            v4l2_exportbuffer exp{};
            exp.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            exp.index = i;
            exp.flags = O_CLOEXEC;
            if (xioctl(fd_, VIDIOC_EXPBUF, &exp) < 0) {
                std::cerr << "[CameraDmaBuf] VIDIOC_EXPBUF " << i << " failed: " << strerror(errno) << "\n";
                return false;
            }
            buffers_[i].dmabuf_fd = exp.fd;
            // The UVC buffer must reach RGA through its exported DMA-BUF fd.
            // Importing the V4L2 mmap virtual address can fault in this kernel's RGA DMA sync path.
            buffers_[i].rga_handle = importbuffer_fd(exp.fd, buffers_[i].length);
            if (!buffers_[i].rga_handle) {
                std::cerr << "[CameraDmaBuf] import exported DMA-BUF " << i << " failed\n";
                return false;
            }
        } else {
            buffers_[i].rga_handle = importbuffer_virtualaddr(buffers_[i].start, buffers_[i].length);
            if (!buffers_[i].rga_handle) {
                std::cerr << "[CameraDmaBuf] import camera buffer " << i << " failed\n";
                return false;
            }
        }
    }

    if (copy_to_mat_) {
        bgr_buf_size_ = static_cast<size_t>(width_) * height_ * 3;
        bgr_dmabuf_fd_ = allocateCameraDmaHeapFd(bgr_buf_size_);
        if (bgr_dmabuf_fd_ < 0) {
            std::cerr << "[CameraDmaBuf] DMA heap allocation for RGB output failed\n";
            return false;
        }
        void *mapped = ::mmap(nullptr, bgr_buf_size_, PROT_READ | PROT_WRITE,
                              MAP_SHARED, bgr_dmabuf_fd_, 0);
        if (mapped == MAP_FAILED) {
            std::cerr << "[CameraDmaBuf] DMA heap RGB output mmap failed: " << strerror(errno) << "\n";
            return false;
        }
        bgr_buf_ = static_cast<uint8_t *>(mapped);
        bgr_handle_ = importbuffer_fd(bgr_dmabuf_fd_, bgr_buf_size_);
        if (!bgr_handle_) {
            std::cerr << "[CameraDmaBuf] import DMA-BUF RGB output failed\n";
            return false;
        }
    }

    return true;
}

bool CameraDmaBufCapture::startStream()
{
    for (uint32_t i = 0; i < buffers_.size(); ++i) {
        if (!queueBuffer(i)) return false;
    }

    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
        std::cerr << "[CameraDmaBuf] VIDIOC_STREAMON failed: " << strerror(errno) << "\n";
        return false;
    }
    streaming_ = true;
    return true;
}

void CameraDmaBufCapture::stopStream()
{
    if (fd_ >= 0 && streaming_) {
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(fd_, VIDIOC_STREAMOFF, &type);
        streaming_ = false;
    }
}

bool CameraDmaBufCapture::queueBuffer(uint32_t index)
{
    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = index;
    if (xioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
        std::cerr << "[CameraDmaBuf] VIDIOC_QBUF " << index << " failed: " << strerror(errno) << "\n";
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(buffer_state_mutex_);
        if (index < client_holds_buffer_.size() && client_holds_buffer_[index]) {
            client_holds_buffer_[index] = false;
            if (outstanding_buffers_ > 0)
                --outstanding_buffers_;
            buffer_available_cv_.notify_one();
        }
    }
    return true;
}

bool CameraDmaBufCapture::dequeueBuffer(v4l2_buffer &buf)
{
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    for (;;) {
        int ret = xioctl(fd_, VIDIOC_DQBUF, &buf);
        if (ret == 0) return true;
        if (errno != EAGAIN) {
            std::cerr << "[CameraDmaBuf] VIDIOC_DQBUF failed: " << strerror(errno) << "\n";
            return false;
        }

        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(fd_, &fds);
        timeval tv{};
        tv.tv_sec = 2;
        ret = select(fd_ + 1, &fds, NULL, NULL, &tv);
        if (ret <= 0) {
            std::cerr << "[CameraDmaBuf] select timeout/error: " << strerror(errno) << "\n";
            return false;
        }
    }
}

bool CameraDmaBufCapture::read(cv::Mat &frame)
{
    if (!isOpened()) return false;

    v4l2_buffer buf{};
    if (!dequeueBuffer(buf)) return false;

    bool ok = false;
    if (buf.index < buffers_.size()) {
        Buffer &src_buffer = buffers_[buf.index];
        if (src_buffer.rga_handle) {
            rga_buffer_t src = wrapbuffer_handle(src_buffer.rga_handle, width_, height_, RK_FORMAT_YUYV_422);
            rga_buffer_t dst = wrapbuffer_handle(bgr_handle_, width_, height_, RK_FORMAT_RGB_888);
            int check = imcheck(src, dst, {}, {});
            int ret = check == IM_STATUS_NOERROR ? imcvtcolor(src, dst, RK_FORMAT_YUYV_422, RK_FORMAT_RGB_888) : check;
            if (ret == IM_STATUS_SUCCESS) {
                cv::Mat rgb(height_, width_, CV_8UC3, bgr_buf_);
                cv::cvtColor(rgb, frame, cv::COLOR_RGB2BGR);
                ok = true;
            } else {
                std::cerr << "[CameraDmaBuf] RGA YUYV->BGR failed: " << imStrError((IM_STATUS)ret) << "\n";
            }
        }
    }

    if (!queueBuffer(buf.index)) return false;
    return ok;
}


bool CameraDmaBufCapture::readFrameData(FrameData &frame_data, int frame_index)
{
    if (!isOpened()) return false;

    {
        std::unique_lock<std::mutex> lock(buffer_state_mutex_);
        buffer_available_cv_.wait(lock, [this] {
            return !streaming_ || outstanding_buffers_ < max_outstanding_buffers_;
        });
        if (!streaming_)
            return false;
    }

    v4l2_buffer buf{};
    if (!dequeueBuffer(buf)) return false;
    const int64_t dqbuf_ns = latencyMonotonicNowNs();
    const int64_t v4l2_timestamp_ns = static_cast<int64_t>(buf.timestamp.tv_sec) * 1000000000LL +
                                      static_cast<int64_t>(buf.timestamp.tv_usec) * 1000LL;
    const bool v4l2_timestamp_monotonic =
        (buf.flags & V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC) != 0;

    if (buf.index >= buffers_.size()) {
        queueBuffer(buf.index);
        return false;
    }

    bool duplicate_buffer = false;
    {
        std::lock_guard<std::mutex> lock(buffer_state_mutex_);
        duplicate_buffer = buf.index >= client_holds_buffer_.size() ||
                           client_holds_buffer_[buf.index];
        if (!duplicate_buffer) {
            client_holds_buffer_[buf.index] = true;
            ++outstanding_buffers_;
        }
    }
    if (duplicate_buffer) {
        std::cerr << "[CameraDmaBuf] unexpected duplicate DQBUF index " << buf.index << "\n";
        queueBuffer(buf.index);
        return false;
    }

    Buffer &src_buffer = buffers_[buf.index];
    cv::Mat cloned_frame;
    if (copy_to_mat_) {
        if (!src_buffer.rga_handle) {
            std::cerr << "[CameraDmaBuf] BGR source handle is unavailable\n";
            queueBuffer(buf.index);
            return false;
        }
        rga_buffer_t src = wrapbuffer_handle(src_buffer.rga_handle, width_, height_, RK_FORMAT_YUYV_422);
        rga_buffer_t dst = wrapbuffer_handle(bgr_handle_, width_, height_, RK_FORMAT_RGB_888);
        int check = imcheck(src, dst, {}, {});
        int ret = check == IM_STATUS_NOERROR ? imcvtcolor(src, dst, RK_FORMAT_YUYV_422, RK_FORMAT_RGB_888) : check;
        if (ret != IM_STATUS_SUCCESS) {
            std::cerr << "[CameraDmaBuf] RGA YUYV->RGB failed: " << imStrError((IM_STATUS)ret) << "\n";
            queueBuffer(buf.index);
            return false;
        }

        cv::Mat rgb(height_, width_, CV_8UC3, bgr_buf_);
        cv::cvtColor(rgb, cloned_frame, cv::COLOR_RGB2BGR);
    }

    auto ref = std::make_shared<DmaBufFrameRef>();
    ref->fd = src_buffer.dmabuf_fd;
    ref->virtual_addr = src_buffer.start;
    ref->size = src_buffer.length;
    ref->width = width_;
    ref->height = height_;
    ref->bytesperline = bytesperline_;
    ref->fourcc = pixfmt_;
    ref->rga_handle = src_buffer.rga_handle;
    ref->buffer_index = buf.index;
    ref->source = "camera";
    uint32_t buffer_index = buf.index;
    ref->release = [this, buffer_index]() {
        if (this->fd_ >= 0 && this->streaming_) {
            if (!this->queueBuffer(buffer_index)) {
                std::cerr << "[CameraDmaBuf] delayed QBUF " << buffer_index << " failed\n";
            }
        }
    };

    if (!copy_to_mat_)
        frame_data = FrameData::fromCameraDmaBuf(std::move(ref), frame_index);
    else if (export_dmabuf_)
        frame_data = FrameData::fromCameraDmaBufMat(std::move(cloned_frame), std::move(ref), frame_index);
    else
        frame_data = FrameData::fromCameraMmapMat(std::move(cloned_frame), std::move(ref), frame_index);
    frame_data.latency.frame_index = frame_index;
    frame_data.latency.dqbuf_ns = dqbuf_ns;
    frame_data.latency.v4l2_timestamp_ns = v4l2_timestamp_ns;
    frame_data.latency.v4l2_timestamp_monotonic = v4l2_timestamp_monotonic;
    return true;
}

void CameraDmaBufCapture::close()
{
    stopStream();
    {
        std::lock_guard<std::mutex> lock(buffer_state_mutex_);
        client_holds_buffer_.clear();
        outstanding_buffers_ = 0;
        max_outstanding_buffers_ = 0;
    }
    buffer_available_cv_.notify_all();

    if (bgr_handle_) {
        releasebuffer_handle(bgr_handle_);
        bgr_handle_ = 0;
    }
    if (bgr_buf_) {
        ::munmap(bgr_buf_, bgr_buf_size_);
        bgr_buf_ = nullptr;
    }
    bgr_buf_size_ = 0;
    if (bgr_dmabuf_fd_ >= 0) {
        ::close(bgr_dmabuf_fd_);
        bgr_dmabuf_fd_ = -1;
    }

    for (auto &buffer : buffers_) {
        if (buffer.rga_handle) {
            releasebuffer_handle(buffer.rga_handle);
            buffer.rga_handle = 0;
        }
        if (buffer.dmabuf_fd >= 0) {
            ::close(buffer.dmabuf_fd);
            buffer.dmabuf_fd = -1;
        }
        if (buffer.start) {
            munmap(buffer.start, buffer.length);
            buffer.start = nullptr;
        }
    }
    buffers_.clear();

    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }

    width_ = 0;
    height_ = 0;
    fps_ = 0.0;
    pixfmt_ = 0;
    bytesperline_ = 0;
    actual_format_.clear();
}
