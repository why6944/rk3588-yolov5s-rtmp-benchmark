#ifndef FRAME_DATA_H
#define FRAME_DATA_H

#include <functional>
#include <memory>
#include <stdint.h>
#include <stddef.h>
#include <string>
#include <utility>

#include <opencv2/core.hpp>

#include "rga.h"

enum class FrameStorageType {
    Mat,
    CameraMmapMat,
    CameraDmaBufMat,
    CameraDmaBuf
};

struct DmaBufFrameRef {
    int fd = -1;
    void *virtual_addr = nullptr;
    size_t size = 0;
    int width = 0;
    int height = 0;
    int bytesperline = 0;
    uint32_t fourcc = 0;
    rga_buffer_handle_t rga_handle = 0;
    uint32_t buffer_index = 0;
    std::string source;
    std::function<void()> release;

    ~DmaBufFrameRef()
    {
        if(release)
            release();
    }
};

struct FrameData {
    int index = -1;
    cv::Mat frame;
    FrameStorageType storage = FrameStorageType::Mat;
    std::shared_ptr<DmaBufFrameRef> dmabuf;

    static FrameData fromMat(cv::Mat mat, int frame_index)
    {
        FrameData data;
        data.index = frame_index;
        data.frame = std::move(mat);
        data.storage = FrameStorageType::Mat;
        return data;
    }

    static FrameData fromCameraDmaBufMat(cv::Mat mat, int frame_index)
    {
        FrameData data = fromMat(std::move(mat), frame_index);
        data.storage = FrameStorageType::CameraDmaBufMat;
        return data;
    }

    static FrameData fromCameraMmapMat(cv::Mat mat, std::shared_ptr<DmaBufFrameRef> ref, int frame_index)
    {
        FrameData data = fromMat(std::move(mat), frame_index);
        data.storage = FrameStorageType::CameraMmapMat;
        data.dmabuf = std::move(ref);
        return data;
    }

    static FrameData fromCameraDmaBufMat(cv::Mat mat, std::shared_ptr<DmaBufFrameRef> ref, int frame_index)
    {
        FrameData data = fromCameraDmaBufMat(std::move(mat), frame_index);
        data.dmabuf = std::move(ref);
        return data;
    }

    static FrameData fromCameraDmaBuf(std::shared_ptr<DmaBufFrameRef> ref, int frame_index)
    {
        FrameData data;
        data.index = frame_index;
        data.storage = FrameStorageType::CameraDmaBuf;
        data.dmabuf = std::move(ref);
        return data;
    }

    bool hasMat() const
    {
        return !frame.empty();
    }

    bool hasDmaBuf() const
    {
        return dmabuf && dmabuf->fd >= 0;
    }

    bool hasCameraBuffer() const
    {
        return dmabuf && dmabuf->rga_handle != 0;
    }

    bool cameFromCameraDmaBuf() const
    {
        return storage == FrameStorageType::CameraDmaBufMat ||
               storage == FrameStorageType::CameraDmaBuf;
    }


    bool cameFromCameraMmap() const
    {
        return storage == FrameStorageType::CameraMmapMat;
    }
};

#endif
