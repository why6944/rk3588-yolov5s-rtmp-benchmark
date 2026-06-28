// main.cpp
#include <opencv2/opencv.hpp>
#include <atomic>
#include <thread>
#include <iostream>
#include <string>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>
#include <map>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "SafeQueue.h"
#include "yolov5s.h"
#include "thread_poll.h"
#include "streamer.h"
#include "perf_monitor.h"
#include "benchmark_stats.h"
#include "debug_log.h"

#include "rga.h"
#include "drmrga.h"
#include "im2d.h"
#include "RgaUtils.h"

#define ALIGN(x, a) (((x)+(a)-1)&~((a)-1))

int g_width      = 0;
int g_height     = 0;
int g_hor_stride = 0;
int g_ver_stride = 0;

extern "C" {
int g_verbose_log = 0;
float g_box_threshold = BOX_THRESHOLD;
float g_nms_threshold = NMS_THRESHOLD;
}

enum class RunMode
{
    Full,
    InferOnly,
    RknnOnly,
    MppOnly,
    Rtmp,
    Snapshot
};

struct RunOptions
{
    int thread_count = 3;
    int loops = 900;
    RunMode mode = RunMode::Full;
    std::string rtmp_url = "rtmp://192.168.137.1:1935/live/app";
    int snapshot_frame = 120;
    std::string snapshot_output = "../debug_records/snapshot.png";
};

static const char* modeName(RunMode mode)
{
    switch(mode)
    {
    case RunMode::Full:      return "full";
    case RunMode::InferOnly: return "infer-only";
    case RunMode::RknnOnly:  return "rknn-only";
    case RunMode::MppOnly:   return "mpp-only";
    case RunMode::Rtmp:      return "rtmp";
    case RunMode::Snapshot:  return "snapshot";
    }
    return "full";
}

static bool modeWritesAvi(RunMode mode)
{
    return mode == RunMode::Full;
}

static bool modeUsesMpp(RunMode mode)
{
    return mode == RunMode::Full || mode == RunMode::MppOnly || mode == RunMode::Rtmp;
}

static bool modeDrawsResults(RunMode mode)
{
    return mode == RunMode::Full || mode == RunMode::MppOnly || mode == RunMode::Rtmp;
}

static RunMode parseMode(const char *value)
{
    if(std::strcmp(value, "infer-only") == 0 || std::strcmp(value, "infer") == 0)
        return RunMode::InferOnly;
    if(std::strcmp(value, "rknn-only") == 0 || std::strcmp(value, "rknn") == 0)
        return RunMode::RknnOnly;
    if(std::strcmp(value, "mpp-only") == 0 || std::strcmp(value, "mpp") == 0)
        return RunMode::MppOnly;
    if(std::strcmp(value, "rtmp") == 0 || std::strcmp(value, "rtmp-only") == 0)
        return RunMode::Rtmp;
    if(std::strcmp(value, "snapshot") == 0 || std::strcmp(value, "snap") == 0)
        return RunMode::Snapshot;
    return RunMode::Full;
}

static RunOptions parseOptions(int argc, char **argv)
{
    RunOptions options;
    for(int i = 1; i < argc; ++i)
    {
        if((std::strcmp(argv[i], "--threads") == 0 || std::strcmp(argv[i], "-t") == 0) && i + 1 < argc)
        {
            options.thread_count = std::atoi(argv[++i]);
        }
        else if(std::strncmp(argv[i], "--threads=", 10) == 0)
        {
            options.thread_count = std::atoi(argv[i] + 10);
        }
        else if(std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
        {
            options.mode = parseMode(argv[++i]);
        }
        else if(std::strncmp(argv[i], "--mode=", 7) == 0)
        {
            options.mode = parseMode(argv[i] + 7);
        }
        else if(std::strcmp(argv[i], "--loops") == 0 && i + 1 < argc)
        {
            options.loops = std::atoi(argv[++i]);
        }
        else if(std::strncmp(argv[i], "--loops=", 8) == 0)
        {
            options.loops = std::atoi(argv[i] + 8);
        }
        else if(std::strcmp(argv[i], "--verbose") == 0 || std::strcmp(argv[i], "-v") == 0)
        {
            g_verbose_log = 1;
        }
        else if(std::strcmp(argv[i], "--quiet") == 0)
        {
            g_verbose_log = 0;
        }
        else if(std::strcmp(argv[i], "--box-threshold") == 0 && i + 1 < argc)
        {
            g_box_threshold = std::atof(argv[++i]);
        }
        else if(std::strncmp(argv[i], "--box-threshold=", 16) == 0)
        {
            g_box_threshold = std::atof(argv[i] + 16);
        }
        else if(std::strcmp(argv[i], "--nms-threshold") == 0 && i + 1 < argc)
        {
            g_nms_threshold = std::atof(argv[++i]);
        }
        else if(std::strncmp(argv[i], "--nms-threshold=", 16) == 0)
        {
            g_nms_threshold = std::atof(argv[i] + 16);
        }
        else if(std::strcmp(argv[i], "--rtmp-url") == 0 && i + 1 < argc)
        {
            options.rtmp_url = argv[++i];
        }
        else if(std::strncmp(argv[i], "--rtmp-url=", 11) == 0)
        {
            options.rtmp_url = argv[i] + 11;
        }
        else if(std::strcmp(argv[i], "--snapshot-frame") == 0 && i + 1 < argc)
        {
            options.snapshot_frame = std::atoi(argv[++i]);
        }
        else if(std::strncmp(argv[i], "--snapshot-frame=", 17) == 0)
        {
            options.snapshot_frame = std::atoi(argv[i] + 17);
        }
        else if(std::strcmp(argv[i], "--snapshot-output") == 0 && i + 1 < argc)
        {
            options.snapshot_output = argv[++i];
        }
        else if(std::strncmp(argv[i], "--snapshot-output=", 18) == 0)
        {
            options.snapshot_output = argv[i] + 18;
        }
        else if(argv[i][0] != '-')
        {
            options.thread_count = std::atoi(argv[i]);
        }
    }
    if(options.thread_count <= 0) options.thread_count = 1;
    if(options.loops <= 0) options.loops = 1;
    return options;
}

static void ensureDebugDirs()
{
    std::system("mkdir -p ../debug_records/csv ../debug_records/logs");
}

void BGR_to_NV12_with_RGA(
    const uint8_t       *bgr_src,
    char                *bgr_buf,
    rga_buffer_handle_t  bgr_handle,
    uint8_t             *nv12_buf,
    rga_buffer_handle_t  nv12_handle,
    int width, int height)
{
    memcpy(bgr_buf, bgr_src, width * height * 3);

    rga_buffer_t src = wrapbuffer_handle(bgr_handle,  width,        height,        RK_FORMAT_RGB_888);
    rga_buffer_t dst = wrapbuffer_handle(nv12_handle, g_hor_stride, g_ver_stride,  RK_FORMAT_YCrCb_420_SP);

    int ret = imcheck(src, dst, {}, {});
    if(ret != IM_STATUS_NOERROR)
        printf("%d, imcheck error! %s\n", __LINE__, imStrError((IM_STATUS)ret));

    ret = imcvtcolor(src, dst, RK_FORMAT_RGB_888, RK_FORMAT_YCrCb_420_SP);
    if(ret != IM_STATUS_SUCCESS)
        printf("%d, cvtColor error! %s\n", __LINE__, imStrError((IM_STATUS)ret));
}

const int MAX_CONCURRENT_FRAMES = 10;

struct FrameData {
    cv::Mat frame;
    int index;
};

SafeQueue<FrameData> g_readQueue(50);
SafeQueue<FrameData> g_writeQueue(50);
std::atomic<bool>    g_readFinish(false);
std::atomic<bool>    g_processFinish(false);

void readThreadFunc(cv::VideoCapture &cap)
{
    int idx = 0;
    while(true)
    {
        cv::Mat frame;
        if(!cap.read(frame))
        {
            std::cerr << "[ReadThread] read failed or EOF.\n";
            break;
        }
        FrameData data{ std::move(frame), idx++ };
        g_readQueue.enqueue(std::move(data));
    }
    g_readFinish = true;
    std::cerr << "[ReadThread] finished.\n";
}

void aggregatorThreadFunc(ThreadPoll &npu_pool)
{
    int nextWriteIndex = 0;
    std::map<int, std::future<ProcessResult>> tasks_inflight;

    while(true)
    {
        FrameData inputFD;
        if(!g_readQueue.empty() && tasks_inflight.size() < MAX_CONCURRENT_FRAMES)
        {
            if(g_readQueue.dequeue(inputFD))
            {
                auto fut = npu_pool.submit_task_async(inputFD.index, std::move(inputFD.frame));
                tasks_inflight[inputFD.index] = std::move(fut);
            }
        }

        auto it = tasks_inflight.find(nextWriteIndex);
        while(it != tasks_inflight.end())
        {
            auto status = it->second.wait_for(std::chrono::milliseconds(1));
            if(status == std::future_status::ready)
            {
                ProcessResult result = it->second.get();

                FrameData outputFD;
                outputFD.index = nextWriteIndex;
                outputFD.frame = std::move(result.processed_img);
                g_writeQueue.enqueue(std::move(outputFD));

                tasks_inflight.erase(it);
                nextWriteIndex++;
                it = tasks_inflight.find(nextWriteIndex);
            }
            else
            {
                break;
            }
        }

        if(g_readFinish && g_readQueue.empty() && tasks_inflight.empty())
            break;

        if(tasks_inflight.size() >= MAX_CONCURRENT_FRAMES)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        else
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    g_processFinish = true;
    g_writeQueue.stop();
    std::cerr << "[AggregatorThread] finished.\n";
}

void writeThreadFunc(cv::VideoWriter *writer, RunMode mode)
{
    const bool write_avi = modeWritesAvi(mode);
    const bool use_mpp = modeUsesMpp(mode);
    int bgr_size  = g_width * g_height * 3;
    int nv12_size = g_hor_stride * g_ver_stride * 3 / 2;

    char *bgr_write_buf = nullptr;
    uint8_t *nv12_buf = nullptr;
    rga_buffer_handle_t bgr_handle = 0;
    rga_buffer_handle_t nv12_handle = 0;

    if(use_mpp)
    {
        bgr_write_buf = (char *)malloc(bgr_size);
        nv12_buf      = (uint8_t *)malloc(nv12_size);
        bgr_handle    = importbuffer_virtualaddr(bgr_write_buf, bgr_size);
        nv12_handle   = importbuffer_virtualaddr(nv12_buf,      nv12_size);

        if(bgr_handle == 0 || nv12_handle == 0)
            printf("[WriteThread] RGA buffer prealloc failed!\n");
    }

    while(true)
    {
        if(g_processFinish && g_writeQueue.empty())
            break;

        FrameData outputFD;
        if(!g_writeQueue.dequeue(outputFD))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        if(outputFD.frame.empty())
            continue;

        if(write_avi && writer)
        {
            auto write_start = std::chrono::high_resolution_clock::now();
            writer->write(outputFD.frame);
            auto write_end = std::chrono::high_resolution_clock::now();
            BenchmarkStats::instance().record_write(
                std::chrono::duration_cast<std::chrono::microseconds>(write_end - write_start).count());
        }

        if(use_mpp)
        {
            auto mpp_start = std::chrono::high_resolution_clock::now();
            BGR_to_NV12_with_RGA(
                outputFD.frame.data,
                bgr_write_buf, bgr_handle,
                nv12_buf,      nv12_handle,
                outputFD.frame.cols, outputFD.frame.rows
            );
            process_frame(nv12_buf, nv12_size);
            auto mpp_end = std::chrono::high_resolution_clock::now();
            BenchmarkStats::instance().record_mpp(
                std::chrono::duration_cast<std::chrono::microseconds>(mpp_end - mpp_start).count());
        }
    }

    if(bgr_handle)  releasebuffer_handle(bgr_handle);
    if(nv12_handle) releasebuffer_handle(nv12_handle);
    if(bgr_write_buf) free(bgr_write_buf);
    if(nv12_buf) free(nv12_buf);

    std::cerr << "[WriteThread] finished.\n";
}

static int runSnapshot(const RunOptions &options, const std::string &inPath)
{
    cv::VideoCapture cap(inPath);
    if(!cap.isOpened())
    {
        std::cerr << "Fail to open input video: " << inPath << "\n";
        return -1;
    }

    cv::Mat frame;
    int idx = 0;
    while(idx <= options.snapshot_frame)
    {
        if(!cap.read(frame) || frame.empty())
        {
            std::cerr << "Fail to read snapshot frame: " << options.snapshot_frame << "\n";
            return -1;
        }
        idx++;
    }

    BenchmarkStats::instance().reset(1, modeName(options.mode));
    Yolov5s yolo("../model/yolov5s.rknn", 0);
    detect_result_group_t detections;
    yolo.inference_image(frame, detections);
    yolo.draw_result(frame, detections);

    if(!cv::imwrite(options.snapshot_output, frame))
    {
        std::cerr << "Fail to write snapshot: " << options.snapshot_output << "\n";
        return -1;
    }

    std::cout << "[Snapshot] frame=" << options.snapshot_frame
              << ", boxes=" << detections.box_count
              << ", output=" << options.snapshot_output << std::endl;
    BenchmarkStats::instance().print_summary();
    BenchmarkStats::instance().append_detail_csv("../debug_records/csv/benchmark_detail.csv");
    return 0;
}

static int runRknnOnly(const RunOptions &options, const std::string &inPath)
{
    cv::VideoCapture cap(inPath);
    if(!cap.isOpened())
    {
        std::cerr << "Fail to open input video: " << inPath << "\n";
        return -1;
    }

    cv::Mat frame;
    if(!cap.read(frame) || frame.empty())
    {
        std::cerr << "Fail to read first frame for rknn-only benchmark.\n";
        return -1;
    }

    std::vector<std::shared_ptr<Yolov5s>> yolo_group;
    yolo_group.reserve(options.thread_count);
    for(int i = 0; i < options.thread_count; ++i)
        yolo_group.emplace_back(std::make_shared<Yolov5s>("../model/yolov5s.rknn", i % 3));

    BenchmarkStats::instance().reset(options.thread_count, modeName(options.mode));
    PerfMonitor::instance().start("../debug_records/csv/perf_log.csv", 500);

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> workers;
    workers.reserve(options.thread_count);
    int base_loops = options.loops / options.thread_count;
    int extra = options.loops % options.thread_count;
    for(int i = 0; i < options.thread_count; ++i)
    {
        int loops = base_loops + (i < extra ? 1 : 0);
        workers.emplace_back([yolo = yolo_group[i], frame, loops]() mutable {
            yolo->benchmark_rknn_only(frame, loops);
        });
    }

    for(auto &t : workers)
        if(t.joinable()) t.join();

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "处理总用时：" << elapsed_ms.count() << " ms\n";

    BenchmarkStats::instance().set_total_elapsed_ms(elapsed_ms.count());
    BenchmarkStats::instance().print_summary();
    BenchmarkStats::instance().append_csv("../debug_records/csv/benchmark_summary.csv");
    BenchmarkStats::instance().append_detail_csv("../debug_records/csv/benchmark_detail.csv");

    PerfMonitor::instance().stop();
    std::cerr << "[Main] All done.\n";
    return 0;
}

int main(int argc, char **argv)
{
    ensureDebugDirs();
    RunOptions options = parseOptions(argc, argv);
    std::string inPath  = "../video.mp4";
    std::string outPath = "../output.avi";

    std::cout << "[Main] mode=" << modeName(options.mode)
              << ", thread_count=" << options.thread_count
              << ", loops=" << options.loops
              << ", verbose=" << g_verbose_log
              << ", box_threshold=" << g_box_threshold
              << ", nms_threshold=" << g_nms_threshold;
    if(options.mode == RunMode::Rtmp)
        std::cout << ", rtmp_url=" << options.rtmp_url;
    std::cout << std::endl;

    if(options.mode == RunMode::Snapshot)
        return runSnapshot(options, inPath);

    if(options.mode == RunMode::RknnOnly)
        return runRknnOnly(options, inPath);

    BenchmarkStats::instance().reset(options.thread_count, modeName(options.mode));
    auto start = std::chrono::high_resolution_clock::now();
    PerfMonitor::instance().start("../debug_records/csv/perf_log.csv", 500);

    cv::VideoCapture cap(inPath);
    if(!cap.isOpened())
    {
        std::cerr << "Fail to open input video: " << inPath << "\n";
        PerfMonitor::instance().stop();
        return -1;
    }

    int    width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int    height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps    = cap.get(cv::CAP_PROP_FPS);
    if(fps < 1.0) fps = 25.0;

    int hor_stride = ALIGN(width, 16);
    int ver_stride = ALIGN(height, 16);

    g_width      = width;
    g_height     = height;
    g_hor_stride = hor_stride;
    g_ver_stride = ver_stride;

    const bool write_avi = modeWritesAvi(options.mode);
    const bool use_mpp = modeUsesMpp(options.mode);
    std::unique_ptr<cv::VideoWriter> writer;

    if(use_mpp)
    {
        int bitrate = width * height / 8 * (int)(fps / 1);
        const char *rtmp_url = (options.mode == RunMode::Rtmp) ? options.rtmp_url.c_str() : NULL;
        if(init_streamer(width, height, (int)fps, bitrate, rtmp_url) != 0)
        {
            std::cerr << "Fail to initialize MPP/RTMP streamer.\n";
            PerfMonitor::instance().stop();
            return -1;
        }
    }

    if(write_avi)
    {
        int fourcc  = cv::VideoWriter::fourcc('H','2','6','4');
        writer.reset(new cv::VideoWriter(outPath, fourcc, fps, cv::Size(width, height)));
        if(!writer->isOpened())
        {
            std::cerr << "Fail to create output video: " << outPath << "\n";
            if(use_mpp) close_streamer();
            PerfMonitor::instance().stop();
            return -1;
        }
    }

    bool draw_results = modeDrawsResults(options.mode);
    ThreadPoll npu_pool("../model/yolov5s.rknn", options.thread_count, draw_results);

    std::thread tRead(readThreadFunc, std::ref(cap));
    std::thread tAggregator(aggregatorThreadFunc, std::ref(npu_pool));
    std::thread tWrite(writeThreadFunc, writer.get(), options.mode);

    tRead.join();
    tAggregator.join();
    tWrite.join();

    g_readQueue.stop();
    g_writeQueue.stop();

    if(writer)
        writer->release();
    if(use_mpp)
        close_streamer();

    auto end        = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "处理总用时：" << elapsed_ms.count() << " ms\n";

    BenchmarkStats::instance().set_total_elapsed_ms(elapsed_ms.count());
    BenchmarkStats::instance().print_summary();
    BenchmarkStats::instance().append_csv("../debug_records/csv/benchmark_summary.csv");
    BenchmarkStats::instance().append_detail_csv("../debug_records/csv/benchmark_detail.csv");

    PerfMonitor::instance().stop();

    std::cerr << "[Main] All done.\n";
    return 0;
}
