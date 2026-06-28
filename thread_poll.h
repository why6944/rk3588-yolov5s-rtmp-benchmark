#ifndef THREADPOLL_H
#define THREADPOLL_H

#include <mutex>
#include <thread>
#include <condition_variable>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <queue>
#include <map>
#include <memory>
#include <future>
#include <exception>

#include "yolov5s.h"

using namespace std;
using namespace cv;

struct ProcessResult {
    cv::Mat processed_img;
    detect_result_group_t detection_results;
    bool success = false;
    std::string error_msg;
};

class ThreadPoll
{
public:
    ThreadPoll(const char* model_path, int num_threads, bool draw_results = true);
    ~ThreadPoll();

    std::future<ProcessResult> submit_task_async(int index, cv::Mat img);

private:
    void worker(int id);
    void init(const char* model_path, int num_threads);
    bool draw_results_ = true;

    // -------------------------------------------------------
    // [修改] 任务队列的类型从
    //   packaged_task<ProcessResult()>
    // 改为
    //   packaged_task<ProcessResult(shared_ptr<Yolov5s>)>
    //
    // 原因：
    // 原来任务里自己选 yolo（按帧号），worker 只是执行任务。
    // 新设计：任务只携带帧数据，执行时由 worker 把自己专属的 yolo
    // 作为参数注入进来。这样每个 yolo 实例始终只被一个 worker 使用，
    // 彻底消除多线程对同一 rknn_context 的并发访问。
    // -------------------------------------------------------
    std::queue<std::packaged_task<ProcessResult(std::shared_ptr<Yolov5s>)>> tasks;

    std::mutex              queue_mutex;
    std::condition_variable condition;

    std::vector<std::thread>              threads;
    std::atomic<bool>                     run_flag{true};

    // 每个 worker 对应一个 yolo 实例，下标与 worker id 一一对应
    std::vector<std::shared_ptr<Yolov5s>> yolo_group;
};

#endif