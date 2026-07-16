#include "thread_poll.h"
#include "debug_log.h"

ThreadPoll::ThreadPoll(const char* model_path, int num_threads, bool draw_results)
{
    run_flag = true;
    draw_results_ = draw_results;
    initialized_ = init(model_path, num_threads);
}

ThreadPoll::~ThreadPoll()
{
    LOG_DEBUG("Remaining tasks: %zu\n", tasks.size());

    run_flag = false;
    condition.notify_all();

    for(auto& t : threads)
    {
        if(t.joinable())
            t.join();
    }
    LOG_DEBUG("ThreadPoll destroyed.\n");
}

bool ThreadPoll::init(const char* model_path, int num_threads)
{
    if(num_threads <= 0) num_threads = 1;

    // 第一个 yolo 正常加载模型权重，后续 yolo 用 rknn_dup_context 共享权重。
    // 每个 worker 仍然独占自己的 rknn_context，再分别绑定到 RK3588 三个 NPU core。
    rknn_context* shared_context = nullptr;
    for(int i = 0; i < num_threads; i++)
    {
        auto yolo = std::make_shared<Yolov5s>(model_path, i % 3, shared_context);
        if(!yolo->isInitialized())
        {
            std::cerr << "[ThreadPoll] failed to initialize RKNN worker " << i
                      << "; stop before starting worker threads.\n";
            yolo_group.clear();
            run_flag = false;
            return false;
        }
        if(i == 0)
            shared_context = yolo->get_context_ptr();
        yolo_group.emplace_back(yolo);
    }

    // 启动 num_threads 个工作线程
    for(int i = 0; i < num_threads; i++)
    {
        threads.emplace_back(&ThreadPoll::worker, this, i);
    }
    return true;
}

void ThreadPoll::worker(int id)
{
    // -------------------------------------------------------
    // [修改] worker 现在真正使用自己专属的 yolo 实例
    //
    // 原来的问题：
    //   第56行声明了 yolo = yolo_group[id]，但后面从来没用到它。
    //   实际执行的 current_task() 内部按帧号选 yolo，
    //   导致不同 worker 可能同时操作同一个 yolo 实例（数据竞争）。
    //
    // 新的做法：
    //   任务本身不再选 yolo，worker 在调用任务时把自己的 my_yolo
    //   作为参数传进去：current_task(my_yolo)
    //   这样 my_yolo 永远只有这一个 worker 在用，没有并发冲突。
    // -------------------------------------------------------
    std::shared_ptr<Yolov5s> my_yolo = yolo_group[id];  // 专属 yolo，真正被使用
    LOG_DEBUG("worker线程启动, id=%d\n", id);

    while(run_flag)
    {
        // [修改] 任务类型从 packaged_task<ProcessResult()>
        // 改为 packaged_task<ProcessResult(shared_ptr<Yolov5s>)>
        std::packaged_task<ProcessResult(std::shared_ptr<Yolov5s>)> current_task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            condition.wait(lock, [this]
            {
                return (!tasks.empty() || !run_flag);
            });

            if(!run_flag)
            {
                LOG_DEBUG("worker %d 下班！\n", id);
                break;
            }

            current_task = std::move(tasks.front());
            tasks.pop();
        }

        if(current_task.valid())
        {
            LOG_DEBUG("worker %d get task！\n", id);
            // [修改] 把自己的 my_yolo 作为参数注入任务
            // 原来是 current_task()，任务自己选 yolo
            // 现在是 current_task(my_yolo)，由 worker 决定用哪个 yolo
            current_task(my_yolo);
        }
    }
    LOG_DEBUG("Worker %d exited, remaining tasks: %zu\n", id, tasks.size());
}

bool ThreadPoll::isInitialized() const
{
    return initialized_;
}

std::future<ProcessResult> ThreadPoll::submit_task_async(FrameData frame_data)
{
    if(!initialized_)
    {
        std::promise<ProcessResult> promise;
        ProcessResult result;
        result.error_msg = "ThreadPoll is not initialized";
        promise.set_value(std::move(result));
        return promise.get_future();
    }

    bool draw_results = draw_results_;
    std::packaged_task<ProcessResult(std::shared_ptr<Yolov5s>)> task(
        [frame_data = std::move(frame_data), draw_results](std::shared_ptr<Yolov5s> yolo) mutable
        {
            ProcessResult result;
            try
            {
                LOG_DEBUG("worker get task %d\n", frame_data.index);

                if(!frame_data.hasMat() && !frame_data.hasDmaBuf())
                {
                    result.error_msg = "FrameData has no cv::Mat or DMA-BUF for current processing path";
                    result.success = false;
                    return result;
                }

                detect_result_group_t detections;
                int ret = yolo->inference_frame(frame_data, detections);
                if(ret != 0)
                {
                    result.error_msg = "inference_frame failed";
                    result.success = false;
                    return result;
                }
                if(draw_results && frame_data.hasMat())
                    yolo->draw_result(frame_data.frame, detections);

                result.frame_data         = std::move(frame_data);
                result.detection_results  = detections;
                result.success            = true;
            }
            catch(const std::exception& e)
            {
                result.error_msg = e.what();
                result.success   = false;
            }
            return result;
        }
    );

    std::future<ProcessResult> future = task.get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        tasks.emplace(std::move(task));
        LOG_DEBUG("[submit_task_async] pushed task, queue size=%zu\n", tasks.size());
    }
    condition.notify_one();
    return future;
}
