#include "thread_poll.h"
#include "debug_log.h"

ThreadPoll::ThreadPoll(const char* model_path, int num_threads, bool draw_results)
{
    run_flag = true;
    draw_results_ = draw_results;
    init(model_path, num_threads);
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

void ThreadPoll::init(const char* model_path, int num_threads)
{
    if(num_threads <= 0) num_threads = 1;

    // 创建 num_threads 个 yolo 实例（逻辑不变，i % 3 保证 npu_index 在 0/1/2 内）
    for(int i = 0; i < num_threads; i++)
    {
        auto yolo = std::make_shared<Yolov5s>(model_path, i % 3);
        yolo_group.emplace_back(yolo);
    }

    // 启动 num_threads 个工作线程
    for(int i = 0; i < num_threads; i++)
    {
        threads.emplace_back(&ThreadPoll::worker, this, i);
    }
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

std::future<ProcessResult> ThreadPoll::submit_task_async(int index, cv::Mat img)
{
    // -------------------------------------------------------
    // [修改] lambda 的参数从无参数改为接收一个 yolo 实例
    //
    // 原来：
    //   packaged_task<ProcessResult()> task([this, index, img]() {
    //       auto yolo = yolo_group[index % yolo_group.size()]; // 按帧号选
    //       ...
    //   });
    //
    // 新的：
    //   packaged_task<ProcessResult(shared_ptr<Yolov5s>)> task([index, img](shared_ptr<Yolov5s> yolo) {
    //       // yolo 由调用方（worker）传入，不在这里选
    //       ...
    //   });
    //
    // 关键变化：lambda 不再捕获 this，不再访问 yolo_group，
    // yolo 由 worker 调用时注入，谁执行任务谁决定用哪个 yolo。
    // -------------------------------------------------------
    bool draw_results = draw_results_;
    std::packaged_task<ProcessResult(std::shared_ptr<Yolov5s>)> task(
        [index, img = std::move(img), draw_results](std::shared_ptr<Yolov5s> yolo) mutable
        {
            ProcessResult result;
            try
            {
                LOG_DEBUG("worker get task %d！\n", index);

                detect_result_group_t detections;
                yolo->inference_image(img, detections);
                if(draw_results)
                    yolo->draw_result(img, detections);

                result.processed_img       = std::move(img);
                result.detection_results   = detections;
                result.success             = true;
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
        LOG_DEBUG("[submit_task_async] 已压入tasks队列, 现在大小=%zu\n", tasks.size());
    }
    condition.notify_one();
    return future;
}