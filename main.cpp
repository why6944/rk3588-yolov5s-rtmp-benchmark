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
#include <chrono>
#include <future>
#include <map>
#include <memory>

#include "SafeQueue.h"
#include "yolov5s.h"
#include "thread_poll.h"
#include "streamer.h"

// RGA头文件包含顺序
#include "rga.h"
#include "drmrga.h"
#include "im2d.h"
#include "RgaUtils.h"

void BGR_to_NV12_with_RGA(uint8_t *bgr, uint8_t *nv12, int width, int height, int channels) {
    printf("开始BGR到NV12的转换，图像尺寸: %dx%d\n", width, height);
    rga_buffer_handle_t bgr_handle, yuv_handle;

    memset(nv12, 0x00, width * height * 3 / 2);

    // 导入缓冲区
    bgr_handle = importbuffer_virtualaddr(bgr, width * height * 3);
    yuv_handle = importbuffer_virtualaddr(nv12, width * height * 3 / 2);

    if(bgr_handle == 0 || yuv_handle == 0)
    {
        printf("import va failed.\n");
    }

    // 定义rga缓冲区
    rga_buffer_t bgr_src = wrapbuffer_handle(bgr_handle, width, height, RK_FORMAT_BGR_888);
    rga_buffer_t yuv_src = wrapbuffer_handle(yuv_handle, width, height, RK_FORMAT_YCrCb_420_SP);
    
    // 执行转换
    int ret = imcheck(bgr_src, yuv_src, {}, {});
    if(ret != IM_STATUS_NOERROR)
    {
        printf("%d, imcheck error! %s\n", __LINE__,  imStrError((IM_STATUS)ret));
    }
    
    ret = imcvtcolor(bgr_src, yuv_src, RK_FORMAT_BGR_888, RK_FORMAT_YCrCb_420_SP);
    if(ret == IM_STATUS_SUCCESS)
    {
        printf("BGR888 TO NV12 OK!\n");
    }   
    else
    {
        printf("%d, cvtColor error! %s\n", __LINE__,  imStrError((IM_STATUS)ret));
    }

    if(bgr_handle)
    {
        releasebuffer_handle(bgr_handle);
    }
    if(yuv_handle)
    {
        releasebuffer_handle(yuv_handle);
    }

}

// 在函数外部或文件开头添加一个常量定义
const int MAX_CONCURRENT_FRAMES = 10; // 最大并发处理帧数，根据实际系统内存调整

//-----------------------------------
// 1) 定义一个存放帧和下标的结构
//-----------------------------------
struct FrameData {
    cv::Mat frame;
    int index;
};

// 全局队列 & 全局标志
SafeQueue<FrameData> g_readQueue(50);
SafeQueue<FrameData> g_writeQueue(50);
std::atomic<bool> g_readFinish(false);
std::atomic<bool> g_processFinish(false);

//-----------------------------------
// 2) 读线程：不断从视频文件读取放入 g_readQueue
//-----------------------------------
void readThreadFunc(cv::VideoCapture &cap) {
    int idx = 0;
    while(true)
    {
        cv::Mat frame;
        if(!cap.read(frame))
        {
            // 读不到帧了（到视频末尾或出错）
            std::cerr << "[ReadThread] read failed or EOF.\n";
            break;
        }
        // FrameData data{ frame.clone(), idx++ };
        FrameData data{ std::move(frame), idx++ };
        g_readQueue.enqueue(data);
        std::cout<<"读取队列中的图片数目目前是："<<g_readQueue.size()<<endl;
    }
    // 通知后续不再有新帧
    g_readFinish = true;
    std::cerr << "[ReadThread] finished.\n";
}

//-----------------------------------
// 3) 聚合线程：既提交多帧到线程池并行处理，也按顺序收集结果
//-----------------------------------
void aggregatorThreadFunc(ThreadPoll &npu_pool)
{
    // 用于按正确顺序写入的下标
    int nextWriteIndex = 0;

    // 存储"帧下标 -> future"的映射，实现无阻塞并行提交与按序收集
    std::map<int, std::future<ProcessResult>> tasks_inflight;

    while(true)
    {
        // 步骤A：批量尝试从 g_readQueue 获取新帧并提交到线程池
        //       控制并行度，避免内存溢出
        FrameData inputFD;
        if(!g_readQueue.empty() && tasks_inflight.size() < MAX_CONCURRENT_FRAMES)  // 限制并发任务数量
        {
            if(g_readQueue.dequeue(inputFD)) {
                // 提交异步推理任务
                auto fut = npu_pool.submit_task_async(inputFD.index, inputFD.frame);
                // 将 (index -> future) 存到映射
                tasks_inflight[inputFD.index] = std::move(fut);
                
                // 打印当前并发任务数量，便于监控
                if(inputFD.index % 1 == 0) {
                    std::cout << "当前并发任务数: " << tasks_inflight.size() 
                              << "，处理到第 " << inputFD.index << " 帧" << std::endl;
                }
            }
        }

        // 步骤B：检查是否有"下一个待写帧(nextWriteIndex)"已经推理完成
        //        如果完成，就把其结果按顺序放到 g_writeQueue
        auto it = tasks_inflight.find(nextWriteIndex);
        while(it != tasks_inflight.end())
        {
            // 不阻塞，先检查这条 future 是否ready
            auto status = it->second.wait_for(std::chrono::milliseconds(1));
            if(status == std::future_status::ready)
            {
                // 获取推理结果
                ProcessResult result = it->second.get();

                // 将推理后图像放到 g_writeQueue
                FrameData outputFD;
                outputFD.index = nextWriteIndex;
                outputFD.frame = result.processed_img.clone();
                g_writeQueue.enqueue(outputFD);

                // 移除映射并递增下一个待写index
                tasks_inflight.erase(it);
                cout<<"当前已经处理完成了："<<nextWriteIndex<<"帧图片，剩余任务："<<tasks_inflight.size()<<endl;
                nextWriteIndex++;

                // 继续尝试下一个
                it = tasks_inflight.find(nextWriteIndex);
            }
            else
            {
                // 下一个还没完成，就先退出等待，后面再检测
                break;
            }
        }

        // 步骤C：判断退出条件
        //   若读完了 && 读队列空了 && 当前映射也空了，就说明都处理完了
        if(g_readFinish && g_readQueue.empty() && tasks_inflight.empty())
        {
            cout<<"处理线程已经结束"<<endl;
            break;
        }

        // 如果当前任务已满，适当增加睡眠时间避免CPU空转
        if(tasks_inflight.size() >= MAX_CONCURRENT_FRAMES) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        } else {
            // 为避免CPU空转过高，可稍微睡一下
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // 设置处理完成标志
    g_processFinish = true;
    std::cerr << "[AggregatorThread] finished.\n";
}

//-----------------------------------
// 4) 写线程：从 g_writeQueue 中取出图像写到文件
//-----------------------------------
void writeThreadFunc(cv::VideoWriter &writer)
{
    // 使用智能指针管理内存
    std::unique_ptr<uint8_t[]> nv12_buffer;
    int width = 0;
    int height = 0;
    bool buffer_initialized = false;

    while(true)
    {
        if(g_processFinish && g_writeQueue.empty())
        {
            // 没有更多帧了
            break;
        }

        FrameData outputFD;
        if(!g_writeQueue.dequeue(outputFD)) {
            // 即使没取到，也要尝试继续，直到所有任务结束
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        if(!outputFD.frame.empty())
        {
            // 获取当前帧的尺寸
            int current_width = outputFD.frame.cols;
            int current_height = outputFD.frame.rows;
            int channels = outputFD.frame.channels();
            int nv12_size = current_width * current_height * 3 / 2;

            // 如果是第一次处理或尺寸发生变化，重新分配缓冲区
            if (!buffer_initialized || width != current_width || height != current_height) {
                try {
                    nv12_buffer = std::make_unique<uint8_t[]>(nv12_size);
                    width = current_width;
                    height = current_height;
                    buffer_initialized = true;
                } catch (const std::bad_alloc& e) {
                    std::cerr << "内存分配失败: " << e.what() << std::endl;
                    break;
                }
            }

            // 使用RGA进行BGR到NV12的转换
            BGR_to_NV12_with_RGA(outputFD.frame.data, nv12_buffer.get(), width, height, channels);
            
            // 使用NV12数据进行处理
            process_frame(nv12_buffer.get(), nv12_size);
        }
        cout<<"写入队列帧数："<<g_writeQueue.size()<<endl;
    }
    std::cerr << "[WriteThread] finished.\n";
}

//-----------------------------------
// 5) main 函数，把上述线程和线程池串起来
//-----------------------------------
int main()
{

    auto start = std::chrono::high_resolution_clock::now();

    std::string inPath  = "../video.mp4"; // 你的输入视频
    std::string outPath = "../output.avi";                       // 输出文件

    //rtmp的推流路径
    std::string rtmpPath = "rtmp://192.168.2.20:1935/live/app";

    // 打开输入视频
    cv::VideoCapture cap(inPath);
    if(!cap.isOpened())
    {
        std::cerr << "Fail to open input video: " << inPath << "\n";
        return -1;
    }

    // 获取视频属性
    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    double fps = cap.get(cv::CAP_PROP_FPS);
    if(fps < 1.0)  // 避免某些视频元数据不完整
        fps = 25.0;
    int bitrate = width * height / 8 * (fps / 1);

    int fourcc = cv::VideoWriter::fourcc('H','2','6','4');
    
    //bps换算kbps
    init_streamer(width, height, fps, bitrate, rtmpPath.c_str());

    // 打开输出视频
    cv::VideoWriter writer(outPath, fourcc, fps, cv::Size(width, height));
    if(!writer.isOpened())
    {
        std::cerr << "Fail to create output video: " << outPath << "\n";
        return -1;
    }

    // 创建 thread pool，让它开足核数（例如 12 worker）
    ThreadPoll npu_pool("../model/yolov5s.rknn", 3);

    // 启动：1) 读线程, 2) 聚合线程, 3) 写线程
    std::thread tRead(readThreadFunc, std::ref(cap));
    std::thread tAggregator(aggregatorThreadFunc, std::ref(npu_pool));
    std::thread tWrite(writeThreadFunc, std::ref(writer));

    // 等3个线程退出
    tRead.join();
    tAggregator.join();
    tWrite.join();

    // 给队列发 stop 信号（好习惯，但此时往往都空了）
    g_readQueue.stop();
    g_writeQueue.stop();

    writer.release();

    close_streamer();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "处理总用时：" << elapsed_ms.count() << " ms\n";
    std::cerr << "[Main] All done.\n";
    return 0;
}