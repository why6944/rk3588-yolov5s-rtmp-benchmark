#include "benchmark_stats.h"

#include <fstream>
#include <iomanip>
#include <iostream>

BenchmarkStats& BenchmarkStats::instance()
{
    static BenchmarkStats stats;
    return stats;
}

void BenchmarkStats::reset(int thread_count, const std::string& mode)
{
    std::lock_guard<std::mutex> lock(meta_mutex_);
    thread_count_ = thread_count;
    mode_ = mode;
    total_elapsed_ms_ = 0;

    inference_frames_ = 0;
    preprocess_us_ = 0;
    rknn_us_ = 0;
    postprocess_us_ = 0;
    draw_frames_ = 0;
    draw_us_ = 0;
    write_frames_ = 0;
    write_us_ = 0;
    mpp_frames_ = 0;
    mpp_us_ = 0;
}

void BenchmarkStats::set_total_elapsed_ms(long long elapsed_ms)
{
    std::lock_guard<std::mutex> lock(meta_mutex_);
    total_elapsed_ms_ = elapsed_ms;
}

void BenchmarkStats::record_inference(long long preprocess_us, long long rknn_us, long long postprocess_us)
{
    inference_frames_++;
    preprocess_us_ += preprocess_us;
    rknn_us_ += rknn_us;
    postprocess_us_ += postprocess_us;
}

void BenchmarkStats::record_draw(long long draw_us)
{
    draw_frames_++;
    draw_us_ += draw_us;
}

void BenchmarkStats::record_write(long long write_us)
{
    write_frames_++;
    write_us_ += write_us;
}

void BenchmarkStats::record_mpp(long long mpp_us)
{
    mpp_frames_++;
    mpp_us_ += mpp_us;
}

double BenchmarkStats::avg_ms(long long total_us, long long count)
{
    if (count <= 0) return 0.0;
    return static_cast<double>(total_us) / 1000.0 / static_cast<double>(count);
}

void BenchmarkStats::print_summary() const
{
    std::lock_guard<std::mutex> lock(meta_mutex_);
    const long long frames = inference_frames_.load();
    const double fps = total_elapsed_ms_ > 0
        ? static_cast<double>(frames) * 1000.0 / static_cast<double>(total_elapsed_ms_)
        : 0.0;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "[Benchmark] threads=" << thread_count_
              << ", mode=" << mode_
              << ", frames=" << frames
              << ", elapsed_ms=" << total_elapsed_ms_
              << ", end_to_end_fps=" << fps << std::endl;
    std::cout << "[Benchmark] avg_ms preprocess=" << avg_ms(preprocess_us_.load(), frames)
              << ", rknn=" << avg_ms(rknn_us_.load(), frames)
              << ", postprocess=" << avg_ms(postprocess_us_.load(), frames)
              << ", draw=" << avg_ms(draw_us_.load(), draw_frames_.load())
              << ", write_avi=" << avg_ms(write_us_.load(), write_frames_.load())
              << ", mpp_path=" << avg_ms(mpp_us_.load(), mpp_frames_.load())
              << std::endl;
}

bool BenchmarkStats::append_csv(const std::string& path) const
{
    std::lock_guard<std::mutex> lock(meta_mutex_);

    bool need_header = true;
    {
        std::ifstream ifs(path);
        need_header = !ifs.good() || ifs.peek() == std::ifstream::traits_type::eof();
    }

    std::ofstream ofs(path, std::ios::app);
    if (!ofs.is_open()) {
        std::cerr << "[Benchmark] failed to open csv: " << path << std::endl;
        return false;
    }

    if (need_header) {
        ofs << "threads,mode,frames,elapsed_ms,end_to_end_fps,"
            << "preprocess_avg_ms,rknn_avg_ms,postprocess_avg_ms,draw_avg_ms,write_avi_avg_ms,mpp_path_avg_ms\n";
    }

    const long long frames = inference_frames_.load();
    const double fps = total_elapsed_ms_ > 0
        ? static_cast<double>(frames) * 1000.0 / static_cast<double>(total_elapsed_ms_)
        : 0.0;

    ofs << std::fixed << std::setprecision(3)
        << thread_count_ << ','
        << mode_ << ','
        << frames << ','
        << total_elapsed_ms_ << ','
        << fps << ','
        << avg_ms(preprocess_us_.load(), frames) << ','
        << avg_ms(rknn_us_.load(), frames) << ','
        << avg_ms(postprocess_us_.load(), frames) << ','
        << avg_ms(draw_us_.load(), draw_frames_.load()) << ','
        << avg_ms(write_us_.load(), write_frames_.load()) << ','
        << avg_ms(mpp_us_.load(), mpp_frames_.load()) << '\n';

    return true;
}
