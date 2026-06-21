#ifndef BENCHMARK_STATS_H
#define BENCHMARK_STATS_H

#include <atomic>
#include <mutex>
#include <string>

class BenchmarkStats
{
public:
    static BenchmarkStats& instance();

    void reset(int thread_count, const std::string& mode);
    void set_total_elapsed_ms(long long elapsed_ms);

    void record_inference(long long preprocess_us, long long rknn_us, long long postprocess_us);
    void record_draw(long long draw_us);
    void record_write(long long write_us);
    void record_mpp(long long mpp_us);

    void print_summary() const;
    bool append_csv(const std::string& path) const;

private:
    BenchmarkStats() = default;

    static double avg_ms(long long total_us, long long count);

    mutable std::mutex meta_mutex_;
    int thread_count_ = 0;
    std::string mode_ = "full";
    long long total_elapsed_ms_ = 0;

    std::atomic<long long> inference_frames_{0};
    std::atomic<long long> preprocess_us_{0};
    std::atomic<long long> rknn_us_{0};
    std::atomic<long long> postprocess_us_{0};

    std::atomic<long long> draw_frames_{0};
    std::atomic<long long> draw_us_{0};

    std::atomic<long long> write_frames_{0};
    std::atomic<long long> write_us_{0};

    std::atomic<long long> mpp_frames_{0};
    std::atomic<long long> mpp_us_{0};
};

#endif
