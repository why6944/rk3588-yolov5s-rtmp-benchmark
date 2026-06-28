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
    void record_rknn_detail(long long input_set_us, long long run_us, long long outputs_get_us);
    void record_postprocess_detail(long long decode_us, long long sort_us, long long nms_us, long long result_us, int valid_count, int result_count);
    void record_draw(long long draw_us);
    void record_write(long long write_us);
    void record_mpp(long long mpp_us);

    void print_summary() const;
    bool append_csv(const std::string& path) const;
    bool append_detail_csv(const std::string& path) const;

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
    std::atomic<long long> rknn_input_set_us_{0};
    std::atomic<long long> rknn_run_us_{0};
    std::atomic<long long> rknn_outputs_get_us_{0};
    std::atomic<long long> post_decode_us_{0};
    std::atomic<long long> post_sort_us_{0};
    std::atomic<long long> post_nms_us_{0};
    std::atomic<long long> post_result_us_{0};
    std::atomic<long long> post_valid_count_{0};
    std::atomic<long long> post_result_count_{0};

    std::atomic<long long> draw_frames_{0};
    std::atomic<long long> draw_us_{0};

    std::atomic<long long> write_frames_{0};
    std::atomic<long long> write_us_{0};

    std::atomic<long long> mpp_frames_{0};
    std::atomic<long long> mpp_us_{0};
};

#endif
