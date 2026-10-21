#ifndef LATENCY_MEASUREMENT_H
#define LATENCY_MEASUREMENT_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <time.h>

inline int64_t latencyMonotonicNowNs()
{
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

struct FrameLatencyTrace {
    int frame_index = -1;
    int64_t v4l2_timestamp_ns = -1;
    bool v4l2_timestamp_monotonic = false;
    int64_t dqbuf_ns = -1;
    int64_t read_enqueue_ns = -1;
    int64_t aggregator_dequeue_ns = -1;
    int64_t worker_start_ns = -1;
    int64_t inference_end_ns = -1;
    int64_t aggregator_ready_ns = -1;
    int64_t write_dequeue_ns = -1;
    int64_t rga_start_ns = -1;
    int64_t rga_end_ns = -1;
    int64_t mpp_submit_ns = -1;
    int64_t mpp_complete_ns = -1;
    bool mpp_success = false;
};

class LatencyCollector {
public:
    void configure(const std::string &csv_path, int warmup_frames, bool summary, int expected_frames)
    {
        csv_path_ = csv_path;
        warmup_frames_ = std::max(0, warmup_frames);
        summary_ = summary;
        expected_frames_ = std::max(0, expected_frames);
        records_.clear();
    }

    bool enabled() const { return summary_ || !csv_path_.empty(); }

    void record(const FrameLatencyTrace &trace)
    {
        if(enabled()) records_.push_back(trace);
    }

    void finalize() const
    {
        if(!enabled()) return;
        if(!csv_path_.empty()) writeCsv();
        if(summary_) printSummary();
    }

private:
    static double intervalMs(int64_t start_ns, int64_t end_ns)
    {
        if(start_ns < 0 || end_ns < start_ns)
            return std::numeric_limits<double>::quiet_NaN();
        return static_cast<double>(end_ns - start_ns) / 1000000.0;
    }

    static std::string csvMs(double value)
    {
        if(!std::isfinite(value)) return "";
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(3) << value;
        return stream.str();
    }

    bool inStatistics(const FrameLatencyTrace &trace) const
    {
        return trace.frame_index >= warmup_frames_ && trace.mpp_success &&
               std::isfinite(intervalMs(trace.dqbuf_ns, trace.mpp_complete_ns));
    }

    std::vector<double> samplesFor(double (*extract)(const FrameLatencyTrace &)) const
    {
        std::vector<double> values;
        for(const auto &trace : records_) {
            if(!inStatistics(trace)) continue;
            const double value = extract(trace);
            if(std::isfinite(value)) values.push_back(value);
        }
        std::sort(values.begin(), values.end());
        return values;
    }

    static double totalLatency(const FrameLatencyTrace &trace)
    {
        return intervalMs(trace.dqbuf_ns, trace.mpp_complete_ns);
    }
    static double readQueueLatency(const FrameLatencyTrace &trace)
    {
        return intervalMs(trace.read_enqueue_ns, trace.worker_start_ns);
    }
    static double inferenceLatency(const FrameLatencyTrace &trace)
    {
        return intervalMs(trace.worker_start_ns, trace.inference_end_ns);
    }
    static double reorderWriteLatency(const FrameLatencyTrace &trace)
    {
        return intervalMs(trace.inference_end_ns, trace.write_dequeue_ns);
    }
    static double rgaLatency(const FrameLatencyTrace &trace)
    {
        return intervalMs(trace.rga_start_ns, trace.rga_end_ns);
    }
    static double mppLatency(const FrameLatencyTrace &trace)
    {
        return intervalMs(trace.mpp_submit_ns, trace.mpp_complete_ns);
    }

    static double percentile(const std::vector<double> &values, double fraction)
    {
        if(values.empty()) return std::numeric_limits<double>::quiet_NaN();
        const double position = fraction * static_cast<double>(values.size() - 1);
        const size_t lower = static_cast<size_t>(position);
        const size_t upper = std::min(lower + 1, values.size() - 1);
        const double weight = position - static_cast<double>(lower);
        return values[lower] + (values[upper] - values[lower]) * weight;
    }

    static double mean(const std::vector<double> &values)
    {
        if(values.empty()) return std::numeric_limits<double>::quiet_NaN();
        double sum = 0.0;
        for(double value : values) sum += value;
        return sum / static_cast<double>(values.size());
    }

    static void printMetric(const char *name, const std::vector<double> &values)
    {
        if(values.empty()) {
            std::cout << "[Latency] " << name << ": unavailable\n";
            return;
        }
        std::cout << std::fixed << std::setprecision(3)
                  << "[Latency] " << name
                  << " mean=" << mean(values)
                  << " p50=" << percentile(values, 0.50)
                  << " p90=" << percentile(values, 0.90)
                  << " p95=" << percentile(values, 0.95)
                  << " p99=" << percentile(values, 0.99)
                  << " max=" << values.back() << " ms\n";
    }

    void writeCsv() const
    {
        std::ofstream output(csv_path_);
        if(!output) {
            std::cerr << "[Latency] cannot write CSV: " << csv_path_ << "\n";
            return;
        }
        output << "frame_index,in_statistics,mpp_success,v4l2_timestamp_valid,"
               << "v4l2_timestamp_ns,dqbuf_ns,read_enqueue_ns,aggregator_dequeue_ns,"
               << "worker_start_ns,inference_end_ns,aggregator_ready_ns,write_dequeue_ns,"
               << "rga_start_ns,rga_end_ns,mpp_submit_ns,mpp_complete_ns,"
               << "dqbuf_to_mpp_ms,v4l2_to_mpp_ms,read_queue_ms,inference_ms,"
               << "reorder_write_queue_ms,rga_ms,mpp_ms\n";
        for(const auto &trace : records_) {
            const double driver_to_mpp = trace.v4l2_timestamp_monotonic
                ? intervalMs(trace.v4l2_timestamp_ns, trace.mpp_complete_ns)
                : std::numeric_limits<double>::quiet_NaN();
            output << trace.frame_index << ',' << (inStatistics(trace) ? 1 : 0) << ','
                   << (trace.mpp_success ? 1 : 0) << ','
                   << (trace.v4l2_timestamp_monotonic ? 1 : 0) << ','
                   << trace.v4l2_timestamp_ns << ',' << trace.dqbuf_ns << ','
                   << trace.read_enqueue_ns << ',' << trace.aggregator_dequeue_ns << ','
                   << trace.worker_start_ns << ',' << trace.inference_end_ns << ','
                   << trace.aggregator_ready_ns << ',' << trace.write_dequeue_ns << ','
                   << trace.rga_start_ns << ',' << trace.rga_end_ns << ','
                   << trace.mpp_submit_ns << ',' << trace.mpp_complete_ns << ','
                   << csvMs(totalLatency(trace)) << ',' << csvMs(driver_to_mpp) << ','
                   << csvMs(readQueueLatency(trace)) << ',' << csvMs(inferenceLatency(trace)) << ','
                   << csvMs(reorderWriteLatency(trace)) << ',' << csvMs(rgaLatency(trace)) << ','
                   << csvMs(mppLatency(trace)) << '\n';
        }
        std::cout << "[Latency] CSV written: " << csv_path_
                  << " (records=" << records_.size() << ")\n";
    }

    void printSummary() const
    {
        size_t success = 0;
        size_t failed = 0;
        for(const auto &trace : records_) {
            if(trace.frame_index < warmup_frames_) continue;
            if(trace.mpp_success) ++success;
            else ++failed;
        }
        std::cout << "[Latency] expected_frames=" << expected_frames_
                  << ", warmup_frames=" << warmup_frames_
                  << ", encoded_after_warmup=" << success
                  << ", failed_after_warmup=" << failed
                  << ", missing_after_warmup="
                  << std::max(0, expected_frames_ - warmup_frames_ - static_cast<int>(success + failed))
                  << '\n';
        printMetric("dqbuf_to_mpp", samplesFor(totalLatency));
        printMetric("read_queue", samplesFor(readQueueLatency));
        printMetric("inference", samplesFor(inferenceLatency));
        printMetric("reorder_write_queue", samplesFor(reorderWriteLatency));
        printMetric("rga", samplesFor(rgaLatency));
        printMetric("mpp", samplesFor(mppLatency));
    }

    std::string csv_path_;
    int warmup_frames_ = 0;
    bool summary_ = false;
    int expected_frames_ = 0;
    std::vector<FrameLatencyTrace> records_;
};

#endif
