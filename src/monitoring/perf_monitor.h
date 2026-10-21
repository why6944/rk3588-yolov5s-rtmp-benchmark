#ifndef PERF_MONITOR_H
#define PERF_MONITOR_H

/**
 * @file perf_monitor.h
 * @brief RK3588 NPU + CPU 利用率监控模块
 *
 * 使用方式（最简两行接入）：
 *   PerfMonitor::instance().start("perf_log.csv");  // 程序开始时
 *   PerfMonitor::instance().stop();                  // 程序结束时
 *
 * NPU 利用率来源：/sys/kernel/debug/rknpu/load  （需要 root 或 debugfs 权限）
 * CPU 利用率来源：/proc/stat
 *
 * 输出 CSV 格式：
 *   timestamp_ms, cpu_total%, cpu0%, cpu1%, ..., npu_core0%, npu_core1%, npu_core2%
 */

#include <string>
#include <thread>
#include <atomic>
#include <vector>
#include <cstdint>

struct PerfSample {
    int64_t  timestamp_ms;   // 相对程序启动的毫秒数

    // CPU（全局 + 每核心），单位 %，精度 0.1%
    float    cpu_total;
    std::vector<float> cpu_cores;  // 每核心利用率

    // NPU 三核心，单位 %
    float    npu_core[3];
    bool     npu_available;  // debugfs 是否可读
};

class PerfMonitor {
public:
    /**
     * 获取单例
     */
    static PerfMonitor& instance();

    /**
     * 启动监控
     * @param log_path    CSV 日志文件路径，默认 "perf_log.csv"
     * @param interval_ms 采样间隔毫秒，默认 500ms
     */
    void start(const std::string& log_path = "perf_log.csv",
               int interval_ms = 500);

    /**
     * 停止监控，将所有采样数据刷写到日志文件
     * （程序退出前务必调用，或使用 RAII 析构自动调用）
     */
    void stop();

    /**
     * 析构时自动 stop（保证程序意外退出时也能保存数据）
     */
    ~PerfMonitor();

    // 禁止拷贝
    PerfMonitor(const PerfMonitor&)            = delete;
    PerfMonitor& operator=(const PerfMonitor&) = delete;

private:
    PerfMonitor() = default;

    void monitor_loop();
    PerfSample collect_sample();

    // CPU 利用率计算辅助
    struct CpuStat {
        uint64_t user, nice, system, idle, iowait, irq, softirq, steal;
        uint64_t total()  const { return user+nice+system+idle+iowait+irq+softirq+steal; }
        uint64_t active() const { return user+nice+system+irq+softirq+steal; }
    };
    bool read_cpu_stats(std::vector<CpuStat>& out);
    float calc_usage(const CpuStat& prev, const CpuStat& curr);
    bool read_npu_load(float core[3]);
    void flush_log();

    std::string              log_path_;
    int                      interval_ms_ = 500;
    std::atomic<bool>        running_{false};
    std::thread              thread_;
    std::vector<PerfSample>  samples_;

    // 用于记录程序启动时刻
    int64_t start_time_ms_ = 0;
};

#endif /* PERF_MONITOR_H */