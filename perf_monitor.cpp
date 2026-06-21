/**
 * @file perf_monitor.cpp
 * @brief RK3588 NPU + CPU 利用率监控模块实现
 *
 * NPU 利用率读取路径：/sys/kernel/debug/rknpu/load
 * CPU 利用率读取路径：/proc/stat
 *
 * 编译时需加到 CMakeLists.txt 的 add_executable 中：
 *     perf_monitor.cpp
 */

#include "perf_monitor.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstring>
#include <iomanip>

// ============================================================
// 单例
// ============================================================
PerfMonitor& PerfMonitor::instance()
{
    static PerfMonitor inst;
    return inst;
}

PerfMonitor::~PerfMonitor()
{
    if (running_) {
        stop();
    }
}

// ============================================================
// start / stop
// ============================================================
void PerfMonitor::start(const std::string& log_path, int interval_ms)
{
    if (running_) return;

    log_path_     = log_path;
    interval_ms_  = interval_ms;

    // 记录启动时刻
    start_time_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    samples_.clear();
    samples_.reserve(4096);  // 预留足够空间，避免频繁分配

    running_ = true;
    thread_  = std::thread(&PerfMonitor::monitor_loop, this);

    std::cout << "[PerfMonitor] 启动监控，采样间隔 " << interval_ms_ << "ms"
              << "，日志路径：" << log_path_ << std::endl;
}

void PerfMonitor::stop()
{
    if (!running_) return;
    running_ = false;

    if (thread_.joinable()) {
        thread_.join();
    }

    flush_log();
    std::cout << "[PerfMonitor] 监控已停止，共采样 " << samples_.size()
              << " 条，日志已保存到：" << log_path_ << std::endl;
}

// ============================================================
// 监控主循环
// ============================================================
void PerfMonitor::monitor_loop()
{
    // 用于差分计算 CPU 利用率，先读一次基准
    std::vector<CpuStat> prev_stats;
    read_cpu_stats(prev_stats);

    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
        if (!running_) break;

        // --- 采 CPU ---
        std::vector<CpuStat> curr_stats;
        if (!read_cpu_stats(curr_stats)) {
            continue;
        }

        PerfSample s{};
        s.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count()
            - start_time_ms_;

        // 第 0 项是 "cpu"（汇总），后续依次是 cpu0/cpu1/...
        if (!prev_stats.empty() && !curr_stats.empty()) {
            s.cpu_total = calc_usage(prev_stats[0], curr_stats[0]);
            for (size_t i = 1; i < curr_stats.size() && i < prev_stats.size(); ++i) {
                s.cpu_cores.push_back(calc_usage(prev_stats[i], curr_stats[i]));
            }
        }
        prev_stats = curr_stats;

        // --- 采 NPU ---
        s.npu_available = read_npu_load(s.npu_core);

        samples_.push_back(std::move(s));
    }
}

// ============================================================
// 读 /proc/stat
// ============================================================
bool PerfMonitor::read_cpu_stats(std::vector<CpuStat>& out)
{
    std::ifstream ifs("/proc/stat");
    if (!ifs.is_open()) return false;

    out.clear();
    std::string line;
    while (std::getline(ifs, line)) {
        // 匹配 "cpu" 开头的行（"cpu " 为总计，"cpu0"/"cpu1"... 为每核）
        if (line.substr(0, 3) != "cpu") break;
        if (line.size() < 4) continue;
        // "cpu " 后面是数字，"cpu0" 后面也是数字
        if (line[3] != ' ' && (line[3] < '0' || line[3] > '9')) continue;

        CpuStat st{};
        // 跳过行首 "cpu" 或 "cpu0" 等标签
        std::istringstream ss(line);
        std::string label;
        ss >> label
           >> st.user >> st.nice >> st.system >> st.idle
           >> st.iowait >> st.irq >> st.softirq >> st.steal;
        out.push_back(st);
    }
    return !out.empty();
}

float PerfMonitor::calc_usage(const CpuStat& prev, const CpuStat& curr)
{
    uint64_t total_delta  = curr.total()  - prev.total();
    uint64_t active_delta = curr.active() - prev.active();
    if (total_delta == 0) return 0.0f;
    return static_cast<float>(active_delta) / static_cast<float>(total_delta) * 100.0f;
}

// ============================================================
// 读 /sys/kernel/debug/rknpu/load
// ============================================================
bool PerfMonitor::read_npu_load(float core[3])
{
    // 路径（RK3588 debugfs）
    static const char* NPU_LOAD_PATH = "/sys/kernel/debug/rknpu/load";

    std::ifstream ifs(NPU_LOAD_PATH);
    if (!ifs.is_open()) {
        // 无权限或不存在时静默失败，填 -1 表示不可用
        core[0] = core[1] = core[2] = -1.0f;
        return false;
    }

    std::string line;
    std::getline(ifs, line);
    // 期望格式：NPU load:  Core0:  0%, Core1:  0%, Core2:  0%,
    // 解析方式：找每个 "CoreN:" 后面的数字
    core[0] = core[1] = core[2] = 0.0f;

    for (int i = 0; i < 3; ++i) {
        std::string key = "Core" + std::to_string(i) + ":";
        size_t pos = line.find(key);
        if (pos == std::string::npos) continue;
        pos += key.size();
        // 跳过空格
        while (pos < line.size() && line[pos] == ' ') ++pos;
        // 读数字直到 '%'
        size_t end = line.find('%', pos);
        if (end == std::string::npos) continue;
        try {
            core[i] = std::stof(line.substr(pos, end - pos));
        } catch (...) {
            core[i] = 0.0f;
        }
    }
    return true;
}

// ============================================================
// 将采样数据写入 CSV
// ============================================================
void PerfMonitor::flush_log()
{
    if (samples_.empty()) return;

    std::ofstream ofs(log_path_);
    if (!ofs.is_open()) {
        std::cerr << "[PerfMonitor] 无法打开日志文件：" << log_path_ << std::endl;
        return;
    }

    // --- 确定最大核心数 ---
    size_t max_cores = 0;
    for (const auto& s : samples_) {
        max_cores = std::max(max_cores, s.cpu_cores.size());
    }

    // --- 写 CSV 表头 ---
    ofs << "timestamp_ms,cpu_total%";
    for (size_t i = 0; i < max_cores; ++i) {
        ofs << ",cpu" << i << "%";
    }
    ofs << ",npu_core0%,npu_core1%,npu_core2%,npu_available\n";

    // --- 写数据行 ---
    ofs << std::fixed << std::setprecision(1);
    for (const auto& s : samples_) {
        ofs << s.timestamp_ms << ","
            << s.cpu_total;

        for (size_t i = 0; i < max_cores; ++i) {
            ofs << ",";
            if (i < s.cpu_cores.size()) {
                ofs << s.cpu_cores[i];
            } else {
                ofs << "0.0";
            }
        }

        if (s.npu_available) {
            ofs << "," << s.npu_core[0]
                << "," << s.npu_core[1]
                << "," << s.npu_core[2]
                << ",1\n";
        } else {
            ofs << ",-1,-1,-1,0\n";
        }
    }

    ofs.close();
}