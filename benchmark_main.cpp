#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <pthread.h>
#include "post_process_common.h"
#include "post_process_neon.h"

extern int process_scalar(int8_t*, float*, int, int, int, int, int,
                           std::vector<float>&, std::vector<float>&, std::vector<int>&,
                           float, int32_t, float);
extern int process_auto(int8_t*, float*, int, int, int, int, int,
                         std::vector<float>&, std::vector<float>&, std::vector<int>&,
                         float, int32_t, float);

struct Int8Frame {
    std::vector<int8_t> out0, out1, out2;
    int32_t zp[3];
    float scale[3];
};

static bool load_frame(const char *path, Int8Frame &f) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;
    uint32_t sizes[3];
    if (fread(sizes, 4, 3, fp) != 3) { fclose(fp); return false; }
    if (fread(f.zp, 4, 3, fp) != 3) { fclose(fp); return false; }
    if (fread(f.scale, 4, 3, fp) != 3) { fclose(fp); return false; }
    f.out0.resize(sizes[0]); f.out1.resize(sizes[1]); f.out2.resize(sizes[2]);
    fread(f.out0.data(), 1, sizes[0], fp);
    fread(f.out1.data(), 1, sizes[1], fp);
    fread(f.out2.data(), 1, sizes[2], fp);
    fclose(fp);
    return true;
}

static bool pin_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) == 0;
}

int main(int argc, char **argv) {
    const char *mode = "scalar";
    const char *data_dir = "int8_dumps";
    int nframes = 30, warmup = 100, iterations = 1000, rounds = 5, core = 4;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--postprocess") && i + 1 < argc) mode = argv[++i];
        else if (!strcmp(argv[i], "--data-dir") && i + 1 < argc) data_dir = argv[++i];
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc) nframes = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iterations") && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--core") && i + 1 < argc) core = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--help")) {
            printf("Usage: %s --postprocess scalar|auto|neon [--frames 30] [--warmup 100] [--iterations 1000] [--rounds 5] [--core 4]\n", argv[0]);
            return 0;
        }
    }

    // Load frames
    std::vector<Int8Frame> frames;
    for (int i = 0; i < nframes; i++) {
        char path[256];
        snprintf(path, sizeof(path), "%s/frame_%04d.bin", data_dir, i);
        Int8Frame f;
        if (!load_frame(path, f)) { fprintf(stderr, "Failed to load %s\n", path); return 1; }
        frames.push_back(std::move(f));
    }
    printf("Loaded %d frames from %s/\n", (int)frames.size(), data_dir);

    process_func_t fn = process_scalar;
    const char *mode_name = "scalar";
    if (!strcmp(mode, "auto")) { fn = process_auto; mode_name = "auto"; }
    else if (!strcmp(mode, "neon")) { fn = process_neon; mode_name = "neon"; }

    printf("Mode: %s, Warmup: %d, Iterations: %d, Rounds: %d, Core: %d\n",
           mode_name, warmup, iterations, rounds, core);

    if (!pin_to_core(core)) fprintf(stderr, "Warning: pin to core %d failed\n", core);

    // Warmup
    detect_result_group_t grp;
    post_process_timing_t timing;
    std::vector<int32_t> zps(3);
    std::vector<float> scales(3);
    for (int w = 0; w < warmup; w++) {
        auto &f = frames[w % frames.size()];
        zps = {f.zp[0], f.zp[1], f.zp[2]};
        scales = {f.scale[0], f.scale[1], f.scale[2]};
        post_process_common(f.out0.data(), f.out1.data(), f.out2.data(),
                            640, 640, 0.5f, 0.5f, 1.0f, 1.0f, zps, scales, grp, &timing, fn);
    }

    // Benchmark rounds
    printf("%-6s %10s %10s %10s %10s %10s\n", "Round", "Mean(us)", "Median(us)", "Min(us)", "Max(us)", "StdDev(us)");
    std::vector<double> round_means;
    for (int r = 0; r < rounds; r++) {
        std::vector<double> times; times.reserve(iterations);
        for (int iter = 0; iter < iterations; iter++) {
            auto &f = frames[iter % frames.size()];
            zps = {f.zp[0], f.zp[1], f.zp[2]};
            scales = {f.scale[0], f.scale[1], f.scale[2]};
            auto start = std::chrono::high_resolution_clock::now();
            post_process_common(f.out0.data(), f.out1.data(), f.out2.data(),
                                640, 640, 0.5f, 0.5f, 1.0f, 1.0f, zps, scales, grp, &timing, fn);
            auto end = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        }
        std::sort(times.begin(), times.end());
        double sum = std::accumulate(times.begin(), times.end(), 0.0);
        double mean = sum / times.size();
        double var = 0;
        for (double t : times) var += (t - mean) * (t - mean);
        var /= times.size();
        printf("%-6d %10.1f %10.1f %10.1f %10.1f %10.1f\n",
               r + 1, mean, times[times.size() / 2], times.front(), times.back(), std::sqrt(var));
        round_means.push_back(mean);
        // Also print timing breakdown for first round
        if (r == 0) {
            printf("  breakdown: decode=%lld sort=%lld nms=%lld result=%lld us, valid=%d result=%d\n",
                   timing.decode_us, timing.sort_us, timing.nms_us, timing.result_us,
                   timing.valid_count, timing.result_count);
        }
    }

    double overall = std::accumulate(round_means.begin(), round_means.end(), 0.0) / round_means.size();
    printf("Overall: %.1f us (avg across %d rounds)\n", overall, rounds);
    return 0;
}
