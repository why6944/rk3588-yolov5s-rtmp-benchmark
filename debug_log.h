#ifndef DEBUG_LOG_H
#define DEBUG_LOG_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

extern int g_verbose_log;
extern float g_box_threshold;
extern float g_nms_threshold;

#ifdef __cplusplus
}
#endif

#define LOG_DEBUG(...) \
    do { \
        if (g_verbose_log) { \
            fprintf(stdout, __VA_ARGS__); \
        } \
    } while (0)

#define LOG_DEBUG_FLUSH() \
    do { \
        if (g_verbose_log) { \
            fflush(stdout); \
        } \
    } while (0)

#endif
