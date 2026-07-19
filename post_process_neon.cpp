/**
 * post_process_neon.cpp —— YOLOv5 后处理 objectness 扫描的手写 NEON 加速版
 *
 * ============================================================================
 * 优化策略：粗筛 + 精筛 两级过滤
 * ============================================================================
 *
 *  第一级（NEON 粗筛）:
 *    每次加载 16 个 int8_t objectness 值，用 NEON vcgeq_s8 并行比较。
 *    如果整组 16 个 cell 都低于阈值，vmaxvq_u8 返回 0，直接跳过。
 *    比较用的是 >=（保守策略），允许 conf==thres 的边界假阳性。
 *
 *  第二级（标量精筛）:
 *    只有粗筛通过时才进入，用标量 > 逐个判断，过滤掉假阳性。
 *    判断通过后的坐标解码、类别搜索全部沿用标量代码，
 *    保证与标量版逐 bit 一致。
 *
 * ============================================================================
 * 数据布局前提（Channel-First）
 * ============================================================================
 *
 *  RKNN INT8 输出采用 channel-first 布局。同一 anchor 的 objectness 通道内，
 *  所有 grid cell 连续存放，因此 obj_ptr[0..grid_len-1] 可以按 16 个 int8_t
 *  一组用 NEON 连续加载。三个 grid_len (6400, 1600, 400) 都能被 16 整除。
 *
 *    [Anchor 0]
 *      Channel 0 (cx):      [cell_00, ..., cell_{grid_len-1}]  连续 int8
 *      Channel 1 (cy):      [...]
 *      Channel 4 (obj_conf):[...]  ← NEON 扫描这个通道
 *      ...
 *      Channel 84 (class_79): [...]
 *
 * ============================================================================
 * 不做 NEON 的部分（及原因）
 * ============================================================================
 *
 *  80 类搜索 —— 跨步访存：
 *    class 通道间隔 grid_len 字节（最大 6400），无法 NEON 连续加载。
 *
 *  NMS —— 散列访存：
 *    IoU 计算访问 boxes[n*4] 和 boxes[m*4]，n/m 不连续。
 *    且 NMS 仅占后处理 ~11% 时间，性价比不高。
 *
 * ============================================================================
 * 性能数据（RK3588 Cortex-A76, GCC 9.4.0, 1000 次迭代）
 * ============================================================================
 *
 *   标量版 (-O2 -fno-tree-vectorize):  205.9 us/frame,  303M insn,  298M cycles
 *   编译器自动向量化 (-O3):             188.5 us/frame,  285M insn,  220M cycles
 *                                       (GCC 报告: vectorized 0 loops)
 *   手写 NEON 版:                      131.8 us/frame,  197M insn,  195M cycles
 *
 *   NEON vs 标量:  耗时 -36.0%,  指令数 -35.1%,  cycles -34.5%
 *   decode 阶段:   183us -> 99us  (-45.9%)
 *
 * ============================================================================
 * 对应的 AArch64 NEON 汇编（objdump 验证）
 * ============================================================================
 *
 *   vdupq_n_s8(thres_i8)           -> dup   v0.16b, w26
 *   vld1q_s8(obj_ptr + g*16)      -> ldr   q0, [x24]
 *   vcgeq_s8(confs, vthresh)      -> cmge  v0.16b, v0.16b, v1.16b
 *   vmaxvq_u8(mask)                -> umaxv b0, v0.16b
 *   (extract to scalar)            -> umov  w0, v0.b[0]
 *   (skip if all zero)             -> cbz   w0, <next_group>
 */

#include "post_process_neon.h"
#include <arm_neon.h>
#include <cmath>

int process_neon(int8_t *input, float *anchor, int grid_h, int grid_w,
                  int model_height, int model_width, int stride,
                  std::vector<float> &boxes, std::vector<float> &objProbs,
                  std::vector<int> &classID, float box_threshold,
                  int32_t zp, float scale)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;

    // ---- 阈值准备：sigmoid域 -> logit域 -> int8量化 ----
    float thres = post_process_common_skip_sigmoid() ? box_threshold : unsigmoid_cmn(box_threshold);
    int8_t thres_i8 = qnt_f32_to_int8_cmn(thres, zp, scale);
    int8x16_t vthresh = vdupq_n_s8(thres_i8);  // 广播到 16 个 lane: [T,T,...,T]
    // 对应汇编: dup v0.16b, w26

    for (int a = 0; a < 3; a++) {
        // 定位到 anchor a 的 objectness 通道（第 4 通道）首地址
        // 内存偏移 = (a * 85 + 4) * grid_len
        int8_t *obj_ptr = input + (a * BOX_NUM_SIZE + 4) * grid_len;
        int total = grid_len;
        int groups = total / 16;  // 6400/1600/400 均能被 16 整除

        // ==================================================================
        // 第一级：NEON 粗筛 —— 每 16 个 cell 一批扫描
        // ==================================================================
        for (int g = 0; g < groups; g++) {
            // ① vld1q_s8: 一次加载 16 个连续 int8_t objectness 值
            //    对应汇编: ldr q0, [x24]
            int8x16_t confs = vld1q_s8(obj_ptr + g * 16);

            // ② vcgeq_s8: 16 路并行 >= 比较
            //    满足阈值的 lane → 0xFF, 不满足 → 0x00
            //    用 >= 而非 >：允许 conf==thres 的假阳性，但绝不漏报
            //    对应汇编: cmge v0.16b, v0.16b, v1.16b
            uint8x16_t mask = vcgeq_s8(confs, vthresh);

            // ③ vmaxvq_u8: 水平归约 —— 取 16 个 mask byte 的最大值
            //    全 0 → 整组无候选，直接跳过（99%+ 的 cell 走这条路）
            //    非 0 → 组内有候选，进入第二级精筛
            //    对应汇编: umaxv b0, v0.16b → umov w0, v0.b[0] → cbz w0, next
            if (vmaxvq_u8(mask) == 0) continue;

            // ==============================================================
            // 第二级：标量精筛 —— 逐个 cell 确认并解码
            // ==============================================================
            for (int k = 0; k < 16; k++) {
                int cell = g * 16 + k;

                // ④ 精确判断：标量 > 比较，与标量版完全一致
                //    粗筛中 conf==thres 的假阳性在此被过滤
                if (obj_ptr[cell] <= thres_i8) continue;

                // 从一维 cell 索引反算 (row, col) 坐标
                int i = cell / grid_w;
                int j = cell % grid_w;

                // 此 cell 在输入数据中的 85 通道基地址
                int box_offt = (a * BOX_NUM_SIZE) * grid_len + cell;
                int8_t *box_p = input + box_offt;

                // ---- 解码边界框 (cx, cy, w, h) ----
                // YOLOv5 用通道 0/1/2/3 存储 cx/cy/w/h 的 logit 值
                // 流程: 反量化(int8→float) → sigmoid → 缩放 → 加 grid 偏移 → 左上角坐标
                float bx = deqnt_int8_to_f32_cmn(*box_p, zp, scale); float box_x = apply_activation_cmn(bx) * 2 - 0.5;
                float by = deqnt_int8_to_f32_cmn(*(box_p + 1 * grid_len), zp, scale); float box_y = apply_activation_cmn(by) * 2 - 0.5;
                float bw = deqnt_int8_to_f32_cmn(*(box_p + 2 * grid_len), zp, scale); float box_w = apply_activation_cmn(bw) * 2.0;
                float bh = deqnt_int8_to_f32_cmn(*(box_p + 3 * grid_len), zp, scale); float box_h = apply_activation_cmn(bh) * 2.0;

                // 映射到原图坐标
                box_x = (box_x + j) * (float)stride;
                box_y = (box_y + i) * (float)stride;
                box_w = box_w * box_w * (float)anchor[a * 2];
                box_h = box_h * box_h * (float)anchor[a * 2 + 1];

                // 中心点坐标 → 左上角坐标
                box_x -= box_w / 2.0f;
                box_y -= box_h / 2.0f;

                // 存入结果向量（每框 4 个 float: x, y, w, h）
                boxes.emplace_back(box_x);
                boxes.emplace_back(box_y);
                boxes.emplace_back(box_w);
                boxes.emplace_back(box_h);

                // ---- 80 类搜索 ----
                // 类别通道 5~84，跨步 = grid_len 字节（不连续），标量循环
                // 不能用 NEON 的原因：通道间隔太大（最大 6400），无法连续加载
                int8_t maxClassProb = *(box_p + 5 * grid_len);
                int maxClassId = 0;
                for (int c = 1; c < OBJ_CLASS_NUM; c++) {
                    int8_t prob = *(box_p + (5 + c) * grid_len);
                    if (prob > maxClassProb) { maxClassProb = prob; maxClassId = c; }
                }
                objProbs.emplace_back(apply_activation_cmn(deqnt_int8_to_f32_cmn(maxClassProb, zp, scale)));
                classID.emplace_back(maxClassId);
                validCount++;
            }
        }

        // ==================================================================
        // 标量尾部：不足 16 个的剩余 cell
        // ==================================================================
        // 对本模型 grid_len % 16 == 0 恒成立 (6400/1600/400 都是 16 的倍数)，
        // 此循环体实际一次都不会执行。保留为兼容非标准模型尺寸。
        for (int t = groups * 16; t < total; t++) {
            if (obj_ptr[t] <= thres_i8) continue;
            int i = t / grid_w;
            int j = t % grid_w;
            int box_offt = (a * BOX_NUM_SIZE) * grid_len + t;
            int8_t *box_p = input + box_offt;

            float bx = deqnt_int8_to_f32_cmn(*box_p, zp, scale); float box_x = apply_activation_cmn(bx) * 2 - 0.5;
            float by = deqnt_int8_to_f32_cmn(*(box_p + 1 * grid_len), zp, scale); float box_y = apply_activation_cmn(by) * 2 - 0.5;
            float bw = deqnt_int8_to_f32_cmn(*(box_p + 2 * grid_len), zp, scale); float box_w = apply_activation_cmn(bw) * 2.0;
            float bh = deqnt_int8_to_f32_cmn(*(box_p + 3 * grid_len), zp, scale); float box_h = apply_activation_cmn(bh) * 2.0;

            box_x = (box_x + j) * (float)stride;
            box_y = (box_y + i) * (float)stride;
            box_w = box_w * box_w * (float)anchor[a * 2];
            box_h = box_h * box_h * (float)anchor[a * 2 + 1];
            box_x -= box_w / 2.0f;
            box_y -= box_h / 2.0f;

            boxes.emplace_back(box_x); boxes.emplace_back(box_y);
            boxes.emplace_back(box_w); boxes.emplace_back(box_h);

            int8_t maxClassProb = *(box_p + 5 * grid_len);
            int maxClassId = 0;
            for (int c = 1; c < OBJ_CLASS_NUM; c++) {
                int8_t prob = *(box_p + (5 + c) * grid_len);
                if (prob > maxClassProb) { maxClassProb = prob; maxClassId = c; }
            }
            objProbs.emplace_back(apply_activation_cmn(deqnt_int8_to_f32_cmn(maxClassProb, zp, scale)));
            classID.emplace_back(maxClassId);
            validCount++;
        }
    }
    return validCount;
}
