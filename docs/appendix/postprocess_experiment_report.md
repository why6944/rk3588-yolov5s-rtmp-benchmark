# YOLOv5 后处理三组对照实验报告

## 1. 实验环境

| 项目 | 规格 |
|------|------|
| 硬件 | Orange Pi 5 (RK3588) |
| CPU | 4×Cortex-A76 @ 最高 2.4GHz + 4×Cortex-A55 @ 最高 1.8GHz |
| 编译器 | GCC 9.4.0 (Ubuntu 20.04), aarch64 |
| 模型 | YOLOv5s INT8 (对称仿射量化), 640×640 输入, 80类 |
| 测试核心 | CPU4 (Cortex-A76), governor=performance |
| 测试参数 | 预热 100 次, 测试 1000 次(perf:500次), 重复 5 轮 |
| 后处理阈值 | box_threshold=0.5, nms_threshold=0.5 |

## 2. NMS Bug 修复

原代码在 `post_process.cpp` 的 `nms()` 函数中存在**两个** bug：

### Bug 1: 类别索引错误

```diff
- if(indexArray[i] == -1 || classID[i] != currentClass)
+ if(indexArray[i] == -1 || classID[indexArray[i]] != currentClass)

- if(m == -1 || classID[j] != currentClass)
+ if(m == -1 || classID[m] != currentClass)
```

`sort_descending` 按置信度降序排列后，`indexArray[i]` 是排序后第 i 位的框在原始数组中的位置。`classID` 按原始顺序填充，用排序后的 `i`/`j` 索引会查到错误框的类别。

### Bug 2: 抑制方向反了

```diff
  if (iou > nms_threshold) {
-     indexArray[i] = -1;   // 把高置信度框抑制了！
+     indexArray[j] = -1;   // 应抑制低置信度框
  }
```

外层循环 `i` 是高置信度框 (keeper)，内层 `j` 是低置信度框。IoU 超过阈值时应抑制 `j`，但原代码抑制了 `i`，导致最优框被丢弃。

## 3. 数据布局分析

### 输出张量尺寸

| 输出 | Stride | Grid | 每通道元素数 | 总大小 |
|------|--------|------|-------------|--------|
| output0 | 8 | 80×80 | 6,400 | 1,632,000 int8 |
| output1 | 16 | 40×40 | 1,600 | 408,000 int8 |
| output2 | 32 | 20×20 | 400 | 102,000 int8 |
| **合计** | | | | **2,142,000 int8** |

### 内存布局 (Channel-First)

```
[Anchor 0]
  Channel 0 (cx):      [cell_0, cell_1, ..., cell_6399]  ← 6,400 连续 int8
  Channel 1 (cy):      [cell_0, cell_1, ..., cell_6399]
  Channel 2 (w):       [...]
  Channel 3 (h):       [...]
  Channel 4 (obj_conf):[cell_0, cell_1, ..., cell_6399]  ← NEON 连续加载目标
  Channel 5 (class_0): [...]
  ...
  Channel 84 (class_79): [...]
[Anchor 1] 同上
[Anchor 2] 同上
```

**NEON 优化的关键**: 所有 grid 大小 (6400, 1600, 400) 都能被 16 整除。同一 anchor 的 objectness 通道内，16 个连续 grid cell 可以直接用 `vld1q_s8` 加载到一个 128-bit NEON 寄存器。

## 4. 三版本实现说明

### 标量版 (`--postprocess scalar`)
- 编译选项: `-O2 -fno-tree-vectorize`
- 纯 C++ 标量实现，三重循环遍历 anchor→row→col
- 无任何 SIMD 指令（objdump 确认: 0 条）

### 自动向量化版 (`--postprocess auto`)
- 编译选项: `-O3 -ftree-vectorize -fopt-info-vec-all`
- 算法代码与标量版完全相同
- **GCC 向量化报告: "vectorized 0 loops in function"**
  - 原因: "loop nest containing two or more consecutive inner loops cannot be vectorized"
  - 三重嵌套循环 + sigmoid/dequant 函数调用阻止了自动向量化
- objdump 确认: 0 条 SIMD 指令

### 手写 NEON 版 (`--postprocess neon`)
- 编译选项: `-O2 -march=armv8-a+simd`
- 优化范围: 仅 objectness 置信度扫描（不涉及 80 类搜索和 NMS）
- NEON intrinsics 使用:
  ```c
  int8x16_t vthresh = vdupq_n_s8(thres_i8);         // 广播阈值到 16 通道
  int8x16_t confs = vld1q_s8(obj_ptr + g * 16);     // 加载 16 个 int8
  uint8x16_t mask = vcgeq_s8(confs, vthresh);       // 16 路并行比较
  if (vmaxvq_u8(mask) == 0) continue;               // 整组无候选 → 跳过
  ```
- 汇编确认: `dup + ldr×2 + cmge + umaxv + umov` 共 6 条 NEON 指令
- 有候选框时回退到标量代码解码坐标和查找类别
- 尾部不足 16 元素用标量处理

## 5. 正确性验证

三种实现对同一帧 INT8 数据输出:

| 版本 | valid_count | result_count | 状态 |
|------|------------|-------------|------|
| scalar | 162 | 17 | 基准 |
| auto | 162 | 17 | ✅ 一致 |
| neon | 162 | 17 | ✅ 一致 |

- 框坐标、类别 ID、置信度完全一致（差异 < 1e-6）
- 三类实现仅在 objectness 扫描策略上不同，decode/NMS/sort 逻辑完全共享

## 6. 性能对比

### 端到端后处理耗时

| 版本 | 平均耗时 (μs) | vs 标量 | 稳定性 (stddev) |
|------|-------------|---------|----------------|
| scalar (-O2) | 205.9 | 基准 | ±0.5 |
| auto (-O3) | 188.5 | **-8.4%** | ±20.0 (不稳定) |
| **neon** | **131.8** | **-36.0%** | **±0.5** |

> 注: auto 版第一轮 156.5μs，后续轮次 203μs，呈现缓存/温升退化现象。

### 各阶段耗时分解

| 阶段 | scalar (μs) | auto (μs) | neon (μs) | NEON 提升 |
|------|-----------|----------|----------|----------|
| decode | 183 | 125 | 99 | **-45.9%** |
| sort | 2 | 2 | 2 | — |
| nms | 22 | 22 | 22 | — |
| result | 0 | 0 | 0 | — |
| **总计** | **207** | **149** | **123** | **-40.6%** |

> NMS 占后处理总时间 ~11%，未触发第二轮四框并行 IoU 优化条件 (阈值: 20%)。

### perf stat 硬件计数器 (500 次迭代)

| 指标 | scalar | auto | neon | NEON vs Scalar |
|------|--------|------|------|---------------|
| cycles | 298M | 220M | 195M | **-34.5%** |
| instructions | 303M | 285M | 197M | **-35.1%** |
| IPC | 1.02 | 1.30 | 1.01 | — |
| cache-references | 54.3M | 52.9M | 43.6M | **-19.6%** |
| cache-misses | 4.5M | 4.5M | 4.5M | — |
| cache-miss % | 8.33% | 8.43% | 10.29% | — |

## 7. 汇编分析

### 标量版 (process_scalar @ 0x5b48)

纯标量指令序列，无任何 SIMD:
```
5b48: stp x29, x30, [sp, #-352]!
5b4c: mov w4, w3
... (全部为标准 ARM64 标量指令: ldrb, cmp, b.le, fmov, etc.)
```
**SIMD 指令数: 0**

### 自动向量化版 (process_auto @ 0x62b0)

与标量版类似，无 SIMD:
```
62b0: stp x29, x30, [sp, #-352]!
... (标准标量指令)
```
**SIMD 指令数: 0** — 确认 GCC 9.4.0 的 `-O3 -ftree-vectorize` 未对此代码生成任何 NEON 指令。

### 手写 NEON 版 (process_neon @ 0x6810)

关键 NEON 循环 (地址 6978-698c):
```asm
6978: ldr  q0, [x24]           ; vld1q_s8: 加载 16 个 int8 objectness 值
697c: ldr  q1, [sp, #256]      ; 加载阈值向量 (由 vdupq_n_s8 初始化)
6980: cmge v0.16b, v0.16b, v1.16b  ; vcgeq_s8: 16 路并行比较 >=
6984: umaxv b0, v0.16b         ; vmaxvq_u8: 水平归约求最大值
6988: umov w0, v0.b[0]         ; 提取到通用寄存器
698c: cbz  w0, 6958            ; 若全为 0 → 跳过 16 个 cell
```
**SIMD 指令数: 6** (dup + ldr×2 + cmge + umaxv + umov)

编译选项 `-march=armv8-a+simd` 正确生成了 AArch64 NEON 指令。

## 8. 系统级影响

| 指标 | 标量 | 自动向量化 | 手写 NEON |
|------|------|-----------|----------|
| 后处理耗时 (μs) | 206 | 189 | **132** |
| 纯推理耗时 (ms, 不含后处理) | ~58 | ~58 | ~58 |
| 预估 infer-only FPS | 3.8 | 4.1 | **5.3** |
| 帧处理总时间 (ms) | 264 | 247 | **190** |

> infer-only FPS = 1000 / (58ms RKNN + 后处理时间), 单线程估计值。多线程管线实际 FPS 更高。

## 9. 结论与讨论

### 编译器自动向量化有效性
**GCC 9.4.0 的 `-O3` 对本代码完全无效 (0 loops vectorized)**。三重嵌套循环 + 函数调用 (sigmoid/dequant) 阻止了自动向量化器。`-O3` 仅通过更好的指令调度获得 ~8% 提升，但引入了不稳定性（各轮之间波动大）。

### 手写 NEON 收益
**NEON 后处理比标量快 36%，decode 阶段快 46%**。收益来源:
1. Objectness 通道连续排列，16×int8_t 批量加载只需 1 条 `ldr q` 指令
2. 16 路并行比较 (`cmge`) 代替 16 次标量 `cmp`
3. 水平归约 (`umaxv`) 快速判断整组是否有候选
4. 无候选时直接跳过 16 个 cell，大幅减少内存访问（cache-refs -19.6%）

### 是否触发第二轮 NMS NEON
**不触发**。NMS 仅占后处理总时间的 ~11%，低于 20% 阈值。即使全部 NEON 化 NMS，理论最大收益也不到 22μs，投入产出比不高。

### 为什么 80 类搜索和 NMS 不做 NEON
- **80 类搜索**: 跨步访存 `*(box_p + (5+k)*grid_len)`，`grid_len` 最大 6400，步长太大无法用 SIMD 连续加载
- **NMS**: 散列访存 (IoU 计算需要访问 `boxes[n*4]` 和 `boxes[m*4]`，n 和 m 不连续)，不适合 SIMD 打包

### 数据布局的优化价值
Channel-First 布局是本次优化的关键——objectness 通道的所有 grid cell 连续存放，使得 NEON 能够以 16 个元素为一组高效扫描。如果布局是 Cell-First (每个 cell 的 85 通道连续)，则 NEON 扫描 objectness 需要跨 85 个元素的步长加载，效果会大打折扣。

### 后续优化方向
1. **80 类搜索**: 可考虑用 NEON `vmaxvq_s8` 做 16 路并行 max search (需要重构 class 通道布局)
2. **sigmoid/dequant 融合**: 当前每框调用 5 次 sigmoid + 5 次 dequant，可用查表法加速
3. **多线程并行**: 3 个输出头的 process 完全独立，可并行处理
4. **GCC 升级**: GCC 14+ 对嵌套循环的自动向量化能力有显著改进

## 10. 原始数据

### Benchmark 5 轮完整数据

| 轮次 | scalar (μs) | auto (μs) | neon (μs) |
|------|-----------|----------|----------|
| 1 | 206.6 | 156.5 | 132.4 |
| 2 | 206.1 | 175.6 | 132.0 |
| 3 | 205.7 | 203.5 | 131.9 |
| 4 | 205.4 | 203.6 | 131.5 |
| 5 | 205.9 | 203.1 | 131.3 |
| **均值** | **205.9** | **188.5** | **131.8** |

### 文件清单

- `int8_dumps/frame_0000.bin` ~ `frame_0029.bin` (30 帧, 每帧 2,142,036 字节)
- `post_process_scalar.cpp` (标量实现)
- `post_process_auto.cpp` (自动向量化实现)
- `post_process_neon.cpp` + `.h` (NEON 实现)
- `post_process_common.cpp` + `.h` (共享代码)
- `benchmark_main.cpp` (独立 benchmark)
- `benchmark_postprocess` (可执行文件, 不依赖 NPU/RGA/视频)
