---
title: TLE 编译器 Pass Pipeline
scope: 理解 TLE 代码经过哪些编译器变换、以什么顺序执行
hardware: NVIDIA sm80+ (TLE pass 主要在 NVIDIA 后端集成；HCU 携带 TLE dialect 定义；Mthreads 有 1 个 TLE pass: add_tle_lower_async_load)
prerequisites: 理解 MLIR pass 概念、Triton 编译流程
---

# TLE 编译器 Pass Pipeline

## 概述

TLE 定义了 21 个 MLIR pass，在 NVIDIA 后端的 `make_ttgir()` 阶段调用 18 个，`make_llir()` 阶段调用 1 个。

所有 pass 实现位于 `third_party/tle/dialect/lib/Transforms/`。

## Pass 执行顺序（NVIDIA 后端）

以下是 `third_party/nvidia/backend/compiler.py` 中的实际调用顺序：

### 阶段 1: Raw 预处理（最先执行）

| Pass | 功能 |
|------|------|
| `tle-convert-arg-to-memdesc` | 将 Raw 函数参数转换为 memory descriptor |
| `tle-remove-redundant-copy` | 移除 Raw 中的冗余拷贝 |

### 阶段 2: Tile 操作降低

| Pass | 功能 |
|------|------|
| `triton-tle-lower-extract-tile` | 将 `tle.extract_tile` 降低为索引计算 |
| `triton-tle-lower-insert-tile` | 将 `tle.insert_tile` 降低为索引计算 |

### 阶段 3: Local Pointer 优化链

| Pass | 功能 | 顺序依赖 |
|------|------|----------|
| `triton-tle-optimize-local-pointer-async-stores` | 规范化 async copy 模式 | 必须在 assign-memory-space 之前 |
| `triton-tle-early-assign-memory-space` | 为 tensor pointer 分配存储空间 | — |
| `triton-tle-select-encodings` | 为 local pointer 选择 shared encoding | 依赖 memory space |
| `triton-tle-insert-local-pointer-barriers` | 在 store/load 间插入 CTA barrier | 依赖 encoding |
| `triton-tle-optimize-local-pointer-loads` | 重写为 `ttg.local_load` | 依赖 barrier |
| `triton-tle-optimize-local-pointer-stores` | 重写为 `ttg.local_store` | 依赖 barrier |

### 阶段 4: Pipeline 物化

| Pass | 功能 |
|------|------|
| `triton-tle-promote-local-store-staging` | 提升 staging 为可 pipeline 的 alloc |
| `triton-tle-downgrade-invalid-async-copy` | 降级无效 async copy（条件分支中多次出现） |
| `triton-tle-tile-style-pipeline-schedule` | 重写为 TileLang 风格 preload-2 pipeline |
| `triton-tle-materialize-tile-style-pipeline` | 物化显式 pipeline |
| `TleLowerPipeToNvws` | 将 pipe 操作降低到 NVWS warp specialization dialect（sm90+） |

### 阶段 5: 后期降低

| Pass | 功能 | 条件 |
|------|------|------|
| `triton-tle-downgrade-invalid-async-copy` | 最终降级检查 | 无条件 |
| `triton-tle-lower-tma-copy` | 降低 TMA copy 操作 | sm90+ only |
| `TleScheduleTmaStoreSync` | 调度 TMA store 操作的同步 | 在 lower-tma-copy 之后 |
| `triton-tle-lower-async-load` | 降低 async load | 无条件 |

### 阶段 6: LLVM 降低（make_llir 阶段）

| Pass | 功能 |
|------|------|
| `tle-dslregion-inline` | 内联 DSL region（Raw 代码） |

## 未被调用的 Pass

| Pass | 状态 |
|------|------|
| `triton-tle-optimize-exclusive-cumsum-layouts` | 已实现但未在 NVIDIA backend 中调用 |
| `triton-tle-lower-exclusive-cumsum` | 已实现，通过其他路径触发 |

## 硬件条件分支

`make_ttgir()` 中根据 `capability // 10` 有三个分支：

- **sm80-sm99** (`capability // 10 in [8, 9]`): 执行 downgrade + tile-style pipeline（包括 sm90/H100）
- **sm100+** (`capability // 10 >= 10`): 仅执行 downgrade
- **else**: 跳过 pipeline 相关 pass

`lower-tma-copy` 仅在 `capability // 10 >= 9`（sm90+）时执行。

## 关键不变量

1. Local pointer pass 链有严格顺序依赖，不可重排
2. `downgrade-invalid-async-copy` 可能执行多次（不同条件分支 + 最终兜底）
3. TLE pass 在 Triton 原生 pipeline pass 之前执行
4. Raw pass（`convert-arg-to-memdesc`, `remove-redundant-copy`）最先执行
5. DSL region inline 在 LLVM 降低阶段执行（最后）
