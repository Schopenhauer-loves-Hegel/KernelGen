---
title: TLE 对 Triton 上游 Dialect 的修改
scope: 理解 TLE 对 Triton 核心 IR 做了哪些扩展
hardware: 通用
prerequisites: compiler/ir-operations.md
---

# TLE 对 Triton 上游 Dialect 的修改

TLE 在 Triton 核心 Dialect 中做了 4 处修改，使用 `// begin flagtree tle` / `// end flagtree tle` 注释标记。

## 1. SharedMemory Resource

位置：`include/triton/Dialect/Triton/IR/TritonOps.td:25`

新增 `SharedMemory` resource 定义，使 `tt.atomic_rmw` 等操作能声明对 shared memory 的读写效果。

## 2. tt.load 增加 flagtree_hints 属性

位置：`include/triton/Dialect/Triton/IR/TritonOps.td:256`

为 `tt.load` 操作增加 `flagtree_hints` 属性，用于传递 TLE 的编译器 hint（如 `is_async`）。

编译器通过 `python/triton/compiler/hint_manager.py` 管理这些 hint。

## 3. tt.atomic_rmw 支持 Shared Memory

位置：`include/triton/Dialect/Triton/IR/TritonOps.td:385-388`

扩展 `tt.atomic_rmw` 操作，增加对 shared memory 的读写效果声明。使得原子操作可以作用于 shared memory 中的数据。

## 4. ttg.tma_copy 操作

位置：`include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:601`

新增 `TTG_TMACopyOp`（`ttg.tma_copy`），用于表示 TMA (Tensor Memory Accelerator) 拷贝操作。

由 `triton-tle-lower-tma-copy` pass 从 `tle.gpu.copy` 降低而来。

## 条件编译

NVIDIA 后端的 `TritonNVIDIAGPUToLLVM/DotOpToLLVM.cpp`（及其他多个 NVIDIA 后端文件）中使用 `#ifdef __TLE__` 条件编译 TLE 相关的 lowering patterns，确保不影响非 TLE 构建。

## 影响范围

这些修改是 TLE 与 Triton 上游的唯一耦合点。其余 TLE 功能完全在 `third_party/tle/` 中自包含。
