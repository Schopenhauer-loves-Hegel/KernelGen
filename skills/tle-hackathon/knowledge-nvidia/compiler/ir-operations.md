---
title: TLE IR Operations
scope: TLE MLIR Dialect 中定义的 24 个 IR 操作的语义
hardware: 通用（IR 层面与硬件无关）
prerequisites: 理解 MLIR operation 概念
---

# TLE IR Operations

定义位置：`third_party/tle/dialect/include/IR/TleOps.td`
C++ 实现：`third_party/tle/dialect/lib/IR/Ops.cpp`
Dialect 定义：`third_party/tle/dialect/include/IR/TleDialect.td`

## 核心数据操作

### tle.extract_tile

从 tensor 中提取子区域。

- 输入：源 tensor, 索引, tile 形状
- 输出：提取的 tile tensor
- 降低：`triton-tle-lower-extract-tile` → 索引计算 + load

### tle.insert_tile

将 tile 插入 tensor 的指定位置。

- 输入：目标 tensor, tile, 索引
- 输出：更新后的 tensor
- 降低：`triton-tle-lower-insert-tile` → 索引计算 + store

### tle.exclusive_cumsum

Exclusive 前缀和操作。

- 输入：tensor, axis, reverse flag
- 输出：(exclusive_sum, total_sum)
- 降低：`triton-tle-lower-exclusive-cumsum`
- 布局优化：`triton-tle-optimize-exclusive-cumsum-layouts`

## 内存操作

### tle.local_pointers

获取 shared memory buffer 的指针视图。

- 输入：buffered_tensor, 可选索引
- 输出：pointer tensor
- 后续 pass 链：select-encodings → insert-barriers → optimize-loads/stores

### tle.memdesc_wgmma_view

为 WGMMA 操作创建 memory descriptor 视图。

- 用于 Hopper WGMMA 指令的操作数准备
- 关联：`tle.wgmma_shared_operand_fence`

### tle.wgmma_shared_operand_fence

WGMMA shared memory 操作数的 fence 操作。

- 确保 shared memory 写入对 WGMMA 可见
- 在 WGMMA 操作前插入

### tle.tma_store_commit_group

提交一组 TMA store 操作。

- 将多个 TMA store 操作作为一个组提交
- 确保组内所有 store 在提交后对后续操作可见

## 分布式操作

### tle.remote_pointers

获取远程设备上 tensor 的指针。

- 输入：tensor, shard_id, scope
- 输出：remote pointer tensor
- 底层通过 FlagCX 通信库实现

### tle.distributed_barrier

分布式同步屏障。

- 输入：可选的 mesh scope
- 效果：同步指定范围内的所有设备

## Raw/DSL 操作

### tle.dsl_region

DSL 内联代码区域。

- 包含用户提供的 CUDA/MLIR 代码
- 通过 `tle-dslregion-inline` pass 在 LLVM 降低阶段内联

### tle.yield

DSL region 的返回值操作。

- 从 `dsl_region` 中返回计算结果

## Pointer 解构操作

用于从 tensor pointer 中提取各组成部分：

| Op | 提取内容 |
|----|----------|
| `tle.extract_allocated_ptr` | 分配的基地址 |
| `tle.extract_aligned_ptr` | 对齐后的指针 |
| `tle.extract_offset` | 偏移量 |
| `tle.extract_sizes` | 各维度大小 |
| `tle.extract_strides` | 各维度步长 |
| `tle.extract_ptr` | 完整指针信息 |

这些操作主要用于 Raw API 中，将 Triton 的 tensor pointer 解构为可以传递给内联代码的原始值。

## Pipe 操作 (6)

### tle.pipe.create

从 buffered tensor handles 创建 pipe descriptor。

- 输入：buffered tensor handles
- 输出：pipe descriptor

### tle.pipe.writer_acquire

Writer 获取一个 stage slot。

- 输入：pipe descriptor, stage index
- 输出：acquired stage slot

### tle.pipe.writer_commit

Writer 将数据提交到一个 stage。

- 输入：pipe descriptor, stage slot
- 效果：数据对 reader 可见

### tle.pipe.writer_close

Writer 发出不再有更多数据的信号。

- 输入：pipe descriptor
- 效果：通知 reader 数据流结束

### tle.pipe.reader_wait

Reader 等待某个 stage 的数据就绪。

- 输入：pipe descriptor, stage index
- 效果：阻塞直到数据可用

### tle.pipe.reader_release

Reader 释放一个已消费的 stage。

- 输入：pipe descriptor, stage index
- 效果：释放 stage slot 供 writer 复用

## 打包操作

### tle.pack

将多个值打包为单个 tensor。

- 用于 Raw API 参数传递

## 操作统计

| 类别 | 数量 | 操作 |
|------|------|------|
| 核心数据 | 3 | extract_tile, insert_tile, exclusive_cumsum |
| 内存 | 4 | local_pointers, memdesc_wgmma_view, wgmma_shared_operand_fence, tma_store_commit_group |
| 分布式 | 2 | remote_pointers, distributed_barrier |
| Raw/DSL | 2 | dsl_region, yield |
| Pointer 解构 | 6 | extract_allocated_ptr, extract_aligned_ptr, extract_offset, extract_sizes, extract_strides, extract_ptr |
| Pipe | 6 | pipe.create, pipe.writer_acquire, pipe.writer_commit, pipe.writer_close, pipe.reader_wait, pipe.reader_release |
| 打包 | 1 | pack |
| **总计** | **24** | |
