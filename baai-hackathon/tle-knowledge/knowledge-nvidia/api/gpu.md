---
title: TLE GPU API
scope: GPU 共享内存管理、TMA 拷贝、Pipeline 循环、内存布局控制
hardware: sm80+ (部分功能需要 sm90+ Hopper)
prerequisites: api/core.md, 理解 shared memory 和 warp 概念
---

# TLE GPU API

入口：`tle.gpu`（即 `triton.experimental.tle.language.gpu`）

## tle.gpu.alloc

在 shared memory 或 tensor memory 中分配 buffer。

```python
tle.gpu.alloc(shape, dtype, layout=None, scope=tle.smem,
              init_value=None, nv_mma_shared_layout=True) -> tle.buffered_tensor
```

- `shape`: tuple，分配的形状
- `dtype`: 数据类型（如 `tl.float16`）
- `layout`: 可选，`tle.shared_layout` / `tle.swizzled_shared_layout` / `None`
- `scope`: `tle.smem`（shared memory）或 `tle.tmem`（tensor memory, sm100+）
- `init_value`: 可选初始值 tensor
- `nv_mma_shared_layout=True`: 自动选择适合 WGMMA 的 swizzled layout

**编译器行为：**
- `triton-tle-early-assign-memory-space` 为 alloc 分配存储空间
- 当 `nv_mma_shared_layout=True` 时，编译器使用 `LinearLayout`（基于 f2reduce）计算 swizzle 映射

**典型用法：**
```python
smem_buf = tle.gpu.alloc([BLOCK_M, BLOCK_K], dtype=tl.float16)
```

## tle.gpu.local_ptr

获取 shared memory buffer 的指针视图，用于 `tl.load` / `tl.store`。

```python
tle.gpu.local_ptr(buffer, indices=None) -> tl.tensor
```

- `buffer`: `tle.gpu.alloc` 返回的 `buffered_tensor`
- `indices`: 可选的索引序列，指定访问的子区域

**编译器 pass 链：**
1. `triton-tle-select-encodings` — 为 local pointer 选择 shared encoding
2. `triton-tle-insert-local-pointer-barriers` — 在 store/load 间插入 CTA barrier
3. `triton-tle-optimize-local-pointer-loads` — 重写为 `ttg.local_load`
4. `triton-tle-optimize-local-pointer-stores` — 重写为 `ttg.local_store`

**典型用法：**
```python
smem = tle.gpu.alloc([BLOCK], dtype=tl.float32, nv_mma_shared_layout=False)
ptrs = tle.gpu.local_ptr(smem, (tl.arange(0, BLOCK),))
tl.store(ptrs, data)
result = tl.load(ptrs)
```

## tle.gpu.copy

TMA 拷贝操作，在 global memory 和 shared memory 间高效传输数据。

```python
tle.gpu.copy(src, dst, shape, offsets=None) -> None
```

- `src`: 源（global tensor pointer 或 shared memory buffer）
- `dst`: 目标
- `shape`: 拷贝的形状（必选）
- `offsets`: 可选的偏移序列
- 编译器 pass `triton-tle-lower-tma-copy` 负责降低（需要 sm90+）

**注意：** TMA copy 需要 Hopper (sm90+) 硬件支持。在 sm80/sm89 上，编译器会自动降级。

## tle.gpu.memory_space

为 tensor 指定存储层级 hint。

```python
tle.gpu.memory_space(input, space) -> tensor
```

- `space`: 目标存储空间标识

## tle.gpu.pipeline

Pipeline 循环迭代器，继承自 `tl.range`，支持多阶段流水线。

```python
class tle.gpu.pipeline(tl.range):
    def __init__(self, arg1, arg2=None, step=None,
                 num_stages=None, loop_unroll_factor=None)
```

- `num_stages`: pipeline 阶段数（影响 buffer 数量和延迟隐藏）
- `loop_unroll_factor`: 循环展开因子

**编译器 pass 链：**
1. `triton-tle-promote-local-store-staging` — 提升 staging 为可 pipeline 的 alloc
2. `triton-tle-tile-style-pipeline-schedule` — 重写为 preload-2 pipeline
3. `triton-tle-materialize-tile-style-pipeline` — 物化显式 pipeline

**典型用法（GEMM pipeline）：**
```python
for k in tle.gpu.pipeline(0, K // BLOCK_K, num_stages=3):
    a = tle.load(a_ptr + ..., is_async=True)
    b = tle.load(b_ptr + ..., is_async=True)
    acc = tl.dot(a, b, acc)
```

## 布局相关类型

| 类型 | 说明 |
|------|------|
| `tle.shared_layout` | 基础 shared memory layout |
| `tle.swizzled_shared_layout` | Swizzled layout（减少 bank conflict） |
| `tle.tensor_memory_layout` | Tensor memory layout (sm100+) |
| `tle.nv_mma_shared_layout` | 适合 WGMMA 的 layout（`alloc` 默认启用） |

## Scope 常量

| 常量 | 说明 |
|------|------|
| `tle.smem` | Shared memory |
| `tle.tmem` | Tensor memory (sm100+) |

## 完整 GEMM 示例骨架

```python
@triton.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in tle.gpu.pipeline(0, K // BLOCK_K, num_stages=3):
        a_tile = tle.load(a_ptr + ..., is_async=True)
        b_tile = tle.load(b_ptr + ..., is_async=True)
        acc = tl.dot(a_tile, b_tile, acc)

    tl.store(c_ptr + ..., acc)
```

## tle.gpu.warp_specialize

创建显式 GPU warp 特化区域（需要 Hopper sm90+）。

```python
tle.gpu.warp_specialize(functions_and_args, worker_num_warps, worker_num_regs)
```

- `functions_and_args`: 列表，每个元素为 `(function, args)` 元组
  - `functions_and_args[0]` 是 default partition
  - 后续元素为 worker partition
- `worker_num_warps`: list[int]，每个 worker 的 warp 数量（长度 = worker 数量）
- `worker_num_regs`: list[int]，每个 worker 请求的寄存器数量（长度 = worker 数量）
- 降低到 NVWS dialect

**典型用法：**
```python
def producer(buf):
    # TMA copy into shared memory
    tle.gpu.copy(src, buf, shape)

def consumer(buf):
    # Compute on shared memory data
    data = tle.gpu.local_ptr(buf)
    acc = tl.dot(data, ...)

tle.gpu.warp_specialize(
    [(consumer, (buf,)), (producer, (buf,))],
    worker_num_warps=[1],
    worker_num_regs=[40],
)
```

## tle.pipe

硬件无关的数据流管道，用于在 warp 特化分区间传递数据。

```python
tle.pipe(*, capacity, scope="cta", name=None, readers=None, one_shot=False, **fields) -> pipe_value
```

- `capacity`: int，管道的 buffer 深度（正整数）
- `scope`: 作用域，当前 MVP 仅支持 `"cta"`
- `name`: 可选名称标识
- `readers`: 可选的 reader 名称元组。省略时为 SPSC 管道；指定时为 SPMC 管道
- `one_shot`: `True` 时建模单次 ready/full 边缘（不支持 close）
- `**fields`: 命名字段，值必须是 `tle.gpu.smem` 中的 `buffered_tensor`，首维度 = capacity

**典型用法：**
```python
buf_A = tle.gpu.alloc([num_stages, BLOCK_M, BLOCK_K], dtype=tl.float16)
buf_B = tle.gpu.alloc([num_stages, BLOCK_K, BLOCK_N], dtype=tl.float16)
p = tle.pipe(capacity=num_stages, A=buf_A, B=buf_B)

w = p.writer()
r = p.reader()
```

## Pipe 类型

### pipe_value

管道描述符，由 `tle.pipe(...)` 返回。

| 方法 | 说明 |
|------|------|
| `.writer()` | 返回 `pipe_writer` 端点 |
| `.reader(name=None, fields=None)` | 返回 `pipe_reader` 端点。SPMC 管道需要传 `name` |

### pipe_writer

写端点，由 `pipe_value.writer()` 返回。

| 方法 | 说明 |
|------|------|
| `.acquire(iter)` | 获取 stage slot，返回 `pipe_slot` |
| `.commit(iter)` | 提交当前 stage，通知 reader |
| `.close(iter)` | 关闭管道（`one_shot=True` 时不可用） |

### pipe_reader

读端点，由 `pipe_value.reader(...)` 返回。

| 方法 | 说明 |
|------|------|
| `.wait(iter)` | 等待数据就绪，返回 `pipe_wait_result` |
| `.release(iter)` | 释放 stage slot，允许 writer 重用 |

### pipe_slot

持有命名字段的容器，字段通过属性访问（如 `slot.A`、`slot.B`）。

### pipe_wait_result

`pipe_reader.wait()` 的返回值。

| 属性 | 类型 | 说明 |
|------|------|------|
| `.slot` | `pipe_slot` | 当前 stage 的字段容器 |
| `.is_closed` | `tl.tensor(int1)` | 管道是否已被 writer 关闭 |

## 其他类型

| 类型 | 说明 |
|------|------|
| `tle.gpu.buffered_tensor` | `tle.gpu.alloc` 返回的 buffer 对象，有 `.slot(stage)` 方法用于索引 stage 维度 |
| `tle.gpu.buffered_tensor_type` | `buffered_tensor` 的类型描述符（继承自 `tl.block_type`） |
| `tle.gpu.scope` | 存储类型枚举（实例：`smem`、`tmem`） |
| `tle.gpu.layout` | 布局基类，所有 shared layout 类型的父类 |
| `tle.gpu.storage_kind` | `memory_space` 的向后兼容别名 |
