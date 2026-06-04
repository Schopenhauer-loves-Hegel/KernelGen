---
title: Shared Memory Staging 模式
scope: 使用 local_ptr 实现数据在 shared memory 中的暂存和复用
hardware: sm80+
prerequisites: api/gpu.md
---

# Shared Memory Staging 模式

## 核心思想

将 global memory 数据暂存到 shared memory，实现：
1. 减少重复 global memory 访问
2. 利用 shared memory 的低延迟进行多次读取
3. 在 warp 间共享数据

## 1D Local Staging（元素级）

适用于 elementwise fusion 和短期数据复用。

```python
@triton.jit
def staged_kernel(input_ptr, output_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # 分配 shared memory
    smem = tle.gpu.alloc([BLOCK], dtype=tl.float32,
                         nv_mma_shared_layout=False)
    ptrs = tle.gpu.local_ptr(smem, (tl.arange(0, BLOCK),))

    # Global → Shared
    data = tl.load(input_ptr + offs, mask=mask, other=0.0)
    tl.store(ptrs, data, mask=mask)

    # 从 shared memory 多次读取（低延迟）
    val = tl.load(ptrs, mask=mask, other=0.0)
    result = val * val + val  # 多次使用同一数据

    tl.store(output_ptr + offs, result, mask=mask)
```

## 2D Tile Staging（矩阵级）

适用于 tile 级操作，如矩阵分块。

```python
@triton.jit
def tile_staged_kernel(..., BM: tl.constexpr, BK: tl.constexpr):
    # 分配 2D shared memory buffer
    tile_buf = tle.gpu.alloc([BM, BK], dtype=tl.float16,
                             nv_mma_shared_layout=True)

    # 构造 2D 索引
    rows = tl.broadcast_to(tl.arange(0, BM)[:, None], (BM, BK))
    cols = tl.broadcast_to(tl.arange(0, BK)[None, :], (BM, BK))
    ptr = tle.gpu.local_ptr(tile_buf, (rows, cols))

    # 加载并使用
    tile_data = tl.load(ptr)
    # ... 计算 ...
```

## copy vs load/store 选择

| 场景 | 推荐方式 | 原因 |
|------|----------|------|
| 连续块传输 | `tle.gpu.copy` | TMA 硬件加速（sm90+） |
| 自定义索引模式 | `local_ptr + tl.load/store` | 灵活的访问模式 |
| 需要 mask | `local_ptr + tl.load/store` | copy 不支持 mask |
| 描述符流 | `tle.gpu.copy` | 与 TMA descriptor 配合 |

## 编译器行为

local_ptr staging 触发的 pass 链：

1. `triton-tle-early-assign-memory-space` — 确认 buffer 在 shared memory
2. `triton-tle-select-encodings` — 选择适合的 shared encoding
3. `triton-tle-insert-local-pointer-barriers` — 自动插入 barrier（确保 store 对后续 load 可见）
4. `triton-tle-optimize-local-pointer-loads` → `ttg.local_load`
5. `triton-tle-optimize-local-pointer-stores` → `ttg.local_store`

## local_ptr 使用规则

1. `buffer` 必须来自 `tle.gpu.alloc`
2. `indices` 不能为空，数量必须等于 buffer 维度
3. 索引类型必须是整数
4. 要么全是标量索引，要么全是 tensor 索引（不能混合）
5. tensor 索引模式下，所有索引 tensor 形状必须相同

## 何时使用 Staging

适合：
- 同一数据被多次读取（reduction、scan、多步计算）
- 需要 warp 间数据共享
- 访问模式不规则但数据量适合 shared memory

不适合：
- 数据只读一次（直接从 global 读取更简单）
- 数据量超过 shared memory 容量
- 已经有足够的 L1/L2 cache 命中
