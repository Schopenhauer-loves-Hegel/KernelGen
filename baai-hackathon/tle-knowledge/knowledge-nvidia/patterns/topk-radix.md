---
title: TopK Radix 模式
scope: 使用 alloc + cumsum + extract_tile 实现高效 TopK 选择
hardware: sm80+
prerequisites: api/core.md, api/gpu.md
---

# TopK Radix 模式

## 核心思想

Radix-based TopK 通过逐位（bit-by-bit）筛选避免完整排序，利用 TLE 的以下能力：

1. `tle.gpu.alloc` — shared memory 中的 histogram buffer
2. `tle.cumsum` — exclusive prefix sum 计算 histogram 的累积分布
3. `tle.extract_tile` — 从大 tensor 中提取子区域

## 算法流程

```
对于每一位（从高位到低位）:
  1. 构建 histogram（统计每个 radix bucket 的元素数）
  2. cumsum 得到 prefix sum（确定每个 bucket 的起始位置）
  3. 根据 prefix sum 判断 TopK 落在哪个 bucket
  4. 缩小搜索范围到目标 bucket
```

## TLE 实现骨架

```python
@triton.jit
def topk_radix_kernel(
    input_ptr, output_ptr, N, K,
    BLOCK: tl.constexpr, NUM_BUCKETS: tl.constexpr,
):
    pid = tl.program_id(0)

    # Shared memory histogram
    hist_buf = tle.gpu.alloc([NUM_BUCKETS], dtype=tl.int32,
                             nv_mma_shared_layout=False)
    hist_ptrs = tle.gpu.local_ptr(hist_buf, (tl.arange(0, NUM_BUCKETS),))

    # 初始化 histogram
    tl.store(hist_ptrs, tl.zeros([NUM_BUCKETS], dtype=tl.int32))

    # 逐位处理
    for bit in range(BITS - 1, -1, -1):
        # 统计每个 bucket 的元素数
        # ... (atomic add to hist_ptrs)

        # Prefix sum
        hist_vals = tl.load(hist_ptrs)
        prefix, total = tle.cumsum(hist_vals, axis=0)

        # 确定 TopK 落在哪个 bucket
        # ... (比较 prefix sum 与 K)

        # 重置 histogram
        tl.store(hist_ptrs, tl.zeros([NUM_BUCKETS], dtype=tl.int32))
```

## cumsum 在 TopK 中的作用

`tle.cumsum` 返回 `(exclusive_sum, total_sum)`：

- `exclusive_sum[i]` = bucket 0 到 bucket i-1 的元素总数
- `total_sum` = 所有 bucket 的元素总数

通过比较 `exclusive_sum[i]` 与 K，可以确定 TopK 的第 K 个元素落在哪个 bucket。

## extract_tile 的应用

当数据分布在多个 tile 中时，使用 `extract_tile` 提取目标区域：

```python
# 从大 tensor 中提取当前处理的 tile
tile_data = tle.extract_tile(full_tensor, [tile_idx * TILE_SIZE], [TILE_SIZE])
```

## 调优要点

| 参数 | 影响 | 建议 |
|------|------|------|
| `NUM_BUCKETS` | Radix 基数，影响迭代次数 | 256 (8-bit radix) |
| `BLOCK` | 每个 thread block 处理的元素数 | 1024-4096 |
| Radix 位宽 | 迭代次数 vs histogram 大小 | 4-8 bits |

## 参考实现

- `python/tutorials/tle/03-topk.py` — Radix-Only TopK 教程
- `python/tutorials/tle/deepseek_v32/01-topk_selector.py` — 4-step radix selector (166KB)
- `python/triton_kernels/triton_kernels/topk.py` — 高层 `topk_forward()` 封装
- `python/triton_kernels/triton_kernels/topk_details/` — streaming topk 内核实现
