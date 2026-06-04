---
title: 性能调优参数优先级
scope: TLE kernel 调优的系统方法 — 参数顺序、benchmark 方法、stop conditions
hardware: sm80+
prerequisites: api/gpu.md, patterns/gemm-pipeline.md
---

# 性能调优参数优先级

## 调优顺序（影响从大到小）

1. **Tile sizes** (`BLOCK_M`, `BLOCK_N`, `BLOCK_K` 或 1D `BLOCK`)
2. **num_warps**
3. **num_stages**
4. **Memory path choice** (`tle.gpu.copy` vs `local_ptr + tl.load/store`)
5. **Layout settings** (`nv_mma_shared_layout`, swizzled layout)

## 一次改一个参数

每次调优迭代：
1. 固定 shape、seed、grid
2. 只改一个参数
3. 运行正确性检查
4. 运行计时 benchmark
5. 记录 TTGIR/PTX 证据

## Benchmark 骨架

```python
import torch
import time
import triton

def bench(fn, warmup=10, rep=50):
    """最小化 benchmark 函数"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / rep
    return elapsed * 1000  # ms

# 使用 triton.testing 的更精确版本
ms = triton.testing.do_bench(lambda: kernel[grid](...))
```

## Stop Conditions

满足以下任一条件时停止调优：

1. 连续 3 次单参数试验无可测量改进
2. 回归风险上升（正确性不稳定、mask 边界脆弱）
3. 已达到目标性能

## Tile Size 调优指南

| 操作类型 | 建议范围 | 考虑因素 |
|----------|----------|----------|
| GEMM | M/N: 64-256, K: 32-128 | 计算密度 vs occupancy |
| Elementwise | 256-4096 | 内存带宽利用 |
| Reduction | 256-2048 | 并行度 vs 同步开销 |
| TopK | 1024-4096 | histogram 大小 vs 并行度 |

## num_warps 调优

| 值 | 适用场景 |
|----|----------|
| 4 | 小 tile、内存密集型 |
| 8 | 大 tile、计算密集型 GEMM |
| 16 | 超大 tile (256x256+) |

## num_stages 调优

| 值 | 效果 | 代价 |
|----|------|------|
| 1 | 无 pipeline | 最少 shared memory |
| 2 | 基础延迟隐藏 | 2x buffer |
| 3 | 良好延迟隐藏 | 3x buffer |
| 4-5 | 最大延迟隐藏 | 可能超出 shared memory |

权衡：stages 越多，shared memory 占用越大，可能降低 occupancy。

## Memory Path 选择

| 条件 | 推荐 |
|------|------|
| sm90+ + 连续块传输 | `tle.gpu.copy`（TMA 加速） |
| sm80/89 | `local_ptr + tl.load/store` |
| 需要 mask 或自定义索引 | `local_ptr + tl.load/store` |
| GEMM operand 加载 | `tle.load(..., is_async=True)` + pipeline |

## 编译器产物检查

调优时检查 TTGIR/PTX 确认编译器行为符合预期：

```python
compiled = kernel.warmup(*args, grid=grid)
ttgir = compiled.asm.get('ttgir', '')
ptx = compiled.asm.get('ptx', '')

# 检查关键模式
assert 'cp.async' in ptx or 'tma' in ptx  # 确认异步加载生效
assert 'tle.local_pointers' not in ttgir   # 确认已被降低
```

## Autotuner 集成

FlagTree 提供 `auto_adjust_block_sizes` 功能：

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        # ...
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(...):
    ...
```
