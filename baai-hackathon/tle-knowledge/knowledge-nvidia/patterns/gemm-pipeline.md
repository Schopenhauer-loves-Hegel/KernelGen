---
title: GEMM Pipeline 模式
scope: 使用 TLE pipeline + alloc + async load 实现高性能矩阵乘法
hardware: sm80+ (pipeline), sm90+ (TMA copy)
prerequisites: api/gpu.md, compiler/pass-pipeline.md
---

# GEMM Pipeline 模式

## 核心思想

GEMM 的性能瓶颈通常在于 global memory 到 shared memory 的数据搬运延迟。TLE 通过以下机制隐藏延迟：

1. `tle.gpu.pipeline` — 多阶段循环，自动管理 buffer 轮转
2. `tle.load(..., is_async=True)` — 异步加载 hint
3. `tle.gpu.alloc` — 显式 shared memory 分配（可选，用于精细控制）

## 基础 Pipeline GEMM

```python
@triton.jit
def gemm_pipeline(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in tle.gpu.pipeline(0, K // BLOCK_K, num_stages=3):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        a = tle.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            is_async=True
        )
        b = tle.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            is_async=True
        )
        acc = tl.dot(a, b, acc)

    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc)
```

## 编译器行为

上述代码经过以下 pass 变换：

1. `triton-tle-promote-local-store-staging` — 识别 async load 的目标，提升为 pipeline buffer
2. `triton-tle-tile-style-pipeline-schedule` — 重写为 preload-2 schedule（提前 2 个迭代加载）
3. `triton-tle-materialize-tile-style-pipeline` — 物化为显式多 buffer 循环

最终生成的代码等价于手写的多 buffer pipeline，但由编译器自动管理 buffer 分配和同步。

## 显式 Shared Memory 控制（高级）

当需要精确控制 layout 时，可以显式分配：

```python
a_smem = tle.gpu.alloc([BLOCK_M, BLOCK_K], dtype=tl.float16, nv_mma_shared_layout=True)
b_smem = tle.gpu.alloc([BLOCK_K, BLOCK_N], dtype=tl.float16, nv_mma_shared_layout=True)

for k in tle.gpu.pipeline(0, K // BLOCK_K, num_stages=3):
    tle.gpu.copy(a_ptr + ..., a_smem, shape=[BLOCK_M, BLOCK_K])
    tle.gpu.copy(b_ptr + ..., b_smem, shape=[BLOCK_K, BLOCK_N])
    a = tl.load(tle.gpu.local_ptr(a_smem))
    b = tl.load(tle.gpu.local_ptr(b_smem))
    acc = tl.dot(a, b, acc)
```

## 调优要点

| 参数 | 影响 | 建议范围 |
|------|------|----------|
| `BLOCK_M/N` | 计算密度 vs occupancy | 64-256 |
| `BLOCK_K` | 内存带宽利用 | 32-128 |
| `num_stages` | 延迟隐藏 vs shared memory 占用 | 2-5 |
| `num_warps` | 并行度 | 4-8 |
| `nv_mma_shared_layout` | WGMMA 兼容性 | True for GEMM |

## 参考实现

- `python/tutorials/tle/deepseek_v32/02-sparse-mla.py` — Sparse MLA 中的 TLE GEMM 变体
