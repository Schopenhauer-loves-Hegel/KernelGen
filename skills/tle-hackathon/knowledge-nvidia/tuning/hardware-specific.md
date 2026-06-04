---
title: 硬件特定差异
scope: sm80/sm89/sm90/sm100 各代 GPU 对 TLE 功能的支持差异
hardware: sm80+
prerequisites: compiler/pass-pipeline.md
---

# 硬件特定差异

## 各代 GPU 能力对照

| 功能 | sm80 (A100) | sm89 (L40/4090) | sm90 (H100/H200) | sm100+ |
|------|:-----------:|:---------------:|:-----------------:|:------:|
| Shared memory staging | ✅ | ✅ | ✅ | ✅ |
| Async copy (cp.async) | ✅ | ✅ | ✅ | ✅ |
| TMA copy | ❌ | ❌ | ✅ | ✅ |
| WGMMA | ❌ | ❌ | ✅ | ✅ |
| Tensor memory (tmem) | ❌ | ❌ | ❌ | ✅ |
| Tile-style pipeline | ✅ | ✅ | ✅ | ❌ |
| Warp Specialization | ❌ | ❌ | ✅ | ✅ |
| Cluster launch | ❌ | ❌ | ✅ | ✅ |

## 编译器条件分支

`third_party/nvidia/backend/compiler.py` 中根据 `capability // 10` 分支：

### sm80-sm99 路径 (`capability // 10 in [8, 9]`)

```
downgrade_invalid_async_copy → tile_style_pipeline_schedule → materialize_tile_style_pipeline
```

- 使用 TileLang 风格 preload-2 pipeline（sm90 也走此路径）
- sm80/sm89 不支持 TMA，使用 cp.async；sm90 额外支持 TMA（见下方 TMA 降低条件）

### sm100+ 路径 (`capability // 10 >= 10`)

```
downgrade_invalid_async_copy
```

- 跳过 tile-style pipeline（使用 Warp Specialization 替代）
- 支持 tensor memory

### TMA 降低条件

```
if capability // 10 >= 9:
    lower_tma_copy
```

仅 sm90+ 执行 `triton-tle-lower-tma-copy`。

## 对 kernel 编写的影响

### sm80/sm89 最佳实践

```python
# 使用 async load + pipeline
for k in tle.gpu.pipeline(0, K // BK, num_stages=3):
    a = tle.load(a_ptr + ..., is_async=True)  # → cp.async
    b = tle.load(b_ptr + ..., is_async=True)
    acc = tl.dot(a, b, acc)
```

### sm90 (Hopper) 最佳实践

```python
# 可以使用 TMA copy + WGMMA layout
a_smem = tle.gpu.alloc([BM, BK], dtype=tl.float16, nv_mma_shared_layout=True)
tle.gpu.copy(a_ptr + ..., a_smem, shape=[BM, BK])  # → TMA
```

### 跨代兼容写法

```python
# 编译器会根据硬件自动选择最优路径
# is_async=True 在 sm80 → cp.async, sm90 → 可能 TMA
a = tle.load(a_ptr + ..., is_async=True)
```

## Warp Specialization (sm90+)

Hopper 引入 Warp Specialization，由 NVWS dialect 管理：

- `third_party/nvidia/include/Dialect/NVWS/IR/NVWSOps.td`
- Pass: `NVWSLowerWarpGroup`, `NVWSAssignStagePhase`, `NVWSLowerAref`
- 入口: `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization.cpp`

TLE 的 tile-style pipeline 在 sm90+ 上不执行，因为 Warp Specialization 提供了更好的延迟隐藏。

## alloc scope 选择

| Scope | 硬件要求 | 用途 |
|-------|----------|------|
| `tle.smem` | sm80+ | Shared memory（默认） |
| `tle.tmem` | sm100+ | Tensor memory（更大容量） |

## 性能建议

| 硬件 | 关键调优点 |
|------|-----------|
| sm80 | num_stages 对性能影响大（cp.async 延迟高） |
| sm89 | 类似 sm80，但 L2 cache 更大 |
| sm90 | 优先使用 TMA + WGMMA，减少 num_stages 依赖 |
| sm100+ | 考虑 tensor memory 减少 shared memory 压力 |

### 非 NVIDIA 后端 TLE 支持现状

| 后端 | TLE 集成状态 |
|------|-------------|
| HCU | 携带 TLE dialect 定义（`#ifdef __TLE__` 条件编译） |
| Mthreads | 初步集成：1 个 TLE pass (`add_tle_lower_async_load`) + 测试 |
| Enflame | 1 个 TLE lowering pass (`-tle-to-triton-gcu`) |
| AMD | 无 TLE 代码 |
