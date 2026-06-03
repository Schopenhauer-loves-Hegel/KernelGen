---
title: TLE Core API
scope: 基础数据操作 — 异步加载、tile 提取/插入、前缀和
hardware: sm80+
prerequisites: triton.language 基础 (tl.load, tl.store, tl.arange)
---

# TLE Core API

入口：`import triton.experimental.tle.language as tle`（通常通过 `tle = tl.extra.tle` 访问）

## tle.load

异步加载 hint，告知编译器可以将 load 提前调度。

```python
tle.load(pointer, mask=None, other=None, boundary_check=(),
         padding_option="", cache_modifier="", eviction_policy="",
         volatile=False, is_async=False)
```

- `is_async=True` 时，编译器会尝试将该 load 转换为异步拷贝（cp.async / TMA）
- 编译器 pass `triton-tle-lower-async-load` 负责降低
- 如果硬件不支持或条件不满足，`triton-tle-downgrade-invalid-async-copy` 会自动降级为同步 load

**典型用法：**
```python
a = tle.load(a_ptr + offs, mask=mask, is_async=True)
```

## tle.extract_tile

从大 tensor 中提取一个 tile（子区域）。

```python
tle.extract_tile(x, index, tile_shape) -> tl.tensor
```

- `x`: 源 tensor
- `index`: tile 在各维度的起始索引
- `tile_shape`: 提取的 tile 形状（tuple）
- 编译器 pass `triton-tle-lower-extract-tile` 负责降低

**语义：** 等价于 `x[index[0]:index[0]+tile_shape[0], index[1]:index[1]+tile_shape[1], ...]`

## tle.insert_tile

将一个 tile 插入到大 tensor 的指定位置。

```python
tle.insert_tile(x, tile, index) -> tl.tensor
```

- `x`: 目标 tensor
- `tile`: 要插入的 tile
- `index`: 插入位置的起始索引
- 编译器 pass `triton-tle-lower-insert-tile` 负责降低

**语义：** 等价于 `x[index[0]:..., index[1]:...] = tile`，返回更新后的 tensor

## tle.cumsum

Exclusive 前缀和，始终返回 (exclusive_sum, total_sum) 二元组。

```python
tle.cumsum(input, axis=0, reverse=False, dtype=None) -> (tl.tensor, tl.tensor)
```

- `input`: 输入 tensor
- `axis`: 沿哪个轴计算
- `reverse`: 是否反向累加
- `dtype`: 输出类型（默认与 input 相同）
- 编译器 pass `triton-tle-lower-exclusive-cumsum` 负责降低
- `triton-tle-optimize-exclusive-cumsum-layouts` 优化布局

**返回值：**
- `exclusive_sum[i] = sum(input[0:i])`（不含当前元素）
- `total_sum = sum(input[:])`

**典型用法（TopK 中的 histogram scan）：**
```python
prefix, total = tle.cumsum(histogram, axis=0)
```

## 与标准 tl.* 的关系

| 操作 | TLE 提供 | 标准 Triton |
|------|----------|-------------|
| 加载 | `tle.load` (async hint) | `tl.load` |
| 存储 | — | `tl.store` |
| 点积 | — | `tl.dot` |
| 归约 | — | `tl.reduce` |
| 原子操作 | — | `tl.atomic_*` |
| Scan | `tle.cumsum` (exclusive) | `tl.associative_scan` |
| Tile 操作 | `tle.extract_tile` / `tle.insert_tile` | 手动索引计算 |

TLE 不替代标准 `tl.*`，而是在特定场景提供更高层的抽象或编译器 hint。
