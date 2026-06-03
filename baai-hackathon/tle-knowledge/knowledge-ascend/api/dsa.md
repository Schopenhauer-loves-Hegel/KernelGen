---
title: TLE DSA API (Ascend)
version: tag 0.5.0+ascend3.2
scope: DSA 层内存管理、数据搬运、向量计算、类型转换、slice 操作
hardware: Ascend NPU (910B/910_95)
prerequisites: triton.language 基础 (tl.load, tl.store, tl.arange), Ascend 内存层次理解
---

# TLE DSA API (Ascend)

入口：`import triton.experimental.tle as tle`，使用 `tle.dsa.*` 访问 DSA 接口。

Ascend 地址空间通过 `tle.dsa.ascend.*` 访问（UB, L1, L0A, L0B, L0C）。

## Ascend 内存层次

```
┌─────────────────────────────────────────┐
│  Global Memory (GM) — HBM/DDR          │  ← tl.load / tl.store 操作的空间
├─────────────────────────────────────────┤
│  L1 Buffer — 大容量片上缓存              │  ← 矩阵搬运中转 (GM→L1 使用 ND2NZ 格式转换)
├─────────────────────────────────────────┤
│  UB (Unified Buffer) — 向量计算缓存     │  ← 向量运算的主要工作空间
├─────────────────────────────────────────┤
│  L0A / L0B — 矩阵计算输入缓存           │  ← Cube 单元的左/右矩阵输入
├─────────────────────────────────────────┤
│  L0C — 矩阵计算累加器                   │  ← Cube 单元的输出累加
└─────────────────────────────────────────┘
```

## 地址空间常量

```python
tle.dsa.ascend.UB   # Unified Buffer — 向量计算工作空间
tle.dsa.ascend.L1   # L1 Buffer — 大容量中转
tle.dsa.ascend.L0A  # 矩阵单元左输入
tle.dsa.ascend.L0B  # 矩阵单元右输入
tle.dsa.ascend.L0C  # 矩阵单元累加器输出
```

定义来源：`python/triton/experimental/tle/language/dsa/ascend/core.py` 引用 `triton.language.extra.cann.extension.core.ascend_address_space`。

---

## tle.dsa.alloc

在指定地址空间分配片上 buffer。

```python
tle.dsa.alloc(shape: List[constexpr], dtype: tl.dtype, mem_addr_space: address_space) -> buffer
```

- `shape`: buffer 形状，例如 `[128]` 或 `[64, 64]`
- `dtype`: 元素类型，例如 `tl.float32`, `tl.float16`
- `mem_addr_space`: 目标地址空间（必须提供）

**典型用法：**
```python
a_ub = tle.dsa.alloc([BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
```

**底层实现：** 生成 `memref.alloc` 操作，带有 address space 属性。

---

## tle.dsa.copy

在不同内存区域之间进行 DMA 数据搬运。

```python
tle.dsa.copy(src, dst, shape: List[int|constexpr], inter_no_alias: bool = False)
```

- `src`: 源地址（pointer 或 buffer）
- `dst`: 目标地址（pointer 或 buffer）
- `shape`: 搬运形状（不能为空）
- `inter_no_alias`: 若为 True，标注不同迭代间无别名（用于 pipeline 优化）

**典型搬运方向：**
- `GM → UB`: 从全局内存搬入向量计算缓存
- `UB → GM`: 计算结果写回全局内存
- `GM → L1`: 全局到 L1 中转（使用 ND2NZ 格式转换）

**典型用法：**
```python
tle.dsa.copy(x_ptr + offsets, a_ub, [tail_size])   # GM → UB
tle.dsa.copy(c_ub, output_ptr + offsets, [tail_size])  # UB → GM
```

**底层 IR：** 生成 `tle.dsa_copy` op，后续由 `DSACopyConverter` 降低。

---

## tle.dsa.to_buffer

将 tensor 转换为 buffer（memref）。

```python
tle.dsa.to_buffer(tensor: tl.tensor, space: address_space = None, bind_buffer: buffer = None) -> buffer
```

- `tensor`: 要转换的输入 tensor（必须有 shape）
- `space`: 可选的地址空间标注
- `bind_buffer`: 必须为 None（保留参数）

**底层实现：** 生成 `bufferization.to_memref` 操作。

---

## tle.dsa.to_tensor

将 buffer（memref）转换回 tensor。

```python
tle.dsa.to_tensor(memref: buffer, writable: bool = True, target_shape = None) -> tl.tensor
```

- `memref`: 输入 buffer
- `writable`: 若为 True，生成的 tensor 在 bufferization 阶段标记为可写
- `target_shape`: 可选的目标形状（若提供且不同于 memref.shape，会插入 convert_layout）

**底层实现：** 生成 `bufferization.to_tensor` 操作。

---

## tle.dsa.subview

创建 buffer 的子视图。

```python
tle.dsa.subview(src: buffer, offsets: List[constexpr|int|tensor], sizes: List[constexpr|int], strides: List[constexpr|int]) -> buffer
```

- `src`: 源 buffer
- `offsets`: 每个维度的起始偏移（支持动态 tensor 值）
- `sizes`: 每个维度的大小（正整数）
- `strides`: 每个维度的步长（正整数）

**约束：** sizes 和 strides 必须为正值；offset 不能为负值。

**底层实现：** 生成 `memref.subview` 操作，并计算结果 buffer 的内存布局 strides。

---

## 向量运算 (Elementwise Ops)

所有向量运算接受三个 buffer 参数：输入 A、输入 B、输出 result。操作在 buffer 的全部元素上逐元素执行。

```python
tle.dsa.add(input: buffer, other: buffer, result: buffer)  # result = input + other
tle.dsa.sub(input: buffer, other: buffer, result: buffer)  # result = input - other
tle.dsa.mul(input: buffer, other: buffer, result: buffer)  # result = input * other
tle.dsa.div(input: buffer, other: buffer, result: buffer)  # result = input / other
tle.dsa.max(input: buffer, other: buffer, result: buffer)  # result = max(input, other)
tle.dsa.min(input: buffer, other: buffer, result: buffer)  # result = min(input, other)
```

**注意：** 三个参数的 shape 和 element type 必须一致（IR 层面有 `SameOperandsShape` 和 `SameOperandsElementType` 约束）。

**底层 IR：** 各自生成对应的 `tle.dsa_add` / `dsa_sub` / ... op，由 `MathConverter` 降低为 HIVM 向量指令。

---

## Tensor Slice 操作

### tle.dsa.extract_slice

从 tensor 中提取子区域。

```python
tle.dsa.extract_slice(ful: tensor, offsets: List, sizes: List[int], strides: List[int]) -> tensor
```

- `ful`: 源 tensor
- `offsets`: 各维度起始偏移
- `sizes`: 提取区域的各维度大小（均 >= 1）
- `strides`: 各维度步长（均 >= 0）

返回形状为 `sizes` 的新 tensor。底层生成 `tensor.extract_slice` 操作。

### tle.dsa.insert_slice

将 tensor 插入到另一个 tensor 的指定位置。

```python
tle.dsa.insert_slice(ful: tensor, sub: tensor, offsets: List, sizes: List[int], strides: List[int]) -> tensor
```

- `ful`: 目标 tensor（接收者）
- `sub`: 要插入的 tensor
- `offsets`, `sizes`, `strides`: 同 extract_slice

返回更新后的 tensor。底层生成 `tensor.insert_slice` 操作。

### tle.dsa.extract_element

从 tensor 中提取单个标量元素。

```python
tle.dsa.extract_element(src: tensor, indice: List) -> scalar
```

- `src`: 源 tensor
- `indice`: 索引列表（长度必须等于 tensor rank）

返回标量值。底层生成 `tensor.extract` 操作。

---

## tle.dsa.hint

编译器提示上下文管理器。用于 AST 解析阶段向编译器传递额外信息。

```python
with tle.dsa.hint(**kwargs):
    ...
```

**注意：** 不能在 JIT 函数外直接调用（会 raise RuntimeError）。在 JIT 编译期间由 AST 解析器处理。

---

## tle.dsa.pipeline

软件流水线循环迭代器。

```python
for i in tle.dsa.pipeline(start, end=None, step=None, num_stages=None, loop_unroll_factor=None):
    ...
```

继承自 `tle.dsa.range`，添加软件流水线语义。允许编译器将循环流水化为多个阶段（多个迭代同时 in-flight）。

- `num_stages`: 流水线阶段数
- `loop_unroll_factor`: 循环展开因子

---

## tle.dsa.parallel

并行循环迭代器。

```python
for i in tle.dsa.parallel(start, end=None, step=None, num_stages=None, loop_unroll_factor=None):
    ...
```

继承自 `tle.dsa.range`，标示循环迭代间无数据依赖，可并行执行。

---

## 类型系统

### address_space

抽象基类，表示 buffer 的地址空间。具体实现由后端提供（Ascend: UB/L1/L0A/L0B/L0C）。

```python
class address_space:
    def to_ir(self, builder) -> ir.type: ...
```

### buffer_type

描述 buffer 的完整类型信息。

```python
class buffer_type(tl.dtype):
    element_ty: tl.dtype    # 元素类型
    shape: List[int]        # buffer 形状
    space: address_space    # 地址空间
    strides: List[int]      # 内存布局步长（可选）
```

### buffer

buffer 值类型，持有 IR handle 和类型元信息。

```python
class buffer(tl._value):
    type: buffer_type
    dtype: tl.dtype          # element_ty.scalar
    shape: List[int]
    space: address_space
    strides: List[int]
```

---

## 完整 API 导出列表

```python
from triton.experimental.tle.language.dsa import (
    alloc, copy, pipeline, parallel,
    to_tensor, to_buffer,
    add, sub, mul, div, max, min,
    hint, extract_slice, insert_slice, extract_element,
    subview, ascend,
)
```
