---
title: Vector Add 模式 (Ascend DSA)
version: tag 0.5.0+ascend3.2
scope: 使用 TLE-DSA 在 Ascend NPU 上实现向量加法的基础模式
hardware: Ascend NPU (910B/910_95)
prerequisites: api/dsa.md, compiler/dialect-and-lowering.md
---

# Vector Add 模式 (Ascend DSA)

## 核心思想

Ascend NPU 的向量计算必须在片上 UB (Unified Buffer) 中进行。基本模式是：

1. **分配** UB buffer
2. **搬入** — 从 Global Memory (GM) 拷贝数据到 UB
3. **计算** — 在 UB 中执行向量运算
4. **搬出** — 将结果从 UB 拷贝回 GM

## 内存层次数据流

```
Global Memory (GM)          UB (Unified Buffer)
┌──────────────┐           ┌──────────────┐
│   x_ptr      │──copy──→  │   a_ub       │
│   y_ptr      │──copy──→  │   b_ub       │
└──────────────┘           ├──────────────┤
                           │   c_ub       │ ← add(a_ub, b_ub, c_ub)
┌──────────────┐           ├──────────────┤
│  output_ptr  │←─copy───  │   c_ub       │
└──────────────┘           └──────────────┘
```

## 完整 Kernel 示例

```python
import torch
import torch_npu
import triton
import triton.language as tl
import triton.experimental.tle as tle


@triton.jit
def add_kernel(
    x_ptr,        # 第一个输入向量指针
    y_ptr,        # 第二个输入向量指针
    output_ptr,   # 输出向量指针
    n_elements,   # 向量总长度
    BLOCK_SIZE: tl.constexpr,  # 每个 program 处理的元素数
):
    # 1. 计算当前 block 的偏移
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 2. 在 UB 中分配三个 buffer
    a_ub = tle.dsa.alloc([BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    b_ub = tle.dsa.alloc([BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    c_ub = tle.dsa.alloc([BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)

    # 3. 计算实际搬运大小（处理尾部不足一个 block 的情况）
    t0 = n_elements - block_start
    tail_size = tl.minimum(t0, BLOCK_SIZE)

    # 4. 从 GM 搬入到 UB
    tle.dsa.copy(x_ptr + offsets, a_ub, [tail_size])
    tle.dsa.copy(y_ptr + offsets, b_ub, [tail_size])

    # 5. 在 UB 中执行向量加法
    tle.dsa.add(a_ub, b_ub, c_ub)

    # 6. 从 UB 搬出到 GM
    tle.dsa.copy(c_ub, output_ptr + offsets, [tail_size])
```

## Host 端调用

```python
def custom_func(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=128)
    return output

# 使用示例
x = torch.rand(1024, dtype=torch.float).npu()
y = torch.rand(1024, dtype=torch.float).npu()
result = custom_func(x, y)
```

## 关键要点

### 1. alloc 必须指定地址空间

```python
# 正确 — 必须显式指定 mem_addr_space
a_ub = tle.dsa.alloc([BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)

# 错误 — mem_addr_space 不能为 None
# a_ub = tle.dsa.alloc([BLOCK_SIZE], dtype=tl.float32, mem_addr_space=None)  # AssertionError
```

### 2. copy 的 shape 参数用于尾部处理

`shape` 参数指定实际搬运的元素数，允许动态值。这对处理最后一个 block（元素数可能不足 BLOCK_SIZE）至关重要。

### 3. 向量运算作用于整个 buffer

`tle.dsa.add(a_ub, b_ub, c_ub)` 对 buffer 全部元素操作。buffer 的 shape 和 dtype 必须一致。

### 4. 不需要 mask 机制

与标准 Triton 的 `tl.load(..., mask=mask)` 不同，DSA 模式通过 `copy` 的 shape 参数实现边界处理，无需显式 mask。

## 编译降低路径

```
tle.dsa.alloc    → memref.alloc (带 UB address space 属性)
tle.dsa.copy     → memref.copy (GM↔UB 方向)
tle.dsa.add      → hivm.VAddOp (向量加法指令)
```

## 参考

- 完整可运行测试：`python/test/tle/test_vec_add.py`
- 相关变体：`python/test/tle/test_vec_add_2d.py`（二维版本）、`python/test/tle/test_vec_add_mix.py`（混合精度）
- Ascend 官方 tutorial：`third_party/ascend/tutorials/01-vector-add.py`（标准 Triton 风格，无 DSA）
