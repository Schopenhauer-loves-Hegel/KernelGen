---
title: TLE DSA Dialect 与 Ascend 编译降低
version: tag 0.5.0+ascend3.2
scope: 理解 TLE DSA IR 操作定义、到 HIVM/linalg 的转换模式、以及 Ascend NPU 编译管线
hardware: Ascend NPU (910B/910_95)
prerequisites: MLIR 基础概念、memref 语义、Triton 编译流程
---

# TLE DSA Dialect 与 Ascend 编译降低

## 概述

TLE DSA dialect 定义了面向 Domain-Specific Accelerator 的 IR 操作集，当前主要服务于 Ascend NPU 后端。该 dialect 定义了 7 个活跃 IR ops（1 个 dot op 已注释），通过 2 个 C++ 转换模式降低到 HIVM/memref 层级。

源码位置：`third_party/tle/dsa/`

## TLE DSA IR Operations（7 个活跃）

定义文件：`third_party/tle/dsa/dialect/include/IR/TleOps.td`

| Op | 操作数 | 语义 |
|----|--------|------|
| `tle.dsa_copy` | src, dst, shape(variadic I32) | DMA 数据搬运（跨内存区域） |
| `tle.dsa_add` | lhs, rhs, res (AnyMemRef) | 逐元素加法 |
| `tle.dsa_sub` | lhs, rhs, res (AnyMemRef) | 逐元素减法 |
| `tle.dsa_mul` | lhs, rhs, res (AnyMemRef) | 逐元素乘法 |
| `tle.dsa_div` | lhs, rhs, res (AnyMemRef) | 逐元素除法 |
| `tle.dsa_max` | lhs, rhs, res (AnyMemRef) | 逐元素取最大值 |
| `tle.dsa_min` | lhs, rhs, res (AnyMemRef) | 逐元素取最小值 |

**注释掉的 Op：**
- `tle.dsa_dot` — 矩阵乘：`d = matmul(a, b) + c`，带 size/initC/traA/traB/enableHf32 属性。当前未启用。

**共同约束：**
- 算术 ops 带有 `SameOperandsShape` 和 `SameOperandsElementType` traits
- 所有 ops 带有 `MemoryEffects<[MemWrite]>` 副作用标注
- 所有 ops 继承 `VerifyTensorLayoutsTrait`

---

## 转换模式（Conversion Patterns）

**特点：** 没有 `Passes.td` 文件 — 直接注册为 RewritePattern，由上层 pass 调用。

### 1. DSACopyConverter

文件：`third_party/tle/dsa/dialect/lib/Conversion/TleToLinalg/DSACopyConverter.cpp`

将 `tle.dsa_copy` 根据源/目标地址空间降低为不同操作：

| 源空间 | 目标空间 | 降低目标 |
|--------|----------|----------|
| GM | UB | `memref.copy`（通用 DMA） |
| UB | GM | `memref.copy`（通用 DMA） |
| GM | L1 | `hivm.ND2NZOp`（数据格式转换：Normal Data → NZ 格式） |

**处理流程：**
1. 将 I32 shape 转为 index 类型
2. 对 src/dst 生成 `memref.subview`（根据动态 shape 切片）
3. 提取 MemRefType 的 address space 属性
4. 根据地址空间组合选择目标 op

**注释掉的路径：** L0C → GM 的 `hivm.FixpipeOp`（NZ→ND 格式回写）暂未实现。

注册入口：
```cpp
void populateTleCopyOpConversionPatterns(TypeConverter &, RewritePatternSet &);
```

### 2. MathConverter

文件：`third_party/tle/dsa/dialect/lib/Conversion/TleToLinalg/MathConverter.cpp`

使用模板化的 `BinaryMathConverter` 将 6 个算术 ops 一对一映射到 HIVM 向量指令：

| TLE Op | HIVM Op |
|--------|---------|
| `DSAAddOp` | `hivm::VAddOp` |
| `DSASubOp` | `hivm::VSubOp` |
| `DSAMulOp` | `hivm::VMulOp` |
| `DSADivOp` | `hivm::VDivOp` |
| `DSAMaxOp` | `hivm::VMaxOp` |
| `DSAMinOp` | `hivm::VMinOp` |

注释掉的：`MatMulConverter<DSADotOp>` 矩阵乘转换。

注册入口：
```cpp
void populateTleMathOpConversionPatterns(TypeConverter &, RewritePatternSet &);
```

---

## pybind11 绑定层

文件：`third_party/tle/dsa/tle_ir.cc`

定义 `DSAOpBuilder` 类（继承 `TritonOpBuilder`），暴露以下方法到 Python：

| Python 方法 | 生成的 MLIR Op |
|-------------|----------------|
| `create_dsa_alloc(memrefType)` | `memref.alloc` |
| `create_dsa_copy(src, dst, shape, inter_no_alias)` | `tle.dsa_copy` |
| `create_dsa_add/sub/mul/div/max/min(lhs, rhs, res)` | 对应的 `tle.dsa_*` op |
| `dsa_to_buffer(src, addressSpace)` | `bufferization.to_memref` |
| `dsa_to_tensor(src, writable)` | `bufferization.to_tensor` |
| `create_dsa_extract_scalar(src, indices)` | `tensor.extract` |
| `create_dsa_extract_slice(ful, offs, sizes, strides)` | `tensor.extract_slice` |
| `create_dsa_insert_slice(ful, sub, offs, sizes, strides)` | `tensor.insert_slice` |
| `create_dsa_subview(source, offsets, sizes, strides)` | `memref.subview` |
| `dsa_get_buffer_type(shape, elemTy, memSpace)` | `MemRefType` 构造 |
| `dsa_get_buffer_type_with_strides(...)` | 带 StridedLayout 的 `MemRefType` |

还包括 `load_dialects` 函数，注册 `memref`, `bufferization`, `tle` 三个 dialect 到 context。

---

## Ascend 编译管线

文件：`third_party/ascend/backend/compiler.py`（`AscendBackend.add_stages()`）

### NPU 后端流程

```
Triton IR (ttir)
    │
    ▼  make_ttir()  — 通用优化 (inline, combine, canonicalize, CSE, LICM)
Optimized TTIR
    │
    ▼  ttir_to_linalg()  — Ascend 特有 pass pipeline
TTAdapter MLIR (linalg)
    │
    ▼  bishengir-compile  — NPU 原生编译器
NPU Binary (.o)
```

### ttir_to_linalg() 阶段的 Ascend 特有 Pass（按顺序）

调用位于 `ascend.passes.ttir.*`：

1. `add_triton_to_structure_incubated` — 结构化转换（mask 处理、动态 offset 优化）
2. `add_discrete_mask_access_conversion` — 离散 mask 访问转换
3. `add_triton_to_annotation` — 注解添加
4. `add_triton_to_unstructure_incubated` — 非结构化转换
5. **`add_triton_to_hivm`** — Triton → HIVM dialect 转换
6. **`add_triton_to_hfusion`** — Triton → HFusion dialect 转换
7. **`add_triton_to_llvm`** — Triton → LLVM dialect 转换
8. `add_bubble_up_operation` — 操作上提优化
9. `add_triton_to_structure_incubated` — 第二轮结构化转换
10. `add_triton_to_linalg_incubated` — 最终 linalg 降低

### 3 个 Ascend 特有 Conversion Passes

定义在 `third_party/ascend/include/`:

| Pass | 文件 | 功能 | 依赖 Dialect |
|------|------|------|------|
| `triton-to-hivm` | `TritonToHIVM/Passes.td` | Triton → HIVM dialect（向量/DMA 指令） | `hivm::HIVMDialect` |
| `triton-to-hfusion` | `TritonToHFusion/Passes.td` | Triton → HFusion dialect（算子融合） | `hfusion::HFusionDialect` |
| `triton-to-llvm` | `TritonToLLVM/Passes.td` | Triton → LLVM dialect（标量/控制流） | `LLVM::LLVMDialect`, `tensor::TensorDialect`, `arith::ArithDialect` |

### bishengir-compile 阶段

最终二进制由华为 `bishengir-compile` 编译器生成，主要选项：
- `--target=<arch>` — 目标硬件架构
- `--enable-hfusion-compile=true` — 启用 HFusion 编译
- `--enable-triton-kernel-compile=true` — Triton kernel 编译模式
- `--enable-auto-multi-buffer=<bool>` — 自动多缓冲
- `--enable-ubuf-saving=<bool>` — UB 空间节省优化

### SIMT-Only 模式快速路径

当 `force_simt_only=True` 时，跳过 linalg 转换，直接：
```
TTIR → bishengir-compile --enable-triton-ir-compile --pure-simt → NPU Binary
```

### CPU 后端流程（开发调试用）

```
TTIR → linalg → LLVM MLIR → LLVM IR → Assembly (.s)
```

使用 `mlir-opt` 标准 pass 链：linalg → affine loops → bufferize → scf → cf → LLVM。

---

## TLE DSA 在 Ascend 编译中的位置

TLE DSA ops（`tle.dsa_copy`, `tle.dsa_add` 等）在 `ttir_to_linalg()` 阶段被 `DSACopyConverter` 和 `MathConverter` 消费，转换为 `memref.copy` / `hivm.ND2NZOp` / `hivm.VAddOp` 等低层 ops，随后由 `bishengir-compile` 进一步编译为 NPU 原生指令。
