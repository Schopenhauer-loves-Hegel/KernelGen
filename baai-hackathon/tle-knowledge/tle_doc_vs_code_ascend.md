# TLE 文档与源码对照分析 (Ascend NPU — tag `0.5.0+ascend3.2`)

本文档对比 TLE 设计描述与仓库 `0.5.0+ascend3.2`（triton v3.2.x 分支，Ascend NPU 后端）实际源码的对应关系。

---

## 1. 总体架构对照

### 文档描述的三层设计

| 文档概念 | 代码实现 | 对应路径 |
|----------|----------|----------|
| Level 1: Numpy/PyTorch 算法级 | **未实现** — 不存在高层算法级 API | — |
| Level 2: Tile 级编程 + 分布式 | **未实现** — 无 GPU Tile 级 API，无分布式 API | — |
| Level 3: 硬件特定扩展 | TLE-DSA — Ascend NPU 专用 buffer/DMA 操作 | `python/triton/experimental/tle/language/dsa/` |

### 实际代码的架构

| 概念层 | 定位 | 入口 |
|--------|------|------|
| **TLE-DSA** | Ascend NPU 缓冲区管理与数据搬运（UB/L1/L0 地址空间操作） | `tle/language/dsa/core.py`, `tle/language/dsa/types.py`, `tle/language/dsa/semantic.py` |

> 注：ascend3.2 版本的 TLE 与 triton_v3.6.x 完全不同。v3.6.x 的 TLE 面向 GPU（shared memory、pipeline、warp specialization），而 ascend3.2 的 TLE 是面向 Ascend NPU 的 DSA (Domain-Specific Accelerator) API，操作对象是 NPU 的多级存储层次（UB、L1、L0A/B/C）。无 GPU 特定 API 源码（仅存在 .pyc），无分布式 API，无 Raw API 源码。

---

## 2. 前端 API 对照

### 已实现的 API（全部通过 `tle.dsa.*` 命名空间）

| 文档描述 | 实际 API | 源码位置 |
|----------|----------|----------|
| Buffer 分配 | `tle.dsa.alloc(shape, dtype, mem_addr_space)` — 在指定地址空间分配 buffer | `python/triton/experimental/tle/language/dsa/core.py` |
| DMA 拷贝 | `tle.dsa.copy(src, dst, shape, inter_no_alias=False)` — 内存区域间的 DMA 搬运 | `python/triton/experimental/tle/language/dsa/core.py` |
| Tensor → Buffer | `tle.dsa.to_buffer(tensor, space, bind_buffer)` — 将 tensor 转为 buffer | `python/triton/experimental/tle/language/dsa/core.py` |
| Buffer → Tensor | `tle.dsa.to_tensor(memref, writable, target_shape)` — 将 buffer 转为 tensor | `python/triton/experimental/tle/language/dsa/core.py` |
| Subview | `tle.dsa.subview(src, offsets, sizes, strides)` — 创建 buffer 的子视图 | `python/triton/experimental/tle/language/dsa/core.py` |
| 逐元素加法 | `tle.dsa.add(input, other, result)` | `python/triton/experimental/tle/language/dsa/core.py` |
| 逐元素减法 | `tle.dsa.sub(input, other, result)` | `python/triton/experimental/tle/language/dsa/core.py` |
| 逐元素乘法 | `tle.dsa.mul(input, other, result)` | `python/triton/experimental/tle/language/dsa/core.py` |
| 逐元素除法 | `tle.dsa.div(input, other, result)` | `python/triton/experimental/tle/language/dsa/core.py` |
| 逐元素最大值 | `tle.dsa.max(input, other, result)` | `python/triton/experimental/tle/language/dsa/core.py` |
| 逐元素最小值 | `tle.dsa.min(input, other, result)` | `python/triton/experimental/tle/language/dsa/core.py` |
| Tensor 切片提取 | `tle.dsa.extract_slice(ful, offsets, sizes, strides)` | `python/triton/experimental/tle/language/dsa/core.py` |
| Tensor 切片插入 | `tle.dsa.insert_slice(ful, sub, offsets, sizes, strides)` | `python/triton/experimental/tle/language/dsa/core.py` |
| 标量提取 | `tle.dsa.extract_element(src, indice)` | `python/triton/experimental/tle/language/dsa/core.py` |
| 编译器 hint | `tle.dsa.hint(**kwargs)` — 编译器提示上下文管理器 | `python/triton/experimental/tle/language/dsa/core.py` |
| 软件流水循环 | `tle.dsa.pipeline(...)` — 带软件流水语义的循环（`range` 子类） | `python/triton/experimental/tle/language/dsa/core.py` |
| 并行循环 | `tle.dsa.parallel(...)` — 带并行语义的循环（`range` 子类） | `python/triton/experimental/tle/language/dsa/core.py` |

### DSA 类型系统

| 类型 | 说明 | 源码位置 |
|------|------|----------|
| `buffer` | DSA buffer 实例 | `python/triton/experimental/tle/language/dsa/types.py` |
| `buffer_type` | DSA buffer 类型 | `python/triton/experimental/tle/language/dsa/types.py` |
| `address_space` | 地址空间枚举 | `python/triton/experimental/tle/language/dsa/types.py` |

### Ascend 地址空间

| 地址空间 | 说明 | 源码位置 |
|----------|------|----------|
| `UB` | Unified Buffer — 通用缓冲区 | `python/triton/experimental/tle/language/dsa/ascend/__init__.py` |
| `L1` | Level 1 缓存 | `python/triton/experimental/tle/language/dsa/ascend/__init__.py` |
| `L0A` | Level 0 Matrix A 缓冲区 | `python/triton/experimental/tle/language/dsa/ascend/__init__.py` |
| `L0B` | Level 0 Matrix B 缓冲区 | `python/triton/experimental/tle/language/dsa/ascend/__init__.py` |
| `L0C` | Level 0 Matrix C（累加器）缓冲区 | `python/triton/experimental/tle/language/dsa/ascend/__init__.py` |

### v3.6.x 中存在但 ascend3.2 中不存在的 API

| API | 状态 | 说明 |
|-----|------|------|
| `tle.extract_tile` / `tle.insert_tile` | ❌ 不存在 | v3.6.x 的 Tile 语义操作未移植 |
| `tle.load(..., is_async=True)` | ❌ 不存在 | 异步加载通过 DSA `copy` 实现 |
| `tle.cumsum` | ❌ 不存在 | 无 scan 类操作 |
| `tle.gpu.*` (alloc/copy/pipeline/warp_specialize) | ❌ 不存在 | 无 GPU 特定 API（仅 .pyc 文件） |
| `tle.pipe` | ❌ 不存在 | 无 pipeline 数据流管道 |
| `tle.device_mesh` / `tle.sharding` / `tle.remote` | ❌ 不存在 | 无分布式 API |
| `tle.raw.call` / `tle.raw.call_smem` | ❌ 不存在 | 无 Raw 内联 API 源码 |
| `tle.dsa.dot` | 🚧 已注释 | IR 定义中 `dsa_dot` 被注释掉，未实现 |

---

## 3. 中端编译器对照

### TLE-DSA MLIR Dialect — 已实现

| 组件 | 源码位置 |
|------|----------|
| Dialect 定义 | `third_party/tle/dsa/dialect/include/IR/TleDialect.td` (namespace "tle") |
| Operations 定义 | `third_party/tle/dsa/dialect/include/IR/TleOps.td` |
| Pybind11 绑定 | `third_party/tle/dsa/dialect/tle_ir.cc` — `DSAOpBuilder` 类 |

### 定义的 IR Operations（7 个）

```
tle.dsa_copy,
tle.dsa_add, tle.dsa_sub, tle.dsa_mul,
tle.dsa_div, tle.dsa_max, tle.dsa_min
```

> 注：`dsa_dot` 在 `TleOps.td` 中已被注释掉，未实现。

### 转换模式（2 个，直接 C++ 实现，无 Passes.td）

| 转换器 | 实现文件 | 功能 |
|--------|----------|------|
| `DSACopyConverter` | `DSACopyConverter.cpp` | 降低 `tle.dsa_copy` → `memref.copy`（GM↔UB）或 `hivm.ND2NZOp`（GM→L1） |
| `MathConverter` | `MathConverter.cpp` | 降低数学运算 → HIVM 向量操作（`VAddOp`, `VSubOp`, `VMulOp`, `VDivOp`, `VMaxOp`, `VMinOp`） |

> 注：与 v3.6.x 的 21 个注册 Pass 不同，ascend3.2 不使用 MLIR Pass 基础设施（无 `Passes.td`），而是直接在 C++ 中实现转换模式。

### 与 v3.6.x 中端的对比

| 方面 | ascend3.2 | v3.6.x |
|------|-----------|--------|
| IR Operations 数量 | 7 个（dsa_*） | 24 个（tile, pipe, memory, distributed） |
| MLIR Passes | 0 个（直接转换） | 21 个注册 Pass |
| Dialect namespace | "tle" | "tle" |
| 降低目标 | HIVM / memref | Triton GPU dialect / NVGPU / NVWS |
| 对 Triton 上游的修改 | 无 | 4 处（tt.load hints, atomic_rmw, tma_copy, SharedMemory） |

### 文档提及但未实现的编译器特性

| 特性 | 状态 |
|------|------|
| 降低到 Linalg/Tensor/SCF Dialect | ❌ DSA ops 通过 HIVM/memref 降低，不走 Linalg |
| 自动 Tiling Pass | ❌ 用户显式提供 tile shape |
| 自动并行性映射 Pass | ❌ 用户通过 `parallel` 显式指定 |
| `dsa_dot`（矩阵乘法） | ❌ 在 IR 中已注释，未实现 |

---

## 4. 后端对照

### 各后端 TLE 集成度

| 后端 | 插件名 | TLE 集成 | 关键文件 |
|------|--------|----------|----------|
| **Ascend** | `TritonAscend` | ✅ 深度集成 — Code generator 感知 TLE 的 `pipeline`、`parallel`、`hint` 系统 | `third_party/ascend/backend/spec/triton/compiler/code_generator.py` |
| **NVIDIA** | `TritonNVIDIA` | ❌ 无 TLE 代码 | `third_party/nvidia/backend/compiler.py` |
| **AMD** | `TritonAMD` | ❌ 无 TLE 代码 | `third_party/amd/backend/compiler.py` |
| **Mthreads** | `TritonMthreads` | ❌ 无 TLE 代码 | `third_party/mthreads/backend/compiler.py` |
| **HCU** | `TritonHCU` | ❌ 无 TLE 代码 | `third_party/hcu/backend/compiler.py` |
| **Iluvatar** | `TritonIluvatar` | ❌ 无 TLE 代码 | `third_party/iluvatar/backend/compiler.py` |
| **Cambricon** | `TritonCambricon` | ❌ 无 TLE 代码 | `third_party/cambricon/backend/compiler.py` |
| **XPU** | `TritonXPU` | ❌ 无 TLE 代码 | — |

### Ascend 后端详情

**编译流水线:**
```
Triton IR → linalg → bishengir-compile → NPU binary
```

**Code Generator:**
- 入口: `third_party/ascend/backend/spec/triton/compiler/code_generator.py`
- 对 TLE 的感知: 识别 `tle.dsa.pipeline`、`tle.dsa.parallel`、`tle.dsa.hint` 构造并生成相应的 NPU 指令

**Ascend 自定义转换 Pass（3 个）:**

| Pass 名称 | 功能 |
|-----------|------|
| `TritonToHIVM` | Triton IR → HIVM (Hardware Instruction Virtual Machine) |
| `TritonToHFusion` | Triton IR → HFusion (融合算子) |
| `TritonToLLVM` | Triton IR → LLVM IR |

### 与 v3.6.x 后端的对比

| 方面 | ascend3.2 | v3.6.x |
|------|-----------|--------|
| 主要后端 | Ascend（唯一集成 TLE） | NVIDIA（深度集成 24 处 TLE pass 调用） |
| 其他后端支持 | 无 | HCU（条件编译）, Mthreads（初步）, Enflame（1 个 pass） |
| 降低目标 | HIVM 向量操作 / bishengir | NVGPU / NVWS / LLVM |
| 内存模型 | UB / L1 / L0A / L0B / L0C 地址空间 | smem / tmem scopes |
| Warp Specialization | ❌ 不适用 | ✅ NVWS dialect |
| TMA Copy | ❌ 不适用 | ✅ ttg.tma_copy |

---

## 5. Runtime 对照

| 组件 | 状态 | 说明 |
|------|------|------|
| JIT 编译 | ✅ | 通过 Triton 标准 `JITFunction` 支持 |
| Ascend 编译器集成 | ✅ | bishengir-compile 工具链 |
| Autotuning | ✅ | Triton 标准 autotuner |
| Kernel Cache | ✅ | Triton 标准 cache |
| TLE-Raw CUDA JIT | ❌ | 无 GPU Raw 运行时（无源码） |
| TLE-Raw MLIR JIT | ❌ | 无 MLIR Raw 运行时（无源码） |
| 分布式运行时 | ❌ | 无分布式支持 |
| Profiling (Proton) | — | 依赖 Triton 上游 Proton 基础设施 |

---

## 6. 测试覆盖

### TLE 测试

| 测试类型 | 位置 | 数量 |
|----------|------|------|
| Python TLE 测试 | `python/test/tle/` | 6 个文件 |
| MLIR lit 测试 | — | 0 个 |

**6 个 TLE Python 测试:**
```
test_bind_buffer, test_tle_with_hints, test_vec_add,
test_vec_add_2d, test_vec_add_mix, test_vec_mathOps
```

> 注：所有测试均依赖 `torch_npu`（Ascend NPU PyTorch 扩展），无法在非 Ascend 硬件上运行。

### Ascend 后端测试

| 测试类型 | 位置 | 数量 |
|----------|------|------|
| 单元测试 | `third_party/ascend/unittest/` | 365 个文件 |
| 集成测试 | `third_party/ascend/test/` | 78 个文件 |

### 与 v3.6.x 测试的对比

| 方面 | ascend3.2 | v3.6.x |
|------|-----------|--------|
| TLE Python 单元测试 | 6 个 | 10 个 |
| TLE Python 集成测试 | 0 个 | 6 个 |
| MLIR lit 测试 | 0 个 | 56 个 |
| 后端测试 | 443 个（Ascend） | 依赖 NVIDIA CI |

---

## 7. 教程 / 示例

### TLE 教程

| 文件 | 说明 |
|------|------|
| `python/tutorials/tle/01-sparse-flash-attn-tle.py` | 949 行，Sparse Flash Attention TLE 实现 |

### Ascend 后端教程

| 位置 | 数量 | 说明 |
|------|------|------|
| `third_party/ascend/tutorials/` | 15 个文件 | Ascend NPU 通用教程（不限于 TLE） |

### 与 v3.6.x 教程的对比

| 方面 | ascend3.2 | v3.6.x |
|------|-----------|--------|
| TLE 教程 | 1 个 | 4 个通用 + 3 个 DeepSeek + Raw CUDA/MLIR 教程 |
| 后端教程 | 15 个（Ascend 通用） | — |

---

## 8. 完整架构差异总结

| 方面 | ascend3.2 (0.5.0+ascend3.2) | v3.6.x (0.5.1) |
|------|----------------------------|-----------------|
| 设计焦点 | DSA buffer 操作（Ascend NPU 存储层次） | GPU shared memory + pipeline（NVIDIA） |
| API 命名空间 | `tle.dsa.*` | `tle.*`, `tle.gpu.*` |
| IR Operations | 7 个（dsa_*） | 24 个（tile, pipe, memory, distributed） |
| MLIR Passes | 0 个（直接 C++ 转换） | 21 个注册 Pass |
| 主要后端 | Ascend（唯一集成） | NVIDIA（主要）, HCU/Mthreads（部分） |
| 内存模型 | UB / L1 / L0A / L0B / L0C 地址空间 | smem / tmem scopes |
| 分布式支持 | ❌ 未实现 | ✅ device_mesh, sharding, remote |
| Raw API | ❌ 无源码 | ✅ CUDA + MLIR inline |
| 矩阵乘法 | ❌ dsa_dot 已注释 | ✅ 标准 tl.dot |
| TLE 测试 | 6 个 Python, 0 MLIR | 10 单元 + 6 集成 + 56 MLIR |
| 教程 | 1 个 TLE + 15 个 Ascend | 4 通用 + 3 DeepSeek + 11 Raw |

---

## 9. 结论

### 文档与代码一致的部分
- TLE-DSA API 完整实现了 Ascend NPU buffer/DMA 操作（alloc、copy、to_buffer、to_tensor、subview）
- 6 个逐元素数学运算（add/sub/mul/div/max/min）从 Python API 到 IR 到 HIVM 后端全链路打通
- `pipeline` 和 `parallel` 循环构造在 Code Generator 中有完整的后端支持
- `hint` 上下文管理器与 Ascend 编译器集成
- 7 个 DSA IR ops 和 2 个转换模式（DSACopyConverter + MathConverter）正确实现
- Ascend 地址空间（UB/L1/L0A/L0B/L0C）在类型系统中完整定义
- 6 个 Python 测试覆盖了基本功能（buffer 绑定、hint、向量加法、混合精度、数学运算）

### 文档超前于代码的部分
- **`dsa_dot`（矩阵乘法）** — 在 IR 定义中已注释掉，未实现
- **MLIR lit 测试** — 0 个，编译器转换无 IR 级别测试覆盖
- **分布式 API** — 完全不存在（v3.6.x 已有 device_mesh/sharding/remote）
- **Raw API** — 无源码（仅存在 .pyc 文件）
- **GPU 特定 API** — 无源码（仅存在 .pyc 文件），即 v3.6.x 的 alloc/copy/pipeline/warp_specialize 等均不可用
- **多后端 TLE 支持** — 仅 Ascend 后端集成 TLE，其他 7 个后端均无 TLE 代码
- **Level 1 算法级 API** — 完全未实现
- **自动 Tiling/并行映射** — 编译器未实现自动化，用户需显式指定

---
