# TLE 文档与源码对照分析

本文档对比 `tle_cn.md` / `tle.md` 中的设计描述与仓库实际源码的对应关系。

---

## 1. 总体架构对照

### 文档描述的三层设计

| 文档概念 | 代码实现 | 对应路径 |
|----------|----------|----------|
| Level 1: Numpy/PyTorch 算法级 | **未实现** — 不存在高层算法级 API | — |
| Level 2: Tile 级编程 + 分布式 | TLE-Lite + TLE-Struct | `python/triton/experimental/tle/language/` |
| Level 3: 硬件特定扩展 | TLE-Raw | `python/triton/experimental/tle/raw/` |

### 实际代码的三层命名（文档概念，非代码标识符）

| 概念层 | 定位 | 入口 |
|--------|------|------|
| **TLE-Lite** | 轻量级、后端可移植扩展（async load, tile extract/insert, cumsum, distributed） | `tle/language/core.py`, `tle/language/distributed.py` |
| **TLE-Struct** | 架构感知抽象（GPU shared memory, pipeline, layout） | `tle/language/gpu/` |
| **TLE-Raw** | 直接硬件控制，内联 CUDA/MLIR 代码 | `tle/language/raw/`, `tle/raw/` |

> 注：TLE-Lite/TLE-Struct/TLE-Raw 仅出现在设计文档中，代码中无对应模块名或标识符。代码结构通过目录划分（`core.py` / `gpu/` / `raw/`）体现这三层。

---

## 2. 前端 API 对照（文档 §3.3）

### 已实现的 API

| 文档描述 | 实际 API | 源码位置 |
|----------|----------|----------|
| Tile 语义（extract/insert） | `tle.extract_tile(x, index, tile_shape)` / `tle.insert_tile(x, tile, index)` | `python/triton/experimental/tle/language/core.py` |
| 异步加载 hint | `tle.load(..., is_async=True)` | `python/triton/experimental/tle/language/core.py` |
| Scan 操作 | `tle.cumsum(input, axis=0, reverse=False, dtype=None)` — 始终返回 (exclusive_sum, total_sum) | `python/triton/experimental/tle/language/core.py` |
| 存储层级 hint | `tle.gpu.memory_space(input, space)` | `python/triton/experimental/tle/language/gpu/core.py` |
| 显式分配 | `tle.gpu.alloc(shape, dtype, layout=None, scope=tle.smem, init_value=None, nv_mma_shared_layout=True)` | `python/triton/experimental/tle/language/gpu/core.py` |
| TMA 拷贝 | `tle.gpu.copy(src, dst, shape, offsets=None)` — `shape` 为必选参数 | `python/triton/experimental/tle/language/gpu/core.py` |
| 本地指针 | `tle.gpu.local_ptr(buffer, indices=None)` | `python/triton/experimental/tle/language/gpu/core.py` |
| Pipeline 循环 | `tle.gpu.pipeline` (继承 `tl.range`) | `python/triton/experimental/tle/language/gpu/core.py` |
| Warp Specialization | `tle.gpu.warp_specialize(functions_and_args, worker_num_warps, worker_num_regs)` | `python/triton/experimental/tle/language/gpu/core.py` |
| Pipe 构造 | `tle.pipe(...)` — pipeline 数据流管道 | `python/triton/experimental/tle/language/pipe.py` |
| Pipe 类型 | `tle.pipe_reader`, `tle.pipe_writer`, `tle.pipe_slot`, `tle.pipe_value`, `tle.pipe_wait_result` | `python/triton/experimental/tle/language/gpu/types.py` |
| GPU 类型/常量 | `tle.gpu.buffered_tensor`, `tle.gpu.buffered_tensor_type`, `tle.gpu.smem`, `tle.gpu.tmem`, `tle.gpu.scope` | `python/triton/experimental/tle/language/gpu/types.py` |
| GPU Layout | `tle.gpu.layout`, `tle.gpu.shared_layout`, `tle.gpu.swizzled_shared_layout`, `tle.gpu.nv_mma_shared_layout`, `tle.gpu.tensor_memory_layout` | `python/triton/experimental/tle/language/gpu/types.py` |
| GPU 兼容别名 | `tle.gpu.storage_kind` — `memory_space` 的向后兼容别名 | `python/triton/experimental/tle/language/gpu/__init__.py` |
| 分布式 mesh | `tle.device_mesh(topology=None)` | `python/triton/experimental/tle/language/distributed.py` |
| Sharding | `tle.sharding(mesh, split=None, partial=None)` / `tle.ShardingSpec` / `S`, `P`, `B` | `python/triton/experimental/tle/language/distributed.py` |
| 远程访问 | `tle.remote(tensor, shard_id, scope=None)` / `tle.shard_id(mesh, axis)` | `python/triton/experimental/tle/language/distributed.py` |
| 分布式 barrier | `tle.distributed_barrier(mesh=None)` | `python/triton/experimental/tle/language/distributed.py` |
| Sharded Tensor 构造 | `tle.make_sharded_tensor(handle, sharding, shape=None)` → `ShardedTensor` | `python/triton/experimental/tle/language/distributed.py` |
| Raw 内联 | `tle.raw.call(func, args)` / `tle.raw.call_smem(func, args)` | `python/triton/experimental/tle/language/raw/core.py` |

### 文档提及但未实现的 API

| 文档描述 | 状态 | 说明 |
|----------|------|------|
| `tle.Tile` 类型 | ❌ 不存在 | 通过 `extract_tile`/`insert_tile` 操作实现 tile 语义，无独立类型 |
| `tle.range` (顶层) | ❌ 不存在 | 实际为 `tle.gpu.pipeline`（`tl.range` 子类） |
| `tle.tensor` | ❌ 不存在 | 使用 `buffered_tensor` / `buffered_tensor_type` 替代 |
| `tle.store` | ❌ 不存在 | 使用标准 `tl.store` + `tle.gpu.local_ptr` 获取的指针 |
| `tle.dot` | ❌ 不存在 | 使用标准 `tl.dot` |
| `tle.reduce` | ❌ 不存在 | 使用标准 `tl.reduce` |
| `tle.scan` | ❌ 不存在 | `tle.cumsum` 是唯一的 scan 类操作 |
| `tle.atomic` | ❌ 不存在 | 使用标准 `tl.atomic_*` |
| `tle.distributed_dot` | 🚧 Stub | 抛出 `NotImplementedError("distributed_dot is deferred to M5")` |
| `tle.reshard` | 🚧 Stub | 抛出 `NotImplementedError("reshard is deferred to M4")` |
| `tle.pipeline_group` | ❌ 不存在 | 文档描述的上下文管理器未实现 |
| `pipe_stages`/`pipe_orders`/`executors` 参数 | ❌ 不存在 | pipeline 迭代器无这些参数 |

---

## 3. 中端编译器对照（文档 §3.1-3.2）

### TLE MLIR Dialect — 已实现

| 组件 | 源码位置 |
|------|----------|
| Dialect 定义 | `third_party/tle/dialect/include/IR/TleDialect.td` |
| Operations 定义 | `third_party/tle/dialect/include/IR/TleOps.td` |
| C++ 实现 | `third_party/tle/dialect/lib/IR/Dialect.cpp`, `Ops.cpp` |
| Pybind11 绑定 | `third_party/tle/triton_tle.cc` |

### 定义的 IR Operations

```
tle.extract_tile, tle.insert_tile, tle.local_pointers,
tle.memdesc_wgmma_view, tle.exclusive_cumsum,
tle.wgmma_shared_operand_fence, tle.distributed_barrier,
tle.remote_pointers, tle.dsl_region, tle.yield,
tle.extract_allocated_ptr, tle.extract_aligned_ptr,
tle.extract_offset, tle.extract_sizes, tle.extract_strides,
tle.extract_ptr, tle.pack,
tle.pipe.create, tle.pipe.writer_acquire, tle.pipe.writer_commit,
tle.pipe.writer_close, tle.pipe.reader_wait, tle.pipe.reader_release,
tle.tma_store_commit_group
```

### 优化 Pass（21 个）

| Pass 名称 | 实现文件 | 功能 |
|-----------|----------|------|
| `triton-tle-early-assign-memory-space` | `TleEarlyAssignMemorySpace.cpp` | 为 tensor pointer 分配存储空间 |
| `triton-tle-select-encodings` | `TleSelectEncodings.cpp` | 为 local pointer 选择 shared encoding |
| `triton-tle-insert-local-pointer-barriers` | `TleInsertLocalPointerBarriers.cpp` | 在 store/load 间插入 CTA barrier |
| `triton-tle-optimize-local-pointer-loads` | `TleOptimizeLocalPointerLoads.cpp` | 重写为 `ttg.local_load` |
| `triton-tle-optimize-local-pointer-stores` | `TleOptimizeLocalPointerStores.cpp` | 重写为 `ttg.local_store` |
| `triton-tle-optimize-local-pointer-async-stores` | `TleOptimizeLocalPointerAsyncStores.cpp` | 规范化 async copy 模式 |
| `triton-tle-promote-local-store-staging` | `TlePromoteLocalStoreStaging.cpp` | 提升 staging 为可 pipeline 的 alloc |
| `triton-tle-schedule-tma-store-sync` | `TleScheduleTmaStoreSync.cpp` | 调度 TMA store 同步 |
| `triton-tle-lower-pipe-to-nvws` | `TleLowerPipeToNvws.cpp` | 降低 pipeline 到 NVWS warp specialization |
| `triton-tle-tile-style-pipeline-schedule` | `TleTileStylePipelineSchedule.cpp` | 重写为 TileLang 风格 preload-2 pipeline |
| `triton-tle-materialize-tile-style-pipeline` | `TleMaterializeTileStylePipeline.cpp` | 物化显式 pipeline |
| `triton-tle-downgrade-invalid-async-copy` | `TleDowngradeInvalidAsyncCopy.cpp` | 降级无效 async copy |
| `triton-tle-lower-tma-copy` | `TleLowerTmaCopy.cpp` | 降低 TMA copy 操作 |
| `triton-tle-lower-async-load` | `TleLowerAsyncLoad.cpp` | 降低 async load |
| `triton-tle-lower-exclusive-cumsum` | `TleLowerExclusiveCumsum.cpp` | 降低 exclusive cumsum |
| `triton-tle-lower-extract-tile` | `TleLowerExtractTile.cpp` | 降低 extract_tile |
| `triton-tle-lower-insert-tile` | `TleLowerInsertTile.cpp` | 降低 insert_tile |
| `tle-convert-arg-to-memdesc` | `ConvertArgToMemDesc.cpp` | 转换参数为 memory descriptor (Raw) |
| `tle-remove-redundant-copy` | `RemoveRedundantCopy.cpp` | 移除冗余拷贝 (Raw) |
| `tle-dslregion-inline` | `DSLRegionInline.cpp` | 内联 DSL region (Raw) |
| `triton-tle-optimize-exclusive-cumsum-layouts` | `TleOptimizeExclusiveCumsumLayouts.cpp` | 优化 cumsum 布局 |

所有 Pass 实现位于 `third_party/tle/dialect/lib/Transforms/`。

### 对 Triton 上游 Dialect 的修改

| 修改 | 位置 |
|------|------|
| `tt.load` 增加 `flagtree_hints` 属性 | `include/triton/Dialect/Triton/IR/TritonOps.td:256` |
| `tt.atomic_rmw` 支持 shared memory | `include/triton/Dialect/Triton/IR/TritonOps.td:385-388` |
| 增加 `ttg.tma_copy` 操作 | `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:601` |
| 增加 `SharedMemory` resource | `include/triton/Dialect/Triton/IR/TritonOps.td:25` |

### 文档提及但未实现的编译器特性

| 特性 | 状态 |
|------|------|
| 降低到 Linalg/Tensor/SCF Dialect | ❌ TLE ops 直接通过 Triton pipeline 降低 |
| 自动 Tiling Pass | ❌ 用户显式提供 tile shape |
| 自动并行性映射 Pass | ❌ 依赖现有 Triton 基础设施 |
| Buffer Promotion（自动） | 🔶 仅限 shared memory staging 模式 |

---

## 4. 后端对照（文档 §3.1 架构图）

### 各后端 TLE 集成度

| 后端 | 插件名 | TLE 集成 | 关键文件 |
|------|--------|----------|----------|
| **NVIDIA** | `TritonNVIDIA` | ✅ 深度集成（24 处 TLE pass 调用，19 个唯一 pass） | `third_party/nvidia/backend/compiler.py` |
| **AMD** | `TritonAMD` | ❌ 无 TLE 代码 | `third_party/amd/backend/compiler.py` |
| **HCU** | `TritonHCU` | 🔶 携带 TLE dialect 定义（`#ifdef __TLE__` 条件编译） | `third_party/hcu/backend/compiler.py` |
| **Enflame** | `TritonEnflame` | 🔶 有 1 个 TLE lowering pass | `third_party/enflame/triton_gcu/triton_gcu400/lib/Conversion/TritonToGCU/TleToTritonGCU.cpp` |
| **Mthreads** | `TritonMthreads` | 🔶 初步集成（1 个 TLE pass 调用 + 测试） | `third_party/mthreads/backend/compiler.py` |

### NVIDIA 后端详情

**NVGPU Dialect:**
- 定义: `third_party/nvidia/include/Dialect/NVGPU/IR/`
- 实现: `third_party/nvidia/lib/Dialect/NVGPU/`
- 降低: `third_party/nvidia/lib/NVGPUToLLVM/`

**NVWS (Warp Specialization) Dialect:**
- 定义: `third_party/nvidia/include/Dialect/NVWS/IR/NVWSDialect.td`
- Operations: `NVWSOps.td` — `warp_group`, `aref.*` 等
- Transforms: `NVWSLowerWarpGroup`, `NVWSAssignStagePhase`, `NVWSLowerAref`, `NVWSInsertAref`, `NVWSInsertTmemAref`
- 实现: `third_party/nvidia/lib/Dialect/NVWS/Transforms/`

**Hopper Warp Specialization:**
- 入口: `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization.cpp`
- 子模块: `WSSpecialize`, `WSTaskPartition`, `WSCodePartition`, `WSDataPartition`, `WSBuffer`, `WSLowerMem`, `WSLowerToken`

**TLE 在 NVIDIA pipeline 中的集成:**
- `compiler.py` 导入 `from triton._C.libtriton import tle`
- `make_ttgir()` 中调用 24 处 TLE pass（19 个唯一 pass，部分在互斥条件分支中多次调用）
- `make_llir()` 中调用 `tle.raw_passes.add_tle_dsl_region_inline(pm)`
- `TritonGPUToLLVM` 中通过 `#ifdef __TLE__` 条件编译 TLE lowering patterns（如 `TritonNVIDIAGPUToLLVM/DotOpToLLVM.cpp` 等多个文件）

### TLE 自身的 third_party

- `third_party/tle/third_party/flagcx/` — FlagCX 通信库，支持 `tle.RemotePointersOp` 和 `tle.DistributedBarrierOp`

---

## 5. Runtime 对照（文档 §3.1）

| 组件 | 状态 | 源码位置 |
|------|------|----------|
| JIT 编译 | ✅ | `python/triton/runtime/jit.py` — `JITFunction` 类 |
| 异步编译 | ✅ | `python/triton/runtime/_async_compile.py` |
| Autotuning | ✅ | `python/triton/runtime/autotuner.py` + FlagTree 特有的 `auto_adjust_block_sizes` |
| Kernel Cache | ✅ | `python/triton/runtime/cache.py` |
| Profiling (Proton) | ✅ | `third_party/proton/` — 完整的 profiling 子系统 |
| Hint Manager | ✅ | `python/triton/compiler/hint_manager.py` |
| TLE-Raw CUDA JIT | ✅ | `python/triton/experimental/tle/raw/cuda/runtime.py` — clang → LLVM IR |
| TLE-Raw MLIR JIT | ✅ | `python/triton/experimental/tle/raw/mlir/runtime.py` + `codegen.py` |

---

## 6. Benchmark 对照（文档 §4）

| 文档引用的文件 | 状态 | 实际路径 |
|---------------|------|----------|
| `python/tutorials/tle/deepseek_v32/02-sparse-mla.py` | ✅ 存在 | 132KB，含 Triton + TLE 变体 + TileLang 基线 |
| `python/tutorials/tle/deepseek_v32/01-topk_selector.py` | ✅ 存在 | 166KB，TRT 风格 4-step radix selector |
| `python/tutorials/tle/deepseek_v32/00-topk_radix.py` | ❌ 不存在 | 相关实现在 `python/tutorials/tle/03-topk.py` |

### 其他 TLE 教程/示例

- `python/tutorials/tle/01-fft.py` — FFT 教程
- `python/tutorials/tle/02-moe_align_block_size.py` — MoE align block size 教程
- `python/tutorials/tle/03-topk.py` — Radix-Only TopK 教程，使用 `tle.gpu.alloc`
- `python/tutorials/tle/04-cluster-gemm.py` — Cluster GEMM 教程
- `python/tutorials/tle/raw/cuda/` — Raw CUDA 教程（4 个 .py + 3 个 .cu：vector-add, fused-softmax, matrix-multiplication ×2）
- `python/tutorials/tle/raw/mlir/` — Raw MLIR 教程（7 个 .py：vector-add, fused-softmax, matrix-multiplication ×2, hello-world, topk, test-vassert）
- `python/triton_kernels/triton_kernels/topk.py` — 高层 `topk_forward()` 函数
- `python/triton_kernels/triton_kernels/topk_details/` — streaming topk 内核实现

---

## 7. 测试覆盖

| 测试类型 | 位置 | 数量 |
|----------|------|------|
| MLIR lit 测试 | `third_party/tle/test/` | 56 个 |
| Python 单元测试 | `python/test/tle/unit/` | 10 个文件：extract_tile(2), insert_tile(2), cumsum, local_ptr, distributed, cluster_tutorial_skip, tle, gpu_slot |
| Python 集成测试 | `python/test/tle/integration/` | 6 个文件：GEMM, TMA copy, pipeline E2E, distributed, local store, topk_smem_fallback |

---

## 8. f2reduce 组件

| 方面 | 说明 |
|------|------|
| 位置 | `third_party/f2reduce/` |
| 性质 | 底层编译器工具库，非 TLE 用户 API |
| 功能 | GF(2) 上的矩阵运算，支持 `LinearLayout` 类的 basis 变换 |
| 与 TLE 关系 | 当 `tle.gpu.alloc` 创建 swizzled layout 时，编译器使用 `LinearLayout`（基于 f2reduce）计算线程到内存的映射 |
| 使用位置 | `lib/Tools/LinearLayout.cpp`（主实现）, `third_party/hcu/triton/Tools/LinearLayout.cpp`, `third_party/mthreads/lib/Tools/LinearLayout.cpp` |

---

## 9. 结论

### 文档与代码一致的部分
- 多后端架构（5 个后端均存在）
- TLE MLIR Dialect 及 21 个优化 Pass（含 `TleScheduleTmaStoreSync` 和 `TleLowerPipeToNvws`）
- NVIDIA 深度集成（NVGPU + NVWS + Hopper WS）
- Runtime 基础设施（JIT、Autotuning、Proton）
- 核心 API：tile extract/insert、async load、cumsum、warp_specialize、distributed、GPU memory management、Raw inline
- Benchmark 文件（2/3 存在）

### 文档超前于代码的部分
- **Level 1 算法级 API** — 完全未实现
- **独立的 `tle.Tile` 类型** — 用操作替代
- **`tle.store/dot/reduce/scan/atomic`** — 不存在，使用标准 `tl.*`
- **`tle.pipeline_group` 上下文管理器** — 未实现
- **`tle.distributed_dot`** — 仅 stub，推迟到 M5
- **自动 Tiling/并行映射** — 编译器未实现自动化
- **Linalg/Tensor Dialect 降低路径** — 不存在
- **AMD 的 TLE 集成** — 尚未开始（HCU 携带 TLE dialect 定义，Mthreads 已有初步集成）
- **`00-topk_radix.py` benchmark 文件** — 不存在（替代文件为 `python/tutorials/tle/03-topk.py`）

---
