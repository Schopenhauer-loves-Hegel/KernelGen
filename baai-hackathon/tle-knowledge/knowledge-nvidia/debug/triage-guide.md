---
title: Debug 排查指南
scope: TLE kernel 正确性和性能问题的系统排查流程
hardware: sm80+
prerequisites: api/gpu.md, compiler/pass-pipeline.md
---

# Debug 排查指南

## 快速分诊顺序

1. 用最小 shape 复现问题
2. 与 Torch 参考实现对比确认 mismatch
3. 通过 `warmup(...).asm` dump TTGIR/PTX
4. 定位层级：API → Verifier → Lowering → Runtime

## 症状 → 层级 → 行动

### Verifier 报错（pointer/index shape）

**症状：** 编译时报 `LocalPointersOp` 相关错误

**层级：** API / Verifier

**行动：**
- 检查 `local_ptr` 的 index 数量是否等于 buffer 维度
- 检查 index 类型是否为整数
- 检查是否混用了标量和 tensor 索引
- 查看 `third_party/tle/dialect/lib/IR/Ops.cpp` 中的 verifier 逻辑

```bash
rg -n "LocalPointersOp::verify" third_party/tle/dialect/lib/IR/Ops.cpp
```

### 编译通过但输出错误

**症状：** kernel 运行完成，但结果与参考不符

**层级：** Kernel 逻辑 / Lowering mismatch

**行动：**
1. 缩小 shape 到最小复现（如 `N=256, BLOCK=256`）
2. 隔离单个 tile，逐步检查中间 load/store
3. 检查 TTGIR 中 barrier 是否正确插入

```python
compiled = kernel.warmup(*args, grid=grid)
print(compiled.asm['ttgir'])
```

### local store/load 后间歇性 mismatch

**症状：** 结果有时正确有时错误，或在不同 shape 下表现不一致

**层级：** Barrier / 同步问题

**行动：**
- 检查 `triton-tle-insert-local-pointer-barriers` 是否正确插入 barrier
- 简化 control flow（去掉条件分支）
- 检查 TTGIR 中 `ttg.local_load` 前是否有 barrier

```bash
rg -n "TleInsertLocalPointerBarriers" third_party/tle/dialect/lib/Transforms/TleInsertLocalPointerBarriers.cpp
```

### 使用 local staging 后无性能提升

**症状：** 加了 `alloc + local_ptr` 但性能没有改善

**层级：** Layout 转换 / Pipeline 问题

**行动：**
1. 检查 PTX 中是否有 `ld.shared` / `st.shared`（确认 shared memory 被使用）
2. 检查是否有多余的 layout 转换（TTGIR 中的 `ttg.convert_layout`）
3. 确认 `nv_mma_shared_layout` 设置是否正确

```python
ptx = compiled.asm['ptx']
print('ld.shared count:', ptx.count('ld.shared'))
print('st.shared count:', ptx.count('st.shared'))
```

### TMA copy 不生效

**症状：** 使用 `tle.gpu.copy` 但 PTX 中没有 TMA 指令

**层级：** 硬件条件 / Pass 未执行

**行动：**
- 确认硬件是 sm90+（`capability // 10 >= 9`）
- 检查 `triton-tle-lower-tma-copy` 是否在 pass pipeline 中执行

## 常用排查命令

### 运行目标测试

```bash
<py_exec> -m pytest python/test/tle/unit/test_tle_gpu_local_ptr.py -vv -s
<py_exec> -m pytest python/test/tle/integration/test_tle_local_store.py -vv -s
<py_exec> -m pytest python/test/tle/integration/test_tle_distributed.py -vv -s
```

### 搜索相关代码

```bash
# 查找 local_ptr 实现
rg -n "def local_ptr\(" python/triton/experimental/tle/language/gpu

# 查找 verifier
rg -n "LocalPointersOp::verify|kSharedMemoryAddressSpace" third_party/tle/dialect/lib/IR/Ops.cpp

# 查找 pass 实现
rg -n "TleSelectEncodings|TleInsertLocalPointerBarriers" third_party/tle/dialect/lib/Transforms

# 查找 backend 调用
rg -n "add_early_assign_memory_space|add_select_encodings" third_party/nvidia/backend/compiler.py

# 查找 LLVM lowering
rg -n "populateLocalPointersOpToLLVMPatterns|GPUDialect" third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM.cpp
```

### Dump 编译产物

```python
compiled = kernel.warmup(*args, grid=grid)
for fmt in ['ttir', 'ttgir', 'ptx']:
    content = compiled.asm.get(fmt, '')
    if content:
        with open(f'/tmp/kernel.{fmt}', 'w') as f:
            f.write(content)
        print(f'{fmt}: {len(content)} chars')
```

## 测试覆盖位置

| 测试类型 | 位置 | 数量 |
|----------|------|------|
| MLIR lit 测试 | `third_party/tle/test/` | 56 个（1 Analysis + 33 GPU） |
| Python 单元测试 | `python/test/tle/unit/` | 10 个文件 |
| Python 集成测试 | `python/test/tle/integration/` | 6 个文件 |

## 新功能的最小测试要求

1. 一个正向用例（功能正常工作）
2. 一个负向契约用例（错误输入触发正确报错）
3. 一个回归用例（没有你的修复就会失败）
