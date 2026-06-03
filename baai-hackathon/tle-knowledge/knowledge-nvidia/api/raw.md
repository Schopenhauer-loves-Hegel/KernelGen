---
title: TLE Raw API
scope: 直接硬件控制 — 内联 CUDA/MLIR 代码、绕过 Triton 抽象
hardware: 取决于内联代码的目标硬件
prerequisites: CUDA PTX 或 MLIR 知识
---

# TLE Raw API

入口：
- `triton.experimental.tle.language.raw` — kernel 内调用
- `triton.experimental.tle.raw` — host 端 dialect 定义

## tle.raw.call

在 kernel 中内联调用自定义函数（通过 DSL region 机制）。

```python
tle.raw.call(func, args)
```

- `func`: 自定义函数（CUDA 或 MLIR 实现）
- `args`: 传递给函数的参数列表
- 编译器生成 `tle.dsl_region` IR op
- `tle-dslregion-inline` pass 负责内联

## tle.raw.call_smem

与 `call` 类似，但参数通过 shared memory 传递。

```python
tle.raw.call_smem(func, args)
```

- 适用于需要大量数据传递的场景
- 编译器通过 `tle-convert-arg-to-memdesc` 将参数转换为 memory descriptor

## Host 端 Dialect 定义

`triton.experimental.tle.raw` 提供：

| 组件 | 说明 |
|------|------|
| `tle.raw.dialect` | Raw dialect 注册入口 |
| `tle.raw.Input` | 标记参数为只读输入 |
| `tle.raw.InOut` | 标记参数为读写 |

## 编译器处理流程

1. Python 前端将 `tle.raw.call` 转换为 `tle.dsl_region` op
2. `tle-convert-arg-to-memdesc` — 将参数转换为 memory descriptor（Raw 专用）
3. `tle-remove-redundant-copy` — 移除冗余拷贝（Raw 专用）
4. `tle-dslregion-inline` — 在 `make_llir()` 阶段内联 DSL region

## CUDA JIT Runtime

位置：`python/triton/experimental/tle/raw/cuda/runtime.py`

通过 clang 将 CUDA 代码编译为 LLVM IR，然后链接到 Triton 生成的模块中。

## MLIR JIT Runtime

位置：`python/triton/experimental/tle/raw/mlir/runtime.py` + `codegen.py`

直接将 MLIR 代码编译并嵌入到编译 pipeline 中。

## 使用场景

- 需要使用 Triton 不直接暴露的 PTX 指令
- 需要精确控制寄存器分配或 warp 级操作
- 集成已有的 CUDA kernel 片段
- 实验性硬件特性的快速原型

## 注意事项

- Raw API 绕过了 Triton 的大部分优化 pass，性能调优需要手动完成
- 内联代码的正确性由用户负责
- 跨后端可移植性差（通常绑定到特定硬件）
