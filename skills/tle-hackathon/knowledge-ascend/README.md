# TLE-Ascend 知识库

> **版本基准：tag `0.5.0+ascend3.2`**（commit `b3091e9b1`，2026-03-19）

本知识库为模型在 Ascend NPU 算子优化任务中提供 TLE-DSA (Domain-Specific Accelerator) 的结构化参考。

## 使用方式

按意图检索：
- **"我要用什么 API"** → `api/`
- **"编译器会怎么处理我的代码"** → `compiler/`
- **"怎么写 Ascend kernel"** → `patterns/`

## 目录结构

```
knowledge/tle-ascend/
├── api/
│   └── dsa.md              alloc, copy, to_buffer, to_tensor, subview, elementwise ops, slice ops, hint, pipeline, parallel
├── compiler/
│   └── dialect-and-lowering.md  TLE DSA dialect IR ops, conversion patterns, pybind11 bindings, Ascend compilation pipeline
└── patterns/
    └── vector-add.md       UB buffer allocation + DMA copy + vector add 基础模式
```

## 文件格式

每个文件包含 YAML frontmatter（title, scope, hardware, prerequisites）和 markdown 正文。

## 数据来源

- API 签名基于 `python/triton/experimental/tle/language/dsa/` 源码
- 编译器行为基于 `third_party/tle/dsa/dialect/` 和 `third_party/ascend/backend/compiler.py`
- 详见 `documents/tle_doc_vs_code_ascend.md`
