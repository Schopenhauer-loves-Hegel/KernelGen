# TLE 知识库

> **版本基准：tag `0.5.1`**（commit `f347c6700`，2026-05-20）

本知识库为模型在算子优化任务中提供 TLE (Triton Language Extensions) 的结构化参考。

## 使用方式

按意图检索：
- **"我要用什么 API"** → `api/`
- **"编译器会怎么处理我的代码"** → `compiler/`
- **"有没有类似的优化模式可以参考"** → `patterns/`
- **"怎么调优性能"** → `tuning/`
- **"内核跑不对/跑不快怎么排查"** → `debug/`

## 目录结构

```
knowledge/tle/
├── api/
│   ├── core.md              load, extract_tile, insert_tile, cumsum
│   ├── gpu.md               alloc, copy, local_ptr, pipeline, memory_space, warp_specialize, pipe
│   ├── distributed.md       device_mesh, sharding, remote, shard_id, barrier
│   └── raw.md               call, call_smem, DSL region
├── compiler/
│   ├── pass-pipeline.md     21 个 pass 的触发条件、顺序、效果
│   ├── ir-operations.md     24 个 TLE IR ops 语义
│   └── upstream-modifications.md  对 Triton 上游的修改
├── patterns/
│   ├── gemm-pipeline.md     pipeline + alloc + copy 的 GEMM 模式
│   ├── shared-memory-staging.md  local_ptr staging 模式
│   ├── topk-radix.md        alloc + cumsum + extract_tile
│   └── distributed-sharding.md   multi-GPU sharding 模式
├── tuning/
│   ├── parameter-priority.md     调优顺序和 benchmark 方法
│   └── hardware-specific.md      sm80/sm89/sm90 差异
└── debug/
    └── triage-guide.md      症状→层级映射、排查流程
```

## 文件格式

每个文件包含 YAML frontmatter（title, scope, hardware, prerequisites）和 markdown 正文。

## 数据来源

- API 签名经过 7 轮交叉验证（见 `documents/tle_doc_vs_code.md`）
- 优化模式提取自 `python/tutorials/tle/` 下的实际 benchmark
- 编译器行为基于 `third_party/tle/dialect/` 和 `third_party/nvidia/backend/compiler.py`
