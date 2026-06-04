---
name: tle-hackathon
description: Query TLE knowledge and optimize Triton kernels on NVIDIA GPU or Ascend NPU.
---

## When to Use

- User asks about TLE APIs, compiler passes, optimization patterns, or debugging
- User wants to optimize or implement a kernel using TLE

## Mode 1: Knowledge Query

### Routing Table

| User Intent | NVIDIA (0.5.1) | Ascend (0.5.0+ascend3.2) |
|-------------|-----------------|--------------------------|
| API usage (alloc, copy, pipeline, etc.) | `knowledge-nvidia/api/gpu.md` | `knowledge-ascend/api/dsa.md` |
| Core APIs (extract_tile, cumsum, load) | `knowledge-nvidia/api/core.md` | — |
| Distributed APIs (mesh, sharding) | `knowledge-nvidia/api/distributed.md` | — |
| Raw inline (CUDA/MLIR) | `knowledge-nvidia/api/raw.md` | — |
| Compiler passes / pipeline | `knowledge-nvidia/compiler/pass-pipeline.md` | `knowledge-ascend/compiler/dialect-and-lowering.md` |
| IR operations | `knowledge-nvidia/compiler/ir-operations.md` | (same as above) |
| Upstream Triton modifications | `knowledge-nvidia/compiler/upstream-modifications.md` | — |
| Optimization patterns | `knowledge-nvidia/patterns/` | `knowledge-ascend/patterns/` |
| Performance tuning | `knowledge-nvidia/tuning/` | — |
| Hardware-specific (sm80/89/90) | `knowledge-nvidia/tuning/hardware-specific.md` | — |
| Debugging / triage | `knowledge-nvidia/debug/triage-guide.md` | — |

Platform default: NVIDIA. Switch to Ascend if user mentions Ascend/NPU/昇腾.

Always state version basis in response.

## Mode 2: Optimization Workflow

```text
Target: <kernel name or operator>
Platform: <nvidia / ascend>
```

### Phase 1: Find the Bottleneck

1. Is there a fallback to eager ops (torch.xxx) in the middle of Triton code?
2. Are there multiple kernel launches that could be fused?
3. Is memory being read multiple times when it could be staged?

### Phase 2: Understand the Hardware Algorithm

Answer: **what hardware primitives are needed for an efficient implementation?**

### Phase 3: Can TLE Bridge the Gap?

Capability map:
- NVIDIA: `references/capability-map-nvidia.md`
- Ascend: `references/capability-map-ascend.md`

Conclusions:
1. **TLE fits** — proceed to Phase 4
2. **TLE not needed** — standard Triton suffices.
3. **TLE not enough** — suggest alternatives.

### Phase 4: Implement, Test, Benchmark

**NVIDIA constraints:**
- `tle.gpu.alloc` shape must be power of 2
- `nv_mma_shared_layout=False` for non-GEMM
- `num_stages=1` in autotune (manual smem conflicts with auto-pipeline)
- H20: 228KB smem/SM, A100: 164KB smem/SM

**Ascend constraints:**
- UB capacity ~256KB per core
- All data movement is explicit DMA (`tle.dsa.copy`)
- `inter_no_alias=True` hint for independent DMAs
