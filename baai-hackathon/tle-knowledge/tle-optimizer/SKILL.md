---
name: tle-optimizer
description: Use TLE to optimize Triton kernels. Analyzes how algorithms are implemented on GPU (primarily from CUDA implementations/algorithm knowledge), identifies what standard Triton cannot express, determines whether TLE can bridge the gap, and if so, implements the optimized kernel with correctness tests and benchmarks.
---

# TLE Optimizer

## When to Use

User wants to optimize a kernel using TLE — whether it's an existing Triton kernel with performance bottlenecks, a PyTorch operator that falls back to CUDA ops, or a new algorithm to implement in Triton+TLE.

## Required Input

```text
Target: <kernel name or operator to optimize>
Location: <file path or module>
Server: <where to run, e.g. h20-0, or local>
```

If the user doesn't provide all three, ask.

## Workflow

Execute these 4 phases in order. **Stop after Phase 3 if TLE is not suitable** — explain why and suggest alternatives.

### Phase 1: Find the Bottleneck

Read the target kernel and its surrounding code. Answer:

1. Where is performance being left on the table?
2. Is there a fallback to `torch.xxx` (topk, sort, argsort, cumsum, etc.) in the middle of Triton code?
3. Are there multiple kernel launches that could be fused?
4. Is global memory being read multiple times when it could be staged?

Output a one-paragraph bottleneck summary before moving on.

### Phase 2: Understand the GPU Algorithm

Figure out how this operation is (or should be) efficiently implemented on GPU hardware. Information sources, in order of preference:

1. **CUDA source code** — the most precise source. Read it if available (CUB, CUTLASS, PyTorch CUDA kernels, vLLM kernels, etc.)
2. **Code annotations** — comments like "backed by CUB radix sort" carry algorithmic information
3. **Documentation & papers** — algorithm descriptions with hardware primitive requirements
4. **Algorithm knowledge** — known GPU implementation patterns (radix sort needs smem histogram, reduction needs smem staging, etc.)

The goal is NOT to read CUDA line by line, but to answer one question:

> **What hardware-level capabilities does the efficient GPU implementation of this algorithm require?**

Express the answer as a list of hardware primitives. Examples:
- Shared memory allocation with explicit lifetime
- CTA-internal atomic operations on shared memory
- Multi-pass iteration over the same shared memory buffer
- Explicit data staging from global to shared memory
- Software pipelining with async copy
- Warp-level shuffle / reduction
- Hardware MMA (tensor core) instructions
- Warp specialization

### Phase 3: Can TLE Bridge the Gap?

Compare the hardware primitive list from Phase 2 against TLE's capability boundary.

**TLE can provide:**
- Explicit shared memory allocation and lifetime control
- CTA-internal atomics on shared memory
- Shared memory pointer arithmetic (indexing into smem buffers)
- Automatic barrier insertion between smem store/load
- TMA copy (async bulk tensor copy)
- Software pipeline
- Exclusive prefix sum
- Shared memory layout control (swizzle for bank conflict avoidance)

**TLE cannot provide:**
- Warp specialization
- Hardware MMA instruction control (WGMMA/tcgen05)
- Manual mbarrier control
- Warp-level shuffle primitives
- Tensor memory (TMEM) operations

**Three possible conclusions:**

1. **TLE fits** — all required hardware primitives have TLE equivalents. Proceed to Phase 4.

2. **TLE not needed** — the algorithm doesn't require capabilities beyond standard Triton. This happens when:
   - Data fits entirely in registers (small N, e.g. N=256 for MoE expert selection)
   - Single-pass computation with no data reuse
   - Standard Triton already achieves near-roofline performance
   → Stop. Report why. Don't force TLE where it adds nothing.

3. **TLE not enough** — the algorithm requires capabilities beyond TLE's boundary. This happens when:
   - Code already uses Gluon-level APIs (TMA descriptors, WGMMA, warp specialize)
   - Algorithm fundamentally needs warp-level control
   → Stop. Report what's missing. Suggest Gluon or CUDA if appropriate.

### Phase 4: Implement, Test, Benchmark

#### 4a. Read TLE Knowledge Base

Before writing code, read the relevant pattern documents from the knowledge base at `knowledge/tle/`:

- `api/gpu.md` — alloc, local_ptr, copy, pipeline API details
- `api/core.md` — cumsum, extract_tile, load semantics
- `patterns/topk-radix.md` — if doing multi-pass histogram/radix algorithms
- `patterns/shared-memory-staging.md` — if doing data staging/reuse
- `tuning/parameter-priority.md` — for autotune config design

Also check for existing TLE kernel examples in the repo:
```bash
find . -name "*.py" | xargs grep -l "tle.gpu.alloc" 2>/dev/null
```

#### 4b. Write the TLE Kernel

Known constraints to respect:
- `tle.gpu.alloc` shape must be a power of 2 — use `triton.next_power_of_2(actual_size)`
- `nv_mma_shared_layout=False` for non-GEMM scenarios (histogram buffers, staging buffers)
- `num_stages=1` in autotune configs — manual smem management conflicts with automatic pipelining
- `scope=tle.gpu.smem` (not `tle.smem`)
- Shared memory capacity check: H20 has 228KB per SM, A100 has 164KB per SM

#### 4c. Correctness Test

Write a test script covering:
- Multiple input shapes (small, medium, large)
- Multiple dtypes (fp32, bf16, fp16) as applicable
- Edge cases (single row, maximum size, non-aligned dimensions)
- Compare against the original implementation with `torch.testing.assert_close`

All tests must pass before benchmarking.

#### 4d. Benchmark

Write a benchmark script that:
- Uses `triton.testing.do_bench` or equivalent for timing
- Tests the same shapes as correctness tests
- Reports: original time, TLE time, speedup ratio
- Uses warmup runs to trigger autotune

#### 4e. Parameter Tuning (if initial results are unsatisfying)

If speedup is marginal or negative:
1. Identify the most impactful parameter (usually BLOCK size or algorithm-specific like RADIX_BITS)
2. Write a sweep script or expand autotune configs
3. Priority order: tile sizes > algorithm params > num_warps > memory path

## References

1. `references/capability-map.md` — CUDA-to-TLE primitive mapping table
2. TLE knowledge base: `knowledge/tle/`
3. Existing skill for TLE internals: `skills/tle-developer/`
