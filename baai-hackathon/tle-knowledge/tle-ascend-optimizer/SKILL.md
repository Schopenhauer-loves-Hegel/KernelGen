---
name: tle-ascend-optimizer
description: Use TLE-DSA to optimize Triton kernels on Ascend NPU. Analyzes algorithms for NPU data movement patterns, identifies what standard Triton cannot express on Ascend, determines whether TLE-DSA can bridge the gap, and implements the optimized kernel.
---

# TLE-DSA Ascend Optimizer

## When to Use

User wants to optimize a kernel using TLE-DSA on Ascend NPU — whether it's an existing Triton kernel with performance bottlenecks, a PyTorch operator that falls back to CANN ops, or a new algorithm to implement in Triton+TLE-DSA for NPU.

## Required Input

```text
Target: <kernel name or operator to optimize>
Location: <file path or module>
Server: <where to run, e.g. ascend-0, or local>
```

If the user doesn't provide all three, ask.

## Workflow

Execute these 4 phases in order. **Stop after Phase 3 if TLE-DSA is not suitable** — explain why and suggest alternatives.

### Phase 1: Find the Bottleneck

Read the target kernel and its surrounding code. Answer:

1. Is there excessive GM<->UB data movement?
2. Are there multiple kernel launches that could be fused?
3. Is data being reloaded from GM when it could stay in UB/L1?
4. Is the vector/cube unit pipeline being underutilized?

Output a one-paragraph bottleneck summary before moving on.

### Phase 2: Understand the NPU Algorithm

Figure out how this operation is (or should be) efficiently implemented on Ascend NPU hardware. Information sources, in order of preference:

1. **Existing Ascend CANN kernel implementations** — the most precise source. Read it if available (CANN custom ops, Ascend built-in operators, etc.)
2. **HIVM operation patterns** — Hardware Instruction Virtual Machine sequences that describe how the hardware executes operations
3. **NPU compute architecture documentation** — descriptions of the three compute units and their constraints
4. **Algorithm knowledge** — known NPU implementation patterns (elementwise needs UB staging, reductions need multi-pass accumulation, etc.)

The goal is NOT to read CANN code line by line, but to answer one question:

> **What hardware-level capabilities does the efficient NPU implementation of this algorithm require?**

Express the answer as a list of hardware primitives. The NPU has three compute units:
- **Cube unit** — matrix multiply (GEMM), operates on data in L0A/L0B, accumulates to L0C
- **Vector unit** — elementwise operations (add, sub, mul, div, max, min, etc.), operates on data in UB
- **DMA unit** — explicit data movement between all memory levels (GM, L1, UB, L0A, L0B, L0C)

Key constraint: data must be in UB/L0 before compute; all data movement is explicit DMA.

Example hardware primitive lists:
- UB buffer allocation with explicit sizing
- DMA copy from GM to UB for data staging
- Multi-pass vector operations on UB buffers
- DMA writeback from UB to GM
- Software pipeline overlapping DMA and vector compute
- Multi-core parallel dispatch across AI cores
- L1 buffer allocation for intermediate data reuse
- L0A/L0B allocation for matrix unit inputs

### Phase 3: Can TLE-DSA Bridge the Gap?

Compare the hardware primitive list from Phase 2 against TLE-DSA's capability boundary.

**TLE-DSA can provide:**
- Explicit buffer allocation in UB, L1, L0A, L0B, L0C address spaces
- DMA copy between memory regions (GM<->UB, GM->L1, UB<->L0)
- Elementwise vector operations (add, sub, mul, div, max, min)
- Buffer subview for addressing sub-regions
- Tensor slice and insert operations
- Scalar element extraction from tensors
- Software pipeline iterator for overlapping DMA and compute
- Parallel loop iterator for multi-core dispatch
- Type conversion between buffer and tensor representations
- Compiler hints (`inter_no_alias` for DMA optimization)

**TLE-DSA cannot provide:**
- Cube unit (matrix multiply) control — `dsa_dot` 未实现
- Custom HIVM instruction sequences
- Fine-grained synchronization between cube/vector/DMA pipes
- Arbitrary tensor reshaping in hardware (ND2NZ is handled by converter)

**Three possible conclusions:**

1. **TLE-DSA fits** — all required hardware primitives have TLE-DSA equivalents. Proceed to Phase 4.

2. **TLE-DSA not needed** — the algorithm doesn't require capabilities beyond standard Triton. This happens when:
   - Data fits entirely in registers (small tensors with no staging needed)
   - Single-pass computation with no explicit memory management
   - Standard Triton already achieves near-roofline performance on NPU
   -> Stop. Report why. Don't force TLE-DSA where it adds nothing.

3. **TLE-DSA not enough** — the algorithm requires capabilities beyond TLE-DSA's boundary. This happens when:
   - Algorithm fundamentally needs matrix multiply (GEMM, attention)
   - Code requires fine-grained cube/vector/DMA pipe synchronization
   - Algorithm needs custom HIVM instructions or fixpipe operations
   -> Stop. Report what's missing. Suggest CANN native ops or `triton.language.extra.cann.extension` if appropriate.

### Phase 4: Implement, Test, Benchmark

#### 4a. Read TLE-DSA Knowledge Base

Before writing code, read the relevant pattern documents from the knowledge base at `knowledge/tle-ascend/`:

- `api/dsa.md` — alloc, copy, vector ops, pipeline, parallel API details
- `patterns/vector-add.md` — canonical example of GM->UB->compute->UB->GM pattern

Also check for existing TLE-DSA kernel examples in the repo:
```bash
find . -name "*.py" | xargs grep -l "tle.dsa.alloc" 2>/dev/null
```

#### 4b. Write the TLE-DSA Kernel

Known constraints to respect:
- UB capacity is limited per core (~256KB typical) — size allocations carefully
- L1 is shared across cores — coordinate usage
- DMA alignment requirements — tensor dimensions should be aligned
- `inter_no_alias=True` hint needed for independent DMA operations
- All data must be explicitly staged: GM -> UB -> vector compute -> UB -> GM
- Use `tle.dsa.to_buffer` / `tle.dsa.to_tensor` to bridge between `tl.*` and `tle.dsa.*` APIs

#### 4c. Correctness Test

Write a test script covering:
- Multiple input shapes (small, medium, large)
- Multiple dtypes (fp32, fp16) as applicable
- Edge cases (single row, maximum size, non-aligned dimensions)
- Compare against the original implementation with `torch.testing.assert_close`
- All tests require `torch_npu` — ensure NPU device is available

All tests must pass before benchmarking.

#### 4d. Benchmark

Write a benchmark script that:
- Uses `triton.testing.do_bench` or equivalent for timing
- Tests the same shapes as correctness tests
- Reports: original time, TLE-DSA time, speedup ratio
- Uses warmup runs to trigger autotune
- Benchmark with NPU profiler for detailed analysis

#### 4e. Parameter Tuning (if initial results are unsatisfying)

If speedup is marginal or negative:
1. Identify the most impactful parameter (usually block size or pipeline stages)
2. Write a sweep script or expand autotune configs
3. Priority order: tile sizes > pipeline stages > multi-core dispatch strategy > DMA patterns

## References

1. `references/capability-map.md` — Ascend-to-TLE-DSA primitive mapping table
2. TLE-Ascend knowledge base: `knowledge/tle-ascend/`
3. Ascend tutorials: `third_party/ascend/tutorials/`
