# CUDA-to-TLE Capability Map

When analyzing a CUDA implementation (source code, algorithm description, or documentation), use this table to determine which hardware primitives TLE can bring into Triton.

## Covered: TLE Can Bridge These

| CUDA Primitive | What It Does | TLE Equivalent | Notes |
|---|---|---|---|
| `__shared__ T buf[N]` | Allocate shared memory buffer | `tle.gpu.alloc([N], dtype, scope=tle.gpu.smem)` | Shape must be power of 2. Use `nv_mma_shared_layout=False` for non-GEMM. |
| `&smem[idx]` pointer arithmetic | Index into shared memory | `tle.gpu.local_ptr(buf, (idx,))` | Supports both scalar and tensor indices. |
| `atomicAdd(&smem[i], val)` | CTA-internal atomic on smem | `tl.atomic_add(tle.gpu.local_ptr(buf, (i,)), val, sem="relaxed", scope="cta")` | Also supports `atomic_cas`, `atomic_max`, etc. |
| `__syncthreads()` | Block-level barrier | Automatic | Compiler pass `insert-local-pointer-barriers` handles this. |
| `cp.async.bulk.tensor` (TMA) | Async bulk copy via TMA hardware | `tle.gpu.copy(src, dst, shape)` | Hopper (sm90+) only. |
| Multi-stage software pipeline | Overlap compute with async loads | `tle.gpu.pipeline(start, end, num_stages)` | Less flexible than Gluon's manual mbarrier pipeline. |
| Swizzle layout for bank conflicts | Avoid smem bank conflicts | `nv_mma_shared_layout=True` or `tle.swizzled_shared_layout(...)` | Automatic for GEMM-style access patterns. |
| `ExclusiveScan` / CUB `BlockScan` | Exclusive prefix sum | `tle.cumsum(vals, axis)` → returns `(exclusive_sum, total)` | Standard Triton only has inclusive `tl.cumsum`. |
| Extract sub-tile from tensor | Flexible tile slicing | `tle.extract_tile(tensor, offset, shape)` | Useful for non-contiguous access patterns. |

## Not Covered: TLE Cannot Bridge These

| CUDA Primitive | What It Does | Why TLE Can't | Alternative |
|---|---|---|---|
| `warpgroup_mma()` / WGMMA | Tensor Core MMA instructions | TLE doesn't expose hardware MMA control | Use Gluon (`triton.experimental.gluon`) |
| Warp specialization | Different warps run different code | Not in TLE's abstraction | Use Gluon `gl.warp_specialize(...)` |
| `mbarrier` (hardware barrier) | Fine-grained warp-level sync | TLE auto-inserts barriers, no manual control | Use Gluon mbarrier API |
| `__shfl_sync` / warp shuffle | Warp-level register exchange | Not exposed | Standard Triton has limited warp-level ops |
| Tensor Memory (TMEM) | Blackwell on-chip tensor memory | Not in TLE | Use Gluon TMEM API |
| `asm volatile("...")` | Inline PTX | `tle.raw.dsl_region` provides partial coverage | Full PTX needs Gluon or CUDA |

## Decision Shortcuts

**If the CUDA code's core loop contains:**
- `atomicAdd` on `__shared__` → TLE `alloc + local_ptr + atomic_add` — likely a good fit
- `__shared__` buffer read multiple times → TLE `alloc + local_ptr + store/load` — likely a good fit
- `cp.async` with simple staging → TLE `copy` — worth trying
- `warpgroup_mma` or `warp_specialize` → TLE cannot help — need Gluon
- Only register-level ops on small data → TLE not needed — standard Triton suffices
