# Ascend-to-TLE-DSA Capability Map

When analyzing an Ascend NPU implementation (CANN source code, algorithm description, or documentation), use this table to determine which hardware primitives TLE-DSA can bring into Triton.

## Covered: TLE-DSA Can Bridge These

| Ascend Primitive | What It Does | TLE-DSA Equivalent | Notes |
|---|---|---|---|
| UB buffer allocation | Allocate Unified Buffer | `tle.dsa.alloc(shape, dtype, UB)` | UB is the primary compute buffer. ~256KB per core typical. |
| L1 buffer allocation | Allocate L1 cache | `tle.dsa.alloc(shape, dtype, L1)` | Shared across cores. |
| L0A/L0B allocation | Matrix unit input buffers | `tle.dsa.alloc(shape, dtype, L0A/L0B)` | For cube unit operations. |
| L0C allocation | Accumulator buffer | `tle.dsa.alloc(shape, dtype, L0C)` | Cube unit output. |
| DMA: GM->UB | Load from global to UB | `tle.dsa.copy(gm_tensor, ub_buf, shape)` | Explicit data staging. |
| DMA: UB->GM | Store from UB to global | `tle.dsa.copy(ub_buf, gm_tensor, shape)` | Explicit writeback. |
| DMA: GM->L1 | Load to L1 (ND2NZ transform) | `tle.dsa.copy(gm_tensor, l1_buf, shape)` | Compiler inserts ND2NZ. |
| Vector Add | Elementwise addition | `tle.dsa.add(in, other, out)` | Maps to HIVM VOps. |
| Vector Sub | Elementwise subtraction | `tle.dsa.sub(in, other, out)` | Maps to HIVM VOps. |
| Vector Mul | Elementwise multiplication | `tle.dsa.mul(in, other, out)` | Maps to HIVM VOps. |
| Vector Div | Elementwise division | `tle.dsa.div(in, other, out)` | Maps to HIVM VOps. |
| Vector Max | Elementwise maximum | `tle.dsa.max(in, other, out)` | Maps to HIVM VOps. |
| Vector Min | Elementwise minimum | `tle.dsa.min(in, other, out)` | Maps to HIVM VOps. |
| Tensor slice | Extract sub-tensor | `tle.dsa.extract_slice(tensor, offset, shape)` | |
| Tensor insert | Insert into sub-tensor | `tle.dsa.insert_slice(tensor, value, offset, shape)` | |
| Buffer subview | Address sub-region | `tle.dsa.subview(buf, offsets, sizes, strides)` | |
| Software pipeline | Overlap DMA and compute | `tle.dsa.pipeline(start, end, num_stages)` | |
| Multi-core parallel | Distribute across AI cores | `tle.dsa.parallel(start, end)` | |
| Type conversion | Buffer<->Tensor | `tle.dsa.to_buffer` / `tle.dsa.to_tensor` | Bridge between `tl.*` and `tle.dsa.*`. |
| Compiler hint | Optimization hints | `tle.dsa.hint(inter_no_alias=True)` | Enables independent DMA. |

## Not Covered: TLE-DSA Cannot Bridge These

| Ascend Primitive | What It Does | Why TLE-DSA Can't | Alternative |
|---|---|---|---|
| Cube unit (matrix multiply) | Hardware matrix multiply | `dsa_dot` 未实现 | Use CANN native ops. |
| `sync_block_all` / `set` / `wait` | Fine-grained pipe sync | Not in TLE-DSA API | Use `triton.language.extra.cann.extension` directly. |
| Custom HIVM instructions | Direct hardware control | Not exposed | Write CANN custom ops. |
| `fixpipe` (DMA quantization) | DMA with quantize/dequantize | Not in TLE-DSA | Use CANN extension `fixpipe`. |

## Decision Shortcuts

**If the kernel's core operation is:**
- Elementwise on moderate data -> TLE-DSA `alloc + copy + vector ops` — good fit
- Explicit data staging GM->UB->compute->UB->GM -> TLE-DSA core pattern — good fit
- Multi-pass over the same data in UB -> TLE-DSA `alloc + copy + multiple vector ops` — good fit
- Matrix multiply (GEMM/attention) -> TLE-DSA can't help (`dsa_dot` unimplemented) — use CANN
- Fine-grained cube/vector/DMA synchronization -> use CANN extensions directly
- Only simple element access with no staging -> TLE-DSA not needed — standard Triton suffices
