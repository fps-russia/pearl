# Pearl Blackwell tcgen05 / 5th-Gen Tensor Core Optimization

## CRITICAL FINDING: CUTLASS Int8 Support for SM121

After exhaustive analysis of CUTLASS v4.3.0 (current submodule) and v4.5.1 (latest), **NO native int8 tcgen05 MMA atom exists for SM120/SM121 (consumer Blackwell / GB10 / DGX Spark).**

| Architecture | CUTLASS Support | Int8 MMA Atom | Notes |
|---|---|---|---|
| **SM100** (B100/B200) | `SM100_MMA_S8_SS` | `tcgen05.s8` | Datacenter only, requires `sm_100a` |
| **SM120** (RTX 50-series) | `SM120_16x8x32_TN` | FP8/FP4 only | No int8 support |
| **SM121** (GB10/DGX Spark) | `SM120_16x8x32_TN` | FP8/FP4 only | No int8 support |

**Implication:** PR #118's SM80 `mma.sync` fallback is the **only viable native kernel path** for Pearl's int8 PoUW GEMM on GB10. A true tcgen05 optimization would require either:

1. **CUTLASS upstream contribution** — Adding int8 SM120/SM121 MMA atoms to CUTLASS
2. **Raw PTX** — Bypassing CUTLASS and emitting `tcgen05.s8` PTX directly
3. **Mixed-precision refactor** — Changing Pearl's PoUW to use FP8/FP4 instead of int8 (major protocol change)
4. **Data type conversion** — Converting int8→FP8 on-the-fly (precision loss, needs validation)

---

## Work Breakdown: tcgen05 Optimization Paths

### Path A: Direct SM80 mma.sync Optimization (RECOMMENDED — Immediate)

**Goal:** Extract maximum performance from the SM80 `mma.sync` path on Blackwell without changing the instruction set.

**What PR #118 already does:**
- Uses `SM80_16x8x32_S32S8S8S32_TN` for main GEMM
- Uses `SM80_16x8x16_F32F16F16F32_TN` for denoise
- Stages via `ldmatrix` (SMEM→register)
- Removes SMEM C buffer, streams epilogue to gmem
- Serializes denoise phases via union + NamedBarrier

**Optimization opportunities on SM121:**

1. **Register pressure reduction**
   - Blackwell has more registers per SM than Hopper
   - Audit `collective_mainloop.hpp` for spilled registers
   - Reduce `ldmatrix` staging copies if possible (tcgen05 may support SMEM descriptors, but SM80 mma.sync requires registers)

2. **Pipeline stage tuning**
   - Current heuristic clamps to 2 stages on Blackwell
   - Profile if 2 stages is optimal for 99 KB SMEM vs throughput
   - Consider 1-stage pipelining for small tiles (may reduce register pressure)

3. **Warp group sizing**
   - Current: 4 warps per MMA warpgroup
   - Blackwell may benefit from different warp group configurations
   - Test 2-warpgroup vs 4-warpgroup layouts

4. **TMA store optimization**
   - PR #118 removes TMA store entirely (direct gmem writes)
   - On Blackwell, TMA may have different performance characteristics
   - Benchmark TMA vs direct gmem for the epilogue

5. **Cluster configuration**
   - Current: `cM=1, cN=1` (single CTA)
   - Blackwell may benefit from multi-CTA clusters
   - Test `cM=2, cN=1` or `cM=1, cN=2` for larger tiles

**Files to modify:**
- `miner/pearl-gemm/csrc/gemm/kernel_traits.hpp`
- `miner/pearl-gemm/csrc/gemm/collective_mainloop.hpp`
- `miner/pearl-gemm/csrc/gemm/collective_epilogue.hpp`
- `miner/pearl-gemm/csrc/gemm/heuristics.hpp`
- `miner/pearl-gemm-build-utils/src/pearl_gemm_build_utils/kernel_configs/default_compiled_kernels.py`

**Estimated effort:** 2-3 weeks
**Performance gain:** 10-30% over PR #118 baseline
**Risk:** Low — stays within proven SM80 instruction set

---

### Path B: Raw PTX tcgen05 (HIGH RISK — Research Phase)

**Goal:** Bypass CUTLASS and emit `tcgen05.s8` PTX directly for SM121.

**Challenges:**
1. **PTX documentation** — NVIDIA does not publicly document `tcgen05.s8` for SM121
2. **Architecture compatibility** — `tcgen05` is architecture-accelerated; PTX may require exact `sm_121a` target
3. **CUTLASS integration** — Pearl's kernel heavily uses CUTLASS/CuTe for TMA, pipelines, and SMEM layouts
4. **Validation** — Without CUTLASS reference, correctness verification is harder

**Research steps:**
1. Check CUDA 13.0 PTX ISA documentation for `tcgen05` on SM121
2. Disassemble a simple FP8 tcgen05 kernel from CUTLASS to see PTX patterns
3. Attempt to adapt the SM100 `SM100_MMA_S8_SS` pattern for SM121
4. Write a minimal PoC kernel that does `tcgen05.s8` on SM121
5. Validate bit-identical outputs with SM80 reference

**Files to modify:**
- New file: `miner/pearl-gemm/csrc/gemm/mma_blackwell_raw_ptx.hpp`
- `miner/pearl-gemm/csrc/gemm/kernel_traits.hpp` (conditional include)

**Estimated effort:** 4-6 weeks (research-heavy)
**Performance gain:** 50-100% over SM80 baseline (theoretical)
**Risk:** Very high — may not be possible on SM121

---

### Path C: CUTLASS Upgrade + Int8 Atom Contribution (COMMUNITY — Long Term)

**Goal:** Upgrade CUTLASS to v4.5.1+ and add SM121 int8 MMA atoms.

**Steps:**
1. **Upgrade submodule** — `git submodule update` to v4.5.1
2. **Audit API changes** — CUTLASS 4.4+ changed TMA APIs, copy traits, and MMA atom interfaces
3. **Add SM121 int8 atoms** — Model after `SM100_MMA_S8_SS` in `mma_sm100_umma.hpp`
4. **Upstream contribution** — Submit PR to NVIDIA/cutlass
5. **Integration** — Update Pearl's kernel_traits to use new atoms

**Files to modify:**
- `.gitmodules` (submodule bump)
- `miner/pearl-gemm/csrc/gemm/kernel_traits.hpp`
- `miner/pearl-gemm/csrc/gemm/collective_mainloop.hpp`
- CUTLASS upstream: `include/cute/arch/mma_sm120.hpp` (new int8 atoms)

**Estimated effort:** 6-8 weeks (includes upstream review cycle)
**Performance gain:** 50-100% over SM80 baseline (theoretical)
**Risk:** Medium — dependent on CUTLASS team acceptance

---

### Path D: Mixed-Precision Refactor (PROTOCOL — Not Recommended)

**Goal:** Change Pearl's PoUW from int8 to FP8/FP4 to use native SM120 tcgen05.

**Why this is hard:**
- Pearl's PoUW protocol uses int8 quantization with specific scale factors
- Changing to FP8 would change the noise model, commitment hashes, and ZK proofs
- Would require consensus change across all miners and nodes

**Estimated effort:** 3+ months
**Performance gain:** 100%+ (native tcgen05)
**Risk:** Extreme — protocol-level change

---

## Recommended Path: A + B Parallel

**Immediate (Weeks 1-3):** Path A — Optimize SM80 mma.sync on GB10
- Benchmark all tile configs on DGX Spark
- Tune pipeline stages, warp groups, and cluster sizes
- Target: 10-30% speedup over PR #118

**Research (Weeks 4-8):** Path B — Raw PTX tcgen05
- Disassemble existing kernels
- Write minimal PoC
- If viable, integrate into Pearl
- If not viable, document why and stop

**Community (Ongoing):** Path C — CUTLASS upstream
- Track CUTLASS releases for SM121 int8 support
- If/when added, upgrade submodule and integrate

---

## Performance Benchmarking Plan

**Metrics to collect on DGX Spark:**
1. **Kernel throughput** — GFLOP/s for each tile config
2. **Occupancy** — Achieved vs theoretical warp occupancy
3. **SMEM pressure** — Actual SMEM usage per CTA
4. **Register spills** — NVCC spill count
5. **End-to-end mining** — Accepted shares per hour on testnet
6. **Power efficiency** — Shares per watt

**Tools:**
- `nv-nsight-compute` — kernel profiling
- `nv-nsight-systems` — end-to-end tracing
- `pytest miner/pearl-gemm/tests` — correctness
- `pytest miner/vllm-miner/tests` — integration

## Files to Read for Implementation

| File | Purpose |
|---|---|
| `miner/pearl-gemm/csrc/gemm/kernel_traits.hpp` | MMA atom selection, SMEM layouts |
| `miner/pearl-gemm/csrc/gemm/collective_mainloop.hpp` | GEMM mainloop, ldmatrix staging |
| `miner/pearl-gemm/csrc/gemm/collective_epilogue.hpp` | Epilogue, denoise, SMEM management |
| `miner/pearl-gemm/csrc/gemm/heuristics.hpp` | Pipeline stage calculation |
| `miner/pearl-gemm/csrc/gemm/pow_utils.hpp` | Hash accumulator, transcript extraction |
| `miner/pearl-gemm/csrc/gemm/pearl_gemm_kernel.h` | Kernel launch, warpgroup sync |
| `miner/pearl-gemm/csrc/gemm/pearl_noisingA_kernel.h` | Noising A kernel |
| `miner/pearl-gemm/csrc/gemm/pearl_noisingB_kernel.h` | Noising B kernel |
| `miner/pearl-gemm/csrc/gemm/pearl_gemm_api.cpp` | Python API, reference backend |
| `miner/pearl-gemm-build-utils/.../default_compiled_kernels.py` | Kernel config grid |
