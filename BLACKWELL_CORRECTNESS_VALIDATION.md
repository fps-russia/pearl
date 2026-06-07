# Pearl Blackwell Correctness Validation Strategy

## Overview

This document defines the validation strategy for ensuring bit-identical, cryptographically sound PoUW outputs across architectures (Hopper sm_90a vs Blackwell sm_121a).

## Core Invariant

**The PoUW hash transcript must be bit-identical across all architectures.**

If `xor_reduction(tCrC)` produces different bits on Hopper vs Blackwell, the ZK proof will fail validation, and mining rewards will be lost.

---

## Validation Levels

### Level 1: Unit Correctness (kernel-level)

**Goal:** Verify each kernel produces numerically correct outputs.

**Tests:**
1. `pytest miner/pearl-gemm/tests/test_pearl_gemm.py`
   - `TestSwizzleNoisyGEMM::test_int7_swizzle_noisy_gemm` — canonical 128×256×128 tile
   - All parameterized variants (tile sizes, R values, skip_denoising)
   - **Must pass:** 2874/2874 tests (PR #118 claim on GB10)

2. `pytest miner/pearl-gemm/tests/test_pearl_gemm.py -k "matmul_config1"`
   - PR #130 claims this fails on 128×256×128, R=128
   - **Must investigate and resolve** if it fails

3. `pytest miner/pearl-gemm/tests/test_noising.py`
   - Noising A/B correctness
   - Denoise converter correctness

**Validation criteria:**
- Output tensors match reference (CPU PyTorch) within int8 quantization tolerance
- Scale factors are applied correctly
- No NaN/Inf in outputs

---

### Level 2: Bit-Identical PoUW (transcript-level)

**Goal:** Verify `xor_reduction(tCrC)` produces identical hash transcripts across architectures.

**Why this matters:**
- Pearl's PoUW protocol uses `xor_reduction(tCrC)` to generate a hash transcript
- This transcript is fed into BLAKE3 to generate a proof-of-work
- If the transcript differs, the proof is invalid

**Test implementation:**
```python
def test_bit_identical_transcript():
    """Verify xor_reduction(tCrC) is bit-identical across architectures."""
    # Run on Hopper (if available) or reference CPU backend
    transcript_hopper = run_noisy_gemm_and_extract_transcript(..., arch='sm_90a')
    
    # Run on Blackwell
    transcript_blackwell = run_noisy_gemm_and_extract_transcript(..., arch='sm_121a')
    
    # Must be bit-identical
    assert transcript_hopper == transcript_blackwell
```

**Where to add this:**
- `miner/pearl-gemm/tests/test_pearl_gemm.py` — new test class `TestBitIdenticalTranscript`
- Test all compiled kernel configs
- Test all R values (64, 128)
- Test with and without denoising

**Validation criteria:**
- 100% bit-identical match across all test cases
- If any mismatch: kernel is invalid for mining

---

### Level 3: End-to-End Mining (integration-level)

**Goal:** Verify the full vLLM miner stack works on Blackwell.

**Tests:**
1. `pytest miner/vllm-miner/tests/test_vanilla_gemm.py`
   - Basic GEMM correctness under vLLM
   - Already modified in PR #130 for reference backend

2. `pytest miner/vllm-miner/tests/test_vllm_execution.py`
   - Full vLLM server with Pearl plugin
   - **Requires:** Local pearld node in simnet mode
   - **Requires:** GPU with sufficient VRAM

3. Live testnet mining
   - Run `vllm-miner` against Pearl testnet
   - Monitor accepted vs rejected shares
   - Target: >95% acceptance rate

**Validation criteria:**
- Server starts without errors
- Model loads and serves inference
- Mining loop submits blocks
- Shares are accepted by the network

---

### Level 4: Performance Regression (benchmark-level)

**Goal:** Ensure Blackwell path is not slower than the reference backend.

**Benchmarks:**
1. `pytest -m performance miner/pearl-gemm/tests/`
   - Throughput per kernel config
   - Compare against PR #130's reference backend
   - Target: Native kernel > Reference backend

2. `nv-nsight-compute` profiling
   - Achieved occupancy
   - Tensor core utilization
   - Memory bandwidth
   - SMEM usage

3. End-to-end mining benchmark
   - Shares per hour on testnet
   - Compare against Hopper (if available)
   - Compare against PR #130 reference backend

**Validation criteria:**
- Native kernel is faster than reference backend
- No performance cliff (smooth scaling with problem size)

---

## Validation Matrix

| Test | Hopper (sm_90a) | Blackwell (sm_121a) | Frequency |
|---|---|---|---|
| Unit correctness (pytest) | ✅ Required | ✅ Required | Every PR |
| Bit-identical transcript | ✅ Required | ✅ Required | Every PR |
| End-to-end vLLM | ✅ Required | ✅ Required | Every PR |
| Performance benchmark | ✅ Required | ✅ Required | Weekly |
| Testnet mining | ✅ Required | ✅ Required | Before release |

---

## Specific Test Cases for GB10

### Critical: 128×256×128, R=128, stages=2

This is the canonical mining tile. PR #130 claims it fails on this config.

**Test command:**
```bash
pytest -q 'miner/pearl-gemm/tests/test_pearl_gemm.py::TestSwizzleNoisyGEMM::test_int7_swizzle_noisy_gemm[True-None-matmul_config1-8192-8192-512]'
```

**If this passes:** PR #130's claim is incorrect or based on an older iteration.
**If this fails:** Investigate whether it's SMEM overflow, correctness bug, or test issue.

### Critical: SMEM fit validation

**Test:** Verify the canonical tile fits in 99 KB SMEM.

```python
def test_smem_fit():
    smem_usage = calculate_smem_usage(tile_m=128, tile_n=256, tile_k=128, stages=2, R=128)
    assert smem_usage <= 99 * 1024  # 99 KB cap
```

**Validation:** Run with `nv-nsight-compute` to confirm actual SMEM allocation.

### Critical: Noising A/B bit-identicality

**Test:** Verify noising kernels produce identical outputs on Hopper vs Blackwell.

```python
def test_noising_bit_identical():
    A_hopper = noise_A_hopper(A, EAL, EAR, EBL)
    A_blackwell = noise_A_blackwell(A, EAL, EAR, EBL)
    assert (A_hopper == A_blackwell).all()
```

---

## Debugging Mismatches

If bit-identicality fails:

1. **Isolate the mismatch** — Which kernel? Which tile? Which R?
2. **Check MMA atom** — Are we using the same `mma.sync` instruction? (SM80_16x8x32_S32S8S8S32_TN)
3. **Check ldmatrix staging** — Is ldmatrix producing identical registers? (Should be, it's deterministic)
4. **Check scale application** — Are scale factors applied in the same order? (FP arithmetic is non-associative)
5. **Check denoise** — Is the denoise SMEM union aliasing correctly? (NamedBarrier timing)
6. **Check epilogue** — Are we writing to gmem in the same order? (Order matters for xor_reduction)

**Debugging tool:**
```cpp
// Add to kernel for debug builds
if (EnableDebug) {
    // Print first few elements of tCrC after each k-block
    // Compare between Hopper and Blackwell builds
}
```

---

## CI Integration

### Current CI (Hopper-only)

Pearl's CI runners are Hopper-only (sm_90a). They cannot exercise Blackwell paths.

### Proposed CI additions

1. **GB10 runner** — Add a DGX Spark or GB10 cloud instance to CI
   - Run `pytest miner/pearl-gemm/tests` on every PR
   - Run `pytest miner/vllm-miner/tests` on every PR
   - Run performance benchmarks weekly

2. **Cross-architecture validation** — Compare outputs between Hopper and Blackwell
   - Generate reference outputs on Hopper
   - Verify Blackwell outputs match
   - Fail CI if mismatch

3. **Build verification** — Ensure Blackwell builds compile
   - `PEARL_GEMM_ARCH=sm_121a pip install -e miner/pearl-gemm`
   - Verify `cuobjdump` shows `sm_121a` cubins

---

## Validation Checklist for Implementation

### Before merging PR #118:

- [ ] Run full test suite on GB10: `pytest miner/pearl-gemm/tests -v`
- [ ] Verify 128×256×128, R=128 passes: `pytest -k 'matmul_config1'`
- [ ] Run vLLM miner tests: `pytest miner/vllm-miner/tests`
- [ ] Profile SMEM usage: `nv-nsight-compute --metrics smem_usage`
- [ ] Verify bit-identical transcript: Add `TestBitIdenticalTranscript` class
- [ ] Run testnet mining: Monitor accepted shares for 24 hours
- [ ] Compare performance vs PR #130 reference backend
- [ ] Document any known issues or limitations

### After merging PR #118:

- [ ] Add GB10 CI runner
- [ ] Set up nightly testnet mining validation
- [ ] Track performance regression over time
- [ ] Monitor for any ZK proof validation failures

---

## Known Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Bit-identicality failure | Medium | Critical | Extensive test suite, cross-arch validation |
| SMEM overflow on 128×256×128 | Low | High | Profile actual SMEM usage, fallback to smaller tile |
| PR #130 correctness claim is true | Medium | High | Run the specific test, investigate if it fails |
| Performance worse than reference backend | Low | Medium | Benchmark before merging, optimize if needed |
| tcgen05 never available for SM121 int8 | High | Low | SM80 path is viable, just not optimal |

---

## Summary

The validation strategy is **defense in depth**:

1. **Unit tests** catch numerical errors
2. **Bit-identical tests** catch transcript mismatches
3. **Integration tests** catch stack issues
4. **Live mining** catches real-world problems

**The most important test is bit-identicality.** If the xor_reduction transcript differs by even one bit, the entire PoUW system is broken. This must be validated exhaustively before any Blackwell kernel is used for mining.
