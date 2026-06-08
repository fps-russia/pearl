HANDOFF CONTEXT
===============

USER REQUESTS (AS-IS)
---------------------
- "are you able to look at pull request #118 and #130 for this repo? Also look at the conversation/comments. I have an NVIDIA DGX Spark and want to fully implement what they are attempting to do, and fully leverage the blackwell cores (sm_121a architecture). Help me plan this out."
- "Also earlier: MCP server discovery and enablement for this project"
- "use uv to set up a virtual environment please"
- "Continue if you have next steps, or stop and ask for clarification if you are unsure how to proceed."
- "are you able to fork the pearl repo and commit these changes to it? then I can clone it down on my dgx spark and we can continue"
- "write a handoff file that i can provide to my agent and include it in this repo"

GOAL
----
Implement fully functional and optimized Blackwell (sm_121a) support for Pearl mining on DGX Spark/GB10, starting with PR #118 as baseline and optimizing the SM80 mma.sync path since native tcgen05 int8 is not available in CUTLASS.

WORK COMPLETED
--------------
- Analyzed PR #118 (native kernel, SM80 mma.sync fallback) and PR #130 (reference backend, PyTorch torch._int_mm) in detail using GitHub MCP tools
- Retrieved full diffs, comments, and review threads for both PRs
- Discovered critical finding: CUTLASS v4.3.0 (current submodule) and even v4.5.1 (latest) have NO int8 MMA atoms for SM120/SM121 - only FP8/FP4 variants exist
- This means PR #118's SM80 mma.sync fallback is the ONLY viable native kernel path for Pearl's int8 PoUW GEMM on GB10
- Created comprehensive technical roadmap document: BLACKWELL_TCGEN05_PLAN.md
- Created correctness validation strategy document: BLACKWELL_CORRECTNESS_VALIDATION.md
- Enabled MCP servers: github-official, context7, playwright, docker-docs, mcp-code-interpreter, rust-mcp-filesystem, desktop-commander
- Initialized CUTLASS submodule (was empty): checked out to 291300ff (v4.3.0-89-g291300ff)
- Rebased PR #118 onto current master (f8804af6)
- Added compile-time architecture macro `PEARL_GEMM_ARCH` to setup.py (e.g., sm_90a -> 90, sm_121a -> 121)
- Added architecture guards to kernel_traits.hpp: MMA atom selection (SM80 vs GMMA), SMEM layout conditional
- Added architecture guards to collective_epilogue.hpp: direct gmem store vs SMEM C + TMA
- Fixed AxEB_size bug in heuristics.hpp (operator precedence: original was wrong, now corrected)
- Fixed setup.py platform tag for aarch64 (DGX Spark): linux_{platform.machine()} instead of hardcoded linux_x86_64
- Rewrote get_pipeline_stages() in heuristics.hpp to correctly model the UNION SMEM layout introduced in PR #118. Previous formula double-counted denoise buffers, producing a pessimistic universal 2-stage cap. New formula computes union_size(stages) = max(AB*stages, denoise_phase) + scales + overhead, and finds the largest compiled stage count that fits in sharedMemPerBlockOptin (naturally adapts to Blackwell 99 KB vs Hopper 227 KB).
- Added TestBitIdenticalTranscript to test_pearl_gemm.py with three tests: determinism on canonical tile, comparison against saved reference tensors for cross-arch validation, and a reference-generation helper.
- Pushed all changes to fork: https://github.com/fps-russia/pearl branch `blackwell-121a`

CURRENT STATE
-------------
- Repo: fps-russia/pearl at /home/sparky/pearl
- Branch: blackwell-121a (based on master f8804af6)
- Remote: origin (pearl-research-labs), fps (fps-russia)
- HEAD: 194ac405 (heuristic fix + bit-identical transcript test)
- Clean working tree (all changes committed)
- CUTLASS submodule initialized at miner/pearl-gemm/third_party/cutlass
- MCP servers enabled and configured
- PR #118 is unmerged (open, blocked), PR #130 is unmerged (open, blocked)

PENDING TASKS
-------------
IMMEDIATE (DGX Spark):
- Clone and checkout: git clone https://github.com/fps-russia/pearl.git && git checkout blackwell-121a
- Build with: PEARL_GEMM_ARCH=sm_121a pip install -e .
- Run full test suite: pytest miner/pearl-gemm/tests -v
- Run bit-identical determinism test: pytest miner/pearl-gemm/tests/test_pearl_gemm.py -k "test_canonical_tile_determinism" -v
- Verify the controversial 128x256x128, R=128 test case: pytest -k 'matmul_config1'
- If 128x256x128 fails, investigate root cause (PR #130 claim vs PR #118 results)

OPTIMIZATION (if tests pass):
- Profile actual SMEM usage with nv-nsight-compute for all Blackwell tile configs
- Benchmark stages=2 vs stages=3 for 128x128x64 tile (heuristic now correctly selects 3 on Blackwell)
- Implement register pressure reduction (audit collective_mainloop.hpp for spills, tune warpgroup_reg_alloc)
- Benchmark TMA store vs direct gmem for epilogue (currently using Path 3 direct gmem)
- Test cluster configurations (cM=2, cN=1 or cM=1, cN=2) on Blackwell

CROSS-ARCH VALIDATION:
- Generate reference tensors on a trusted Hopper machine:
  pytest miner/pearl-gemm/tests/test_pearl_gemm.py -k "test_canonical_tile_save_reference" -v
- Copy ref_transcript_hopper.pt to Blackwell DGX Spark
- Run cross-arch comparison:
  pytest miner/pearl-gemm/tests/test_pearl_gemm.py -k "test_canonical_tile_vs_saved_reference" -v

RESEARCH:
- Research raw PTX tcgen05 possibility (very high risk, likely impossible for SM121 int8)
- Track CUTLASS upstream for SM121 int8 support
- Add GB10 CI runner to CI pipeline
- Run 24-hour testnet mining validation

KEY FILES
---------
- BLACKWELL_TCGEN05_PLAN.md - Technical roadmap for Blackwell optimization
- BLACKWELL_CORRECTNESS_VALIDATION.md - Validation strategy for bit-identical PoUW
- HANDOFF_CONTEXT.md - This file
- miner/pearl-gemm/csrc/gemm/kernel_traits.hpp - MMA atom selection, SMEM layouts
- miner/pearl-gemm/csrc/gemm/collective_mainloop.hpp - GEMM mainloop, ldmatrix staging
- miner/pearl-gemm/csrc/gemm/collective_epilogue.hpp - Epilogue, denoise, SMEM management
- miner/pearl-gemm/csrc/gemm/heuristics.hpp - Pipeline stage calculation (union model fixed, architecture-aware)
- miner/pearl-gemm/csrc/gemm/pearl_gemm_kernel.h - Kernel launch, warpgroup sync
- miner/pearl-gemm/csrc/gemm/pearl_gemm_api.cpp - Python API, kernel dispatch
- miner/pearl-gemm/setup.py - Build configuration, CUDA arch selection, PEARL_GEMM_ARCH macro, aarch64 platform fix
- miner/pearl-gemm/tests/test_pearl_gemm.py - Unit tests + new TestBitIdenticalTranscript
- miner/pearl-gemm-build-utils/.../default_compiled_kernels.py - Kernel tile configs

IMPORTANT DECISIONS
-------------------
- PR #118 (native SM80 mma.sync) is the correct baseline, NOT PR #130 (reference backend)
- PR #130 is a functional stopgap but explicitly not performance-competitive
- CUTLASS has no int8 support for SM120/SM121, so true tcgen05 optimization is currently impossible without raw PTX or protocol changes
- The SM80 mma.sync path is viable and can be optimized further (10-30% gain potential)
- Bit-identical PoUW transcript is the most critical invariant - any kernel change must preserve this
- AxEB_size bug fix: original `(sizeof(half)*m + n)*R` had operator precedence issue, corrected to `sizeof(half)*(m+n)*R`
- Heuristic SMEM model was WRONG: it treated denoise buffers as additive to mainloop SMEM, but kernel_traits.hpp uses a UNION (only A+B OR Phase1 OR Phase2 is live at any time). The corrected formula unlocks higher stage counts for smaller tiles on Blackwell.
- Architecture guards use compile-time `#if PEARL_GEMM_ARCH >= 100` to differentiate Blackwell from Hopper
- Fork created at fps-russia/pearl for DGX Spark testing
- Build uses PEARL_GEMM_ARCH=sm_121a environment variable to target Blackwell

EXPLICIT CONSTRAINTS
--------------------
- None from user beyond what is in the original request
- Technical constraints discovered: 99 KB SMEM cap per CTA on GB10, no int8 tcgen05 in CUTLASS
- DGX Spark hardware: GB10 Grace Blackwell (sm_121a, 99 KB SMEM, CUDA 13, aarch64)

CONTEXT FOR CONTINUATION
-------------------------
- The user has a DGX Spark with GB10 (sm_121a, 99 KB SMEM, CUDA 13, aarch64)
- PR #118 is already tested and working on GB10 (author verified 2874/2874 tests, end-to-end mining on testnet)
- Community member chocofoxy also verified on 2x RTX 5060 Ti (sm_120) - 201/201 shares accepted
- The canonical mining tile is 128x256x128 with stages=2 (after PR #118's SMEM restructuring to fit 99 KB)
- CUTLASS v4.3.0 is currently pinned; v4.5.1 is available but doesn't add SM121 int8 support
- PR #118's approach uses __CUDA_ARCH__ >= 1000 guards to differentiate Blackwell from Hopper
- The build uses PEARL_GEMM_ARCH=sm_121a environment variable to target Blackwell
- Critical next step: verify the specific test case that PR #130 claims fails, then proceed with implementation
- All changes are on branch `blackwell-121a` at `https://github.com/fps-russia/pearl`
- To continue: clone, checkout branch, build with PEARL_GEMM_ARCH=sm_121a, run tests

TO CONTINUE IN A NEW SESSION:
1. Press 'n' in OpenCode TUI to open a new session, or run 'opencode' in a new terminal
2. Paste this HANDOFF CONTEXT as your first message
3. Add your request: "Continue from the handoff context above. [Your next task]"

The new session will have all context needed to continue seamlessly.
