# Load-Time Weight Quantization (Feature 008) — Summary Report

**Date:** 2026-10-18
**Branch:** `feature/quantization` (from `release/0.3.0`; rebase onto `master` once 0.3.0 merges)
**Spec:** [`docs/specs/008-load-time-quantization/`](../../docs/specs/008-load-time-quantization/spec.md)
**Status:** Implemented and GPU-verified alongside vLLM. **F-2 resolved.** Not committed.

---

## Outcome

The image server now **runs alongside the vLLM Qwen3.8-27B engine** (`gpu-memory-utilization 0.6`) by default:

| | BF16 (before) | **int8 / int8 (new default on CUDA)** |
|---|---|---|
| Weights in memory | 32.4 GB | **18.5 GB** (−43%) |
| Fits next to vLLM at 0.6 | ❌ | ✅ ≥ ~10 GB still available while loading, ~15 GB while generating |
| 1K generate | — | ~62 s |
| 1024 edit | 61 s (vLLM stopped) | 72 s |
| Quality | reference | exact lettering, clean transparency, edit ≈ BF16 |

The default is `auto`: `int8` on CUDA, and `none` (BF16) on CPU, so local CPU development keeps working. BF16 and 2K output still need vLLM stopped.

## What was built

- **`QUANTIZE_TRANSFORMER` and `QUANTIZE_TEXT_ENCODER`** (`auto` | `none` | `int8` | `fp8` | `nvfp4`). The **official Qwen weights** are quantized while loading, with torchao (8-bit) and NVIDIA ModelOpt (NVFP4). There are no third-party downloads.
- Each component keeps its sensitive layers in BF16: the text encoder's vision tower and `lm_head`, and the image model's input/output, timestep, modulation and norm layers. The module names were verified on meta-device models.
- **Staged loading:** the image model, then the text encoder, then the pipeline, with `gc` and `empty_cache` in between and a memory log line after each step. Compose sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Startup fails loudly when a mode can't load, naming the stage and the modes. There's no silent fallback.
- `get_model_info` reports the effective and configured modes and the per-component weight memory.
- Tests: **174 passing** (152 in round 1, 157 after its fixes, then 163 → 168 → 174).

## GPU runs (all alongside vLLM, under an 8 GB memory watchdog)

| Run | Config | Result |
|-----|--------|--------|
| 1 | nvfp4 / nvfp4 | **Watchdog fired** at 8.2 GB while the text encoder was loading. vLLM was unaffected. → **F-6** |
| 2 | nvfp4 / nvfp4 (staged, with memory logs) | Image model **14.23 GB allocated, its full BF16 size, so no saving**. The watchdog fired again during the text encoder. → **F-7**: NVFP4 doesn't reduce memory with the installed diffusers 0.41, ModelOpt 0.42 and torchao 0.17 on GB10. |
| 3 | fp8 / fp8 | ✅ 18.53 GB, ready in 191 s, minimum available memory 9.8 GB (load) / 15.5 GB (generation). Times 176 / 86 / 87 / 121 s. Quality ✅. |
| 4 | int8 / int8 | ✅ 18.54 GB, ready in 194 s, minimum available memory 10.2 / 14.6 GB. Times **77.5 / 62 / 62 / 71.6 s**. Quality ✅. |

Thumbnails: `docs/specs/008-load-time-quantization/ab/`. Full table: research R-7.

## Process

- **Planning:** 4 decisions (D-1 to D-4). Spec approved.
- **Mid-course, after the GPU findings:** decisions D-5 to D-11, each the operator's choice: mitigations first, a vLLM window rule, keep all modes, then drop NVFP4, keep it documented as unsupported, an int8 default, and `auto`.
- **Implementer (Sonnet) and reviewer (Sonnet): 5 review rounds, the maximum.**
  - R1: changes requested. 1 major (setup errors escaped the mode-naming wrapper), 3 minor.
  - R2: approved.
  - R3: approved the F-6 mitigations, after checking the library source to confirm that staged loading streams weights one at a time.
  - R4: approved the NVFP4 warning, with 2 minor doc notes.
  - R5: approved the `auto` default, with 4 minor doc notes.
  - The orchestrator applied the remaining doc notes during close-out, because no review rounds were left.
- **Standing rule followed:** both watchdog trips were stopped and reported to the operator before continuing. vLLM was never stopped and stayed healthy throughout.

## Open items

| Item | Status |
|------|--------|
| NVFP4 on GB10 | Unsupported with the current libraries (F-7). The option stays, with a startup warning. Re-test after diffusers, ModelOpt or torchao updates. |
| Load-time margin | ~10 GB still available during startup with vLLM idle. Starting while vLLM is under heavy load could cut that margin. |
| 007 AS-6 to AS-8 (BF16 1K vs 2K A/B) | Still deferred. It needs a vLLM-down window. |
| 0.3.0 release | `release/0.3.0` is awaiting merge, then the `v0.3.0` tag. Rebase this branch onto `master` afterwards. |
| Commit and push | Waiting for the operator |

## Host state

- vLLM `qwen3-8-27b-nvfp4-engine`: healthy, never stopped.
- **`local-image-gen-mcp` is running** (int8/int8, healthy), left up at the operator's request.
- The `docker-local-image-gen-mcp` image is rebuilt with the final code, for future starts. The running container is unchanged; its explicit `int8` environment variables equal the new default.
