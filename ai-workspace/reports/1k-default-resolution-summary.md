# 1K Default Resolution and Output-Size Ceiling (Feature 007) — Summary Report

**Date:** 2026-10-18
**Branch:** `feature/1k-default-resolution` (from `master` at `v0.2.0`)
**Spec:** [`docs/specs/007-1k-default-resolution/`](../../docs/specs/007-1k-default-resolution/spec.md)
**Status:** Implemented and reviewed. GPU A/B checks (AS-6 to AS-8) are deferred. Not committed.

---

## Why

On the GB10, a 2K generate peaked at ~68 GB. With vLLM running at `gpu-memory-utilization 0.6`, only ~28–37 GB is free, and about 36 GB of that 2K peak is activations, so even 4-bit weights couldn't make 2K fit. The operator decided to make ~1K the default and to make 2K impossible without an explicit configuration change.

## What changed

| Area | Change |
|------|--------|
| Settings | `GENERATE_OUTPUT_RESOLUTION` (default **1024**) and `MAX_OUTPUT_RESOLUTION` (default **1024**). Both are multiples of 32 in [512, 2048], and an empty value means the default. A startup check rejects any resolution above the ceiling, with the message `X=2048 exceeds MAX_OUTPUT_RESOLUTION=1024; raise MAX_OUTPUT_RESOLUTION to allow it`. |
| Backend | `sizes_for(resolution)`: at 2048 the official model-card table, otherwise the pipeline's own `calculate_dimensions` formula. At 1024: 1:1 1024² · 16:9 1376×768 · 4:3 1184×896 · 3:2 1248×832, plus the portrait versions. |
| **F-5 fix** | `edit_image` with an explicit `aspect_ratio` used the **2K** table, the configuration 006 showed produces broken edits. It now uses `EDIT_OUTPUT_RESOLUTION`. |
| Discovery | `get_model_info` reports `generate_output_resolution`, `edit_output_resolution` and `max_output_resolution`. `list_aspect_ratios` returns the configured table. |
| Docs | Docstrings, server instructions, README, AGENTS.md and `.env.example` no longer promise 2K, and they document how to opt in. The CHANGELOG `[Unreleased]` entry is marked **breaking**. |
| Compose | Passes both new variables through |
| Tests | **120 passed** (up from 99): formula and card table, settings and ceiling, F-5, compose passthrough |

## Process

- Planning: 4 decisions (D-1 to D-4).
- The spec was approved by the operator, with D-5 added: **vLLM must not be stopped this cycle.**
- A Sonnet implementer and an independent Sonnet reviewer. **Approved in round 1**, with no code changes requested. The one MINOR doc note was applied by the orchestrator.
- **Memory-pressure rule:** every Docker run checked `free -m` first and would have stopped and reported on any memory pressure. Available memory stayed at ~37 GB throughout, with no failures. The vLLM engine stayed healthy the whole time.

## Deferred (D-5)

| Item | Why |
|------|-----|
| AS-6: 1K vs 2K text-to-image A/B (photo, typography, transparent) | Needs the image server running. 2K (~68 GB) and probably BF16 1K (≤ ~40 GB, estimated) don't fit beside vLLM at 0.6 without risking swapping or out-of-memory kills on the host. |
| AS-7: operator review of 1K quality | Depends on AS-6. The 1024 default currently rests on decision D-1 and the 006 edit A/B. |
| AS-8: measured 1K peak, the baseline for 008 | Depends on AS-6 |

The runbook is ready in `plan.md`. Run it in a window when vLLM can be stopped, or after 008's quantization brings the footprint under the free memory.

## Host State

- vLLM `qwen3-8-27b-nvfp4-engine`: never touched, healthy.
- The `docker-local-image-gen-mcp` image is rebuilt with the 1K defaults (checked: 16:9 → 1376×768). The image server is stopped.

## Next

- **Spec 008:** quantize the official weights at load time (torchao 8-bit, ModelOpt NVFP4). Estimated 1K peaks of ~25 / ~20 GB would fit beside vLLM.
- Commit and push this branch when the operator approves.
