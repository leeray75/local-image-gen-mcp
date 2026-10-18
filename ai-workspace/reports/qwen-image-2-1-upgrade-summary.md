# Qwen-Image-2.1 Upgrade (Feature 005) — Summary Report

**Date:** 2026-10-18
**Branch:** `feature/qwen-image-2-1` (from `feature/memory-cap-retrieval-tools-sdd-specs`)
**Spec:** [`docs/specs/005-qwen-image-2-1/`](../../docs/specs/005-qwen-image-2-1/spec.md)
**Status:** Implemented, reviewed and GPU-verified. One open quality finding (F-1). Not committed.

---

## Overview

The server's only model is now **Qwen-Image-2.1** instead of Qwen-Image-2512. The upgrade adds:
- native 2K text-to-image
- a new **`edit_image`** tool (1–10 reference images, given as store IDs or base64)
- **transparent RGBA** output on both tools

The work followed the SDD workflow: planning questions, then a spec with clarifications, then operator approval, then implementation by a Sonnet implementer agent, reviewed by a separate Sonnet reviewer agent, then a GPU smoke test.

**License:** Qwen Research License, non-commercial use only. The operator has confirmed the use is personal.

---

## Process

| Stage | Outcome |
|-------|---------|
| Planning | 8 decisions (P-1 to P-8): replace 2512; full scope (t2i, edit, RGBA); backend-native defaults; native 2K table; separate `edit_image`; IDs plus base64 references; 2K-class edit output; a `transparent` flag |
| Spec clarifications | C-1 one ordered `images` list · C-2 uploads not stored · C-3 `negative_defaults` replaced by `transparent_asset` / `extract_subject` · C-4 Docker-only unit tests, GPU test after the operator's go-ahead |
| Memory check | Only ~28 GiB free with vLLM at 0.6. **M-1:** the operator stops vLLM while the image server runs. vLLM was stopped for this work and restarted afterwards. |
| Implementation, round 1 | 75 tests pass. Found that `mcp` 2.x removed `mcp.server.fastmcp`, so it is pinned to `<2`. |
| Review, round 1 | **CHANGES_REQUESTED.** 2 major findings: a decompression-bomb gap (1×–2× pixel band accepted), and the PNG save/encode blocking the event loop. 4 minor findings. |
| Implementation, round 2 | All 6 findings fixed. 77 tests pass. |
| Review, round 2 | **APPROVE.** No blocking findings. Two minor items are documented: model-specific docstrings, which NFR-005-05 requires, and the version-number mismatch. |
| GPU smoke test | AS-14 ✅. AS-15: generate ✅, transparent ✅, edit ⚠️ (F-1) |

The loop finished in **2 of the 5 allowed rounds**.

---

## What Changed

### Code
| File | Change |
|------|--------|
| `models/qwen_image_2_1.py` | **New** `QwenImage21` backend: `QwenImage21Pipeline`, `device_map="balanced"` plus `max_memory`, 2K table, transparency wrapping, RGB/RGBA conversion, edit support |
| `models/qwen_image_2512.py`, `config/aspect_ratios.py` | **Deleted**. Model-specific facts now live on the backend (Constitution II). |
| `models/base.py` | Request gains `reference_images`, `transparent`, and optional size/negative prompt. Result gains the actual size. Interface gains the ratio table, defaults, reference limit and transparency support. |
| `server.py` | `edit_image`; `transparent` flag; parameter resolution (argument → env → backend default, then clamps); a generation lock; save and encode moved off the event loop; discovery and prompt changes |
| `storage/references.py` | **New**: ordered reference resolution, strict UUID detection, base64 decoding with format, size and pixel-limit checks |
| `storage/image_store.py` | `mode`, `transparent`, `reference_image_ids`, `reference_count`; `load_image()` |
| `config/settings.py` | `qwen_image_2_1` only; optional step/CFG overrides; `IMAGE_GPU_MEMORY_LIMIT` and `MAX_REFERENCE_IMAGE_MB` added to `Settings` |
| `pyproject.toml`, `docker/*` | `diffusers>=0.41.0`, `transformers>=5.17`, `mcp[cli]<2`. The step and CFG env defaults that would override the backend are removed. |

### Tests
`tests/test_tools.py` is no longer empty. There is a new `test_references.py`, `test_models.py` is rewritten, and `conftest.py` and `pytest.ini` are added. **77 tests pass** in Docker, with no GPU.

### Docs
README, AGENTS.md, `.env.example`, CHANGELOG (breaking `[Unreleased]` entry), SPECS.md, and the 005 spec set with a full traceability table. Specs 001 and 004 are annotated as superseded where relevant. The **004 memory claim is corrected**: on GB10 unified memory, the cap bounds GPU placement, not total RAM.

### Spec gaps closed
- 001: `cfg_scale` clamp, generation serialization, the misleading `ImageStore` docstring, and the empty `test_tools.py`
- 004: `IMAGE_GPU_MEMORY_LIMIT` bypassed `Settings`

---

## GPU Smoke Test (2026-10-18, DGX Spark GB10, vLLM stopped)

| Check | Result |
|-------|--------|
| Build | diffusers 0.41.0, transformers 5.20.1, mcp 1.30.0, NGC torch |
| Load | One model load. 32.44 GB allocated. Healthy. |
| Weight cache | 31 GB in `~/.cache/huggingface/hub/blobs/`, the shared content-addressed store used by huggingface_hub 1.33. Restarts don't download again. |
| `generate_image` 1:1 | 2048×2048 RGB, **256.5 s**, photorealistic, clean |
| `generate_image` transparent | 2048×2048 RGBA, alpha range 0–255, clean sticker cut-out, **255.7 s** |
| `edit_image` (chained by ID) | 2048×2048 RGB, **372.1 s**. Metadata is correct (`mode=edit`, `aspect_ratio=input`). Layout preserved, color changed. **Over-sharpened, HDR-like artifacts.** |
| Peak host RAM | ~83 GB used during generate, ~91 GB during edit (system-wide, including ~15 GB of other services). That makes the image server ~68–76 GB at peak and ~30 GB idle. |

---

## Open Items

| # | Item | Suggested action |
|---|------|------------------|
| F-1 | 2K edits show over-sharpening and saturation artifacts | A/B the same edit at `output_resolution=1024`. If it is clean, change `EDIT_OUTPUT_RESOLUTION` or make it configurable, and revisit decision P-7. (T022) |
| F-2 | Peak memory (~68–76 GB) is far above the 38–45 GB estimate | It can't run alongside vLLM at 0.6. Either keep switching (stop one to run the other), or spec a quantized variant (e.g. 8-bit text encoder and DiT) as a new feature. |
| T023 | The Dockerfile copies code before `pip install`, so every code change reinstalls all deps | Move the `COPY` of the app code below the `RUN pip install` |
| — | Version mismatch: `pyproject.toml` says 1.0.0, the changelog history is 0.x, and the contract says 0.2.0 | Decide when the release is cut |
| — | Nothing in this feature is committed | Review, then commit on `feature/qwen-image-2-1` |

---

## Current Host State

- **vLLM `qwen3-8-27b-nvfp4-engine`:** restarted and healthy.
- **`local-image-gen-mcp`:** stopped (restart policy `no`). The rebuilt `docker-local-image-gen-mcp` image is the 2.1 version.

To use image generation, stop vLLM first:

```bash
docker stop qwen3-8-27b-nvfp4-engine
cd docker && docker compose up -d        # wait for "Uvicorn running" in docker logs (warm-cache load time not yet measured)
# … when done …
docker stop local-image-gen-mcp && docker start qwen3-8-27b-nvfp4-engine
```

The test image `local-image-gen-mcp:qwen21-test` can be deleted once it is no longer needed for running tests.
