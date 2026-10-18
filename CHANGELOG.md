# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Load-time weight quantization** ([spec 008](docs/specs/008-load-time-quantization/spec.md)): `QUANTIZE_TRANSFORMER` and `QUANTIZE_TEXT_ENCODER` (`auto`, `none`, `int8`, `fp8`, `nvfp4`; default `auto`; empty = `auto`) quantize the official Qwen weights while loading, one setting per component, via a diffusers `PipelineQuantizationConfig` (torchao for int8/fp8, NVIDIA ModelOpt for the image model's nvfp4, a torchao prototype config for the text encoder's nvfp4). The VAE, the text encoder's vision tower and `lm_head`, and the image model's input/output, timestep, modulation and norm layers stay BF16. An unsupported mode fails startup with an error naming both modes, and an explicit `int8`, `fp8` or `nvfp4` fails on CPU (`auto` resolves to `none` there)
- `get_model_info` reports the effective `quantize_transformer` / `quantize_text_encoder`, `configured_quantize_*` when they differ (e.g. `auto`), and `weight_memory_gb` (per-component weight size as stored); the load log prints the same
- `BaseImageModel.quantization_info` (default `{}`); both settings are passed through in `docker/docker-compose.yml`
- **Load-time memory mitigations** (spec 008 F-6, FR-008-10 to 12): with any quantization on, the image model and the text encoder load one at a time with memory freed between them and CUDA/host memory logged after each step (`IMAGE_GPU_MEMORY_LIMIT` is not applied on this path); compose sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` by default. `none`/`none` loading is unchanged
- NVFP4 is accepted but unsupported on GB10 with the current libraries (no resident-memory saving, spec 008 F-7); selecting it logs a startup warning
- Measured on the GPU next to vLLM: `int8`/`int8` and `fp8`/`fp8` both hold 18.5 GB of weights (BF16: 32.4 GB); results table in the README

### Changed
- **The image server now runs alongside vLLM** (spec 005 F-2 resolved): with the default `int8` quantization, ~18.5 GB of weights, it ran next to vLLM at `gpu-memory-utilization 0.6`, with ≥ ~10 GB still available while loading and ~15 GB while generating. BF16 (`none`) and 2K output still need vLLM stopped
- **Default quantization is now `auto`, which means `int8` on CUDA and `none` (BF16) on CPU** (spec 008 D-10, D-11). The server therefore loads int8-quantized weights by default on a GPU (18.5 GB instead of 32.4 GB). Set `QUANTIZE_TRANSFORMER=none` and `QUANTIZE_TEXT_ENCODER=none` for BF16. `get_model_info` reports the effective modes and `configured_quantize_*` when `auto` was resolved

## [0.3.0] — 2026-10-18

> **BREAKING** — the default text-to-image size changes from native 2K to about 1K ([spec 007](docs/specs/007-1k-default-resolution/spec.md)). `generate_image` at `1:1` now returns 1024×1024 (was 2048×2048). To get 2K back, set both `GENERATE_OUTPUT_RESOLUTION=2048` and `MAX_OUTPUT_RESOLUTION=2048`. GPU verification (1K vs 2K A/B, measured 1K peak memory) is deferred: vLLM stays up in this cycle.

### Added
- **`GENERATE_OUTPUT_RESOLUTION`** setting (multiple of 32, 512-2048, default 1024, empty = default): sizes for `generate_image` come from `QwenImage21.sizes_for(resolution)`. At 2048 it is the model card's 2K table; at any other value, the pipeline's `calculate_dimensions` rule (1024: 1:1 1024×1024, 16:9 1376×768, 9:16 768×1376, 4:3 1184×896, 3:4 896×1184, 3:2 1248×832, 2:3 832×1248)
- **`MAX_OUTPUT_RESOLUTION`** ceiling (multiple of 32, 512-2048, default 1024): startup fails with a message naming the setting if `GENERATE_OUTPUT_RESOLUTION` or `EDIT_OUTPUT_RESOLUTION` is above it
- `get_model_info` reports `generate_output_resolution` and `max_output_resolution`
- Both settings are passed through in `docker/docker-compose.yml`

### Changed
- **Breaking:** default `generate_image` output is about 1K (see above); `list_aspect_ratios` returns the table for `GENERATE_OUTPUT_RESOLUTION`. Tool docstrings and server instructions no longer promise 2K
- `QwenImage21.ASPECT_RATIOS` is now `CARD_SIZES_2048`; `BaseImageModel` gains the abstract `sizes_for(resolution)`

### Fixed
- `edit_image` with an explicit `aspect_ratio` used the 2K table, the configuration spec 006 showed produces artifacts. It now uses the size table for `EDIT_OUTPUT_RESOLUTION` (spec 007 F-5)

## [0.2.0] — 2026-10-18

> **BREAKING** — upgrade to Qwen-Image-2.1 ([spec 005](docs/specs/005-qwen-image-2-1/spec.md)). Defaults and output sizes change, `negative_defaults` is removed and `IMAGE_MODEL=qwen_image_2512` is rejected. Qwen-Image-2.1 is under the Qwen Research License (non-commercial use only). GPU smoke test passed on 2026-10-18 on a DGX Spark GB10:
> - load: 32.4 GB on the GPU, one load
> - 2K generate: ~256 s; 2K transparent: ~256 s; 2K edit: ~372 s
> - peak host RAM: ~68 GB (generate), ~76 GB (edit)
>
> - 0.2.0 GPU acceptance (spec 006): edits at the new 1024 default take ~61 s with ~49 GB peak host RAM (2048: ~340 s, ~90 GB); seed determinism, container lifecycle, the bridge network, a 12.7 MB base64 reference and the 003 save workflow all verified
>
> Run it with vLLM stopped. The 2K edit artifacts (spec 005 F-1) are fixed in this release: `EDIT_OUTPUT_RESOLUTION` ([spec 006](docs/specs/006-open-items-closeout/spec.md)) now defaults to 1024, which the A/B test found clean where 2048 was not.

### Added

- **Qwen-Image-2.1 backend** (`models/qwen_image_2_1.py`, `diffusers.QwenImage21Pipeline`, `Qwen/Qwen-Image-2.1`): native 2K text-to-image, editing and RGBA output; ~33 GB of weights
- **`edit_image` tool** — edit or combine 1-10 ordered reference images (store `image_id`s and/or base64 PNG/JPEG/WebP); output follows the last reference's aspect ratio at about `EDIT_OUTPUT_RESOLUTION`² area (default 1024) unless `aspect_ratio` is set
- **`transparent` flag** on `generate_image` and `edit_image` — RGBA output via the model card's transparency phrasing
- **Prompts `transparent_asset` and `extract_subject`**
- `get_model_info` reports `max_reference_images`, `supports_editing`, `supports_transparency`, `has_default_negative_prompt` and the resolved defaults; `images://list` and `get_latest_image_id` include `mode` and `transparent`
- `MAX_REFERENCE_IMAGE_MB` (default 20)
- **`EDIT_OUTPUT_RESOLUTION`** setting (multiple of 32, 512-2048, default 1024 (chosen by the A/B test: 2048 edits were over-sharpened and ignored instructions), empty = default), passed to the backend as the pipeline's `output_resolution` for edits; `get_model_info` reports `edit_output_resolution` ([spec 006](docs/specs/006-open-items-closeout/spec.md))
- **`MAX_REQUEST_BODY_MB`** setting (default computed: 268 MB at `MAX_REFERENCE_IMAGE_MB=20`), passed to `FastMCP(max_request_body_size=…)`; the SDK's 4 MiB default made `edit_image` reject base64 references above ~3 MB with HTTP 413 (spec 006 F-4)
- `AI_BRIDGE_NETWORK` for `docker-compose.override.yml` (default `ai-litellm-proxy_ai-bridge`); the override previously referenced a network that doesn't exist (spec 006 F-3)
- Compose named volumes `image-gen-cuda-cache` and `image-gen-torch-cache` persist the CUDA JIT and PyTorch kernel caches across container recreation (spec 006)
- Tests for the 003 retrieval tools, `EDIT_OUTPUT_RESOLUTION`, the MCP path and the compose volumes (`tests/test_deploy.py`)
- Generation and editing calls are serialized by a single lock
- `storage/references.py`, `ImageStore.load_image()`, and a GPU-free test suite (`tests/conftest.py`, `test_models.py`, `test_references.py`, `test_tools.py`, `pytest.ini`)
- **`get_image_base64(image_id)` tool** — returns a stored image as a plain base64 PNG string so agents can save images to the client filesystem; docstring documents decode-before-write ([spec 003](docs/specs/003-image-retrieval-tools/spec.md))
- **`get_latest_image_id()` tool** — returns the id and metadata of the most recently generated image without reading the `images://list` resource ([spec 003](docs/specs/003-image-retrieval-tools/spec.md))
- **`IMAGE_GPU_MEMORY_LIMIT` env var** — caps the model's device memory (GB) via accelerate `max_memory`, offloading excess layers to CPU. On GB10 unified memory this bounds GPU placement, not total RAM, so it does **not** let the image model coexist with vLLM by itself (see spec 004 correction). Validated as a `Settings` field. Unset = uncapped with a startup warning ([spec 004](docs/specs/004-memory-bounded-loading/spec.md))
- Startup logging of memory mode and post-load PyTorch CUDA allocated/reserved memory
- **Spec-driven development docs** in `docs/specs/` — project constitution plus spec, research, data model, contracts, plan and tasks for features 001–005, with requirement traceability ([index](docs/specs/SPECS.md))
- `AGENTS.md` ([agents.md](https://agents.md) open format) with setup/test commands, architecture, code style, pitfalls, spec workflow, security and PR guidelines for all coding agents; `CLAUDE.md` imports it via `@AGENTS.md`

### Changed

- **Breaking:** `generate_image` defaults are now backend-native: 40 steps, `cfg_scale` 1.0 (guidance off) and no negative prompt (was 50 / 4.0 / Chinese default); `num_inference_steps`, `cfg_scale` and `negative_prompt` default to `null`. Steps are clamped to [20, 100] and `cfg_scale` to [1.0, 10.0]
- **Breaking:** output sizes are the native 2K table (1:1 = 2048×2048, was 1328×1328)
- `MAX_INFERENCE_STEPS` / `DEFAULT_CFG_SCALE` are optional overrides (unset by default); removed from the Dockerfile `ENV` and compose; `ENV IMAGE_MODEL=qwen_image_2_1`
- Model-specific facts (size table, defaults, reference limit, transparency) live on the backend; `server.py` imports nothing model-specific (its docstrings still describe the active backend, as NFR-005-05 requires)
- Dependencies: `diffusers>=0.41.0`, `transformers>=5.17`, `mcp[cli]>=1.27.0,<2` (mcp 2.x removed `mcp.server.fastmcp`)
- Docs now state that the memory cap does not let the server fit alongside vLLM on unified memory: stop vLLM first
- `photorealistic` prompt wording is model-neutral
- **Model loading uses `device_map="balanced"`** instead of `device_map="cuda"` (v0.1.1) — required for `max_memory` to be honored; still avoids post-load `.to()` (meta tensor fix preserved)
- `docker-compose.yml` sets `IMAGE_GPU_MEMORY_LIMIT: "40"` by default
- **Fixed:** `MCP_PATH` is now passed to `FastMCP` as `streamable_http_path` (it was only logged)
- **Dockerfile:** app source is copied after `pip install`, so code-only rebuilds reuse the cached dependency layer
- Version is `0.2.0` in `pyproject.toml` (was `1.0.0`, which conflicted with the 0.x history)

### Removed

- Qwen-Image-2512 backend (`models/qwen_image_2512.py`), its `DEFAULT_NEGATIVE_PROMPT`, `config/aspect_ratios.py` and the `negative_defaults` prompt
- `deploy.resources.limits.memory: 40g` from `docker-compose.yml` (added in `a42846d`) — a cgroup limit does not change model placement and only OOM-kills the container; replaced by `IMAGE_GPU_MEMORY_LIMIT`

### Known issues

Tracked as open tasks in `docs/specs/`:

- The image server (~68 GB peak for a 2K generate) can't run alongside vLLM at `gpu-memory-utilization 0.6`; stop vLLM first (spec 005 F-2)

## [0.1.1] — 2026-05-04

### Fixed

- **Meta tensor `NotImplementedError` on model load** — Qwen-Image-2512 uses `accelerate` meta device initialization; calling `.to(device)` after `from_pretrained()` crashes. Now passes `device_map="cuda"` directly to `from_pretrained()` so device placement happens during weight loading.
- **Per-session lifespan causing model reload on every connection** — MCP SDK 1.27.0 runs lifespan per-session with `json_response=True`. Replaced lifespan pattern with module-level singleton (`_startup_load_model()`) loaded once before `mcp.run()`.
- **`mcp.run()` TypeError in MCP SDK 1.27.0** — `run()` only accepts `transport` argument; `host`/`port` are passed to `FastMCP()` constructor. Removed invalid arguments from `mcp.run()`.
- **Docker healthcheck always failing** — `/mcp` endpoint returns HTTP 406 without MCP headers, causing `curl -f` to fail. Replaced with Python socket connect check to verify server is listening.

### Changed

- **Docker base image** switched from `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime` to `nvcr.io/nvidia/pytorch:26.04-py3` for Blackwell GPU support
- **Docker container** now runs as root (required by NVIDIA NGC container)
- **Removed `uv` dependency** in Dockerfile; using direct `pip install` with NVIDIA-optimized torch/CUDA
- **Added CPU fallback path** in model loading when CUDA is unavailable
- **`images://list` resource** now includes `width`, `height`, and `generation_time_seconds` fields
- **`get_model_info()`** now includes `gpu_memory_reserved_gb` field
- **`torch.cuda.synchronize()`** added to `unload()` for proper GPU memory cleanup
- **Docker `restart` policy** changed from `unless-stopped` to `no`

### Added

- `HF_HOME` environment variable in docker-compose for explicit HuggingFace cache path
- `entrypoint` in docker-compose explicitly set to `["python", "server.py"]`
- Improved docstrings throughout with device placement and usage notes

### Removed

- Non-root user (`appuser`) from Dockerfile (NVIDIA NGC containers run as root)
- `ctx: Context` parameter from `get_model_info()` (no longer needed with singleton pattern)

## [0.1.0] — 2026-05-03

### Added

- **Model-agnostic architecture** with `BaseImageModel` abstract interface for pluggable backends
- **Qwen-Image-2512 backend** — official diffusion pipeline via `diffusers` library
- **FastMCP lifespan pattern** — pipeline loaded once at server startup, shared across all requests
- **Thread-safe `ImageStore`** — in-memory metadata with disk persistence (PNG format)
- **Pydantic `BaseSettings`** — all configuration via environment variables, validated at startup
- **Aspect ratio module** — 7 official aspect ratios with pixel dimensions from Qwen model card
- **MCP tools**:
  - `generate_image` — text-to-image with progress reporting
  - `list_aspect_ratios` — supported dimensions reference
  - `get_model_info` — model metadata and GPU memory status
- **MCP resources**:
  - `image://{image_id}` — retrieve previously generated images
  - `images://list` — list all generated images with metadata
- **MCP prompts**:
  - `photorealistic(subject, setting, style)` — prompt enhancement template
  - `negative_defaults()` — recommended negative prompt
- **Docker support**:
  - Multi-stage Dockerfile (CUDA 12.8, non-root user, healthcheck)
  - `docker-compose.yml` for standalone deployment
  - `docker-compose.override.yml` for DGX Spark stack integration (`ai-bridge` network)
- **Test suite**:
  - Unit tests for model layer (mocked, no GPU required)
  - Integration tests for config, storage, and model factory

### Changed

- Refactored monolithic `server.py` into modular architecture (`config/`, `models/`, `storage/`)
- Switched from lazy-loading to lifespan-based model loading (official MCP SDK 1.27 pattern)
- Replaced global dict image store with thread-safe `ImageStore` class
- Added `asyncio.to_thread()` for blocking diffusion calls — never blocks event loop
- Added `json_response=True` on FastMCP for Streamable HTTP production scalability
- Added `ctx.report_progress()` for generation progress feedback to MCP clients

### Technical Details

- Python 3.11+
- MCP SDK 1.27.0 (`mcp.server.fastmcp`)
- Pydantic v2 + `pydantic-settings`
- `diffusers` >= 0.33.0
- `torch` >= 2.7.0 with CUDA support
- Streamable HTTP transport (production standard)