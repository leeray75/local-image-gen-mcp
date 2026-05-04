# Bugfix v0.1.1 — Summary Report

**Date:** 2026-05-04
**Branch:** `bugfix/meta-tensor-and-lifespan-fixes`
**Base commit:** `d7b7080` (v0.1.0 initial release)

---

## Overview

This report documents four bugfixes and several improvements made since v0.1.0, addressing critical runtime issues that prevented the server from operating correctly in production Docker environments. The changes affect the model loading pipeline, MCP server lifespan management, Docker container configuration, and healthcheck reliability.

---

## Bugs Fixed

### 1. Meta Tensor `NotImplementedError` on Model Load

**File:** `models/qwen_image_2512.py`

**Symptom:** Server crashes on startup with `NotImplementedError: Cannot copy out of meta tensor` when loading Qwen-Image-2512.

**Root cause:** Qwen-Image-2512 uses `accelerate`'s meta device initialization internally. Calling `.to(device)` after `from_pretrained()` attempts to copy meta tensors which have no allocated data, raising `NotImplementedError`.

**Fix:** Pass `device_map="cuda"` directly to `from_pretrained()` so `accelerate` handles device placement during weight loading, avoiding post-hoc tensor copies. Added CPU fallback path (no device_map) when CUDA is unavailable.

**Impact:** Critical — server could not start on GPU.

---

### 2. Per-Session Lifespan Causing Model Reload on Every Request

**File:** `server.py`

**Symptom:** Logs show `Loading image model` on every MCP session connection, including healthchecks. Model is unloaded and reloaded repeatedly, wasting GPU memory and time.

**Root cause:** In `mcp.server.fastmcp` 1.27.0 with `json_response=True`, the lifespan context is run per-session, not once at server startup. Each new MCP session triggers a new lifespan cycle.

**Fix:** Replaced the lifespan pattern with a module-level singleton. Model is loaded once via `_startup_load_model()` called in `__main__` before `mcp.run()`, and accessed via `get_model()` function. This completely sidesteps the per-session lifespan issue.

**Impact:** Critical — model reloaded on every connection, making the server unusable under load.

---

### 3. `mcp.run()` TypeError in MCP SDK 1.27.0

**File:** `server.py`

**Symptom:** `TypeError` when calling `mcp.run(transport="streamable-http", host=..., port=...)`.

**Root cause:** In MCP SDK 1.27.0, `FastMCP.run()` only accepts `transport` argument. Host and port are passed to the `FastMCP()` constructor, not `run()`.

**Fix:** Removed `host`/`port` arguments from `mcp.run()`, keeping only `transport="streamable-http"`. Host and port are already passed to `FastMCP()` constructor.

**Impact:** Critical — server crashes at startup when `mcp.run()` is called.

---

### 4. Docker Healthcheck Always Failing

**Files:** `docker/Dockerfile`, `docker/docker-compose.yml`

**Symptom:** Container always in `unhealthy` state despite server running correctly.

**Root cause:** The `/mcp` endpoint returns HTTP 406 (Not Acceptable) without proper MCP-protocol headers. Using `curl -f` treats 406 as failure, so the healthcheck always fails.

**Fix:** Replaced `curl -f http://localhost:8500/mcp` with Python socket connect check: `python3 -c "import socket; s=socket.socket(); s.connect(('localhost', 8500)); s.close()"`. This confirms the server is listening on the port without depending on HTTP response codes.

**Impact:** High — container orchestration and monitoring cannot detect server health.

---

## Changes by File

### `models/qwen_image_2512.py`
| Change | Type |
|--------|------|
| Use `device_map="cuda"` in `from_pretrained()` | Bugfix |
| CPU fallback path without device_map | Enhancement |
| `torch.cuda.synchronize()` in `unload()` | Enhancement |
| Improved docstrings with device placement notes | Documentation |
| Format: line wrapping, comment clarity | Style |

### `server.py`
| Change | Type |
|--------|------|
| Module-level singleton replacing lifespan pattern | Bugfix |
| `mcp.run(transport="streamable-http")` only | Bugfix |
| `get_model()` function for model access | Refactoring |
| `_startup_load_model()` in `__main__` before `mcp.run()` | Refactoring |
| Removed `ctx: Context` from `get_model_info()` | Refactoring |
| Added `width`, `height`, `generation_time_seconds` to `images://list` | Enhancement |
| Added `gpu_memory_reserved_gb` to `get_model_info()` | Enhancement |
| Improved logging format and docstrings | Documentation |

### `docker/Dockerfile`
| Change | Type |
|--------|------|
| Base image: `nvcr.io/nvidia/pytorch:26.04-py3` | Enhancement |
| Removed non-root user (`appuser`) | Change (required by NVIDIA container) |
| Removed `uv`, using direct `pip install` | Change |
| Pinned `pillow==11.1.0` | Change |
| Healthcheck: Python socket connect | Bugfix |

### `docker/docker-compose.yml`
| Change | Type |
|--------|------|
| Added `entrypoint: ["python", "server.py"]` | Enhancement |
| HF cache path: `/root/.cache/huggingface` | Change (matches root user) |
| Added `HF_HOME` environment variable | Enhancement |
| Healthcheck: Python socket connect | Bugfix |
| `restart: no` instead of `unless-stopped` | Change |

---

## New Files (Untracked)

| File | Description |
|------|-------------|
| `ai-workspace/plans/bug-fix.md` | Fix plan for lifespan singleton pattern |
| `ai-workspace/reports/bugfix-v0.1.1-summary.md` | This report |

---

## Testing Recommendations

1. **GPU load test:** Start container, verify model loads once, check logs for single `Loading image model` message
2. **Healthcheck:** `docker ps` should show `healthy` after start period (180s)
3. **Image generation:** Call `generate_image` tool, verify image returns without meta tensor errors
4. **CPU fallback:** Set `CUDA_VISIBLE_DEVICES=""`, verify server starts with CPU warning
5. **Memory check:** Call `get_model_info`, verify `gpu_memory_reserved_gb` field present
6. **Image list:** Call `images://list`, verify `width`, `height`, `generation_time_seconds` fields

---

## Documents Updated

- `CHANGELOG.md` — v0.1.1 entry added
- `README.md` — Docker section updated for NVIDIA container base image