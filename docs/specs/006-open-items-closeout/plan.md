# Feature 006 — Implementation Plan

**Inputs:** [spec.md](spec.md), [research.md](research.md)
**Branch:** `feature/qwen-image-2-1-edit-rgba-upgrade` (continued, D-4)

## Constitution check

| Principle | Compliance |
|-----------|------------|
| II Model-agnostic | `edit_output_resolution` is a setting the backend receives. `server.py` reads it only to report it. |
| III Env-only config | `EDIT_OUTPUT_RESOLUTION` is validated in `Settings` |
| IV Verify the SDK | `streamable_http_path` was confirmed in mcp 1.30.0 (R-2) |
| VI Shared host | vLLM is stopped only during the GPU window and restarted afterwards (NFR-006-03) |
| VII GPU-free tests | All new logic is unit-tested with mocks. GPU checks are manual and recorded. |

## Changes

> Final default: 1024, chosen by the A/B test (research R-1). The 2048 and "provisional" wording below is historical.

| File | Change |
|------|--------|
| `config/settings.py` | `edit_output_resolution: int = 2048` (provisional, FR-006-03), a validator (multiple of 32 in [512, 2048]), and adding it to the empty-string → default handling |
| `models/qwen_image_2_1.py` | `__init__(…, edit_output_resolution: int = 2048)`, stored as `self._edit_output_resolution` and used instead of the `EDIT_OUTPUT_RESOLUTION` class constant (remove the constant, or keep it only as the default value) |
| `models/base.py` | Optional property `edit_output_resolution -> int | None`, default `None`, for backends without editing |
| `models/__init__.py` | Pass `edit_output_resolution=settings.edit_output_resolution` |
| `server.py` | `FastMCP(..., streamable_http_path=settings.mcp_path)`. Add `edit_output_resolution` to `get_model_info`. Reword the `edit_image` docstring and the instructions (FR-006-02). |
| `docker/Dockerfile` | Move the app `COPY` lines below `RUN pip install` (R-3) |
| `docker/docker-compose.yml` | A named volume, `image-gen-compile-cache`, mounted twice with `volume.subpath` (`nv` → `/root/.nv`, `torch` → `/root/.cache/torch`). If the installed Compose version doesn't support `subpath`, use two named volumes instead (`image-gen-cuda-cache`, `image-gen-torch-cache`) and record the choice. Declare them under top-level `volumes:`. (FR-006-11) |
| `.env.example`, `README.md`, `AGENTS.md` | Document `EDIT_OUTPUT_RESOLUTION`. Remove "~2048²" for edits. Note the cached rebuild. Drop the T023 known-issue wording. |
| `pyproject.toml` | `version = "0.2.0"` |
| `CHANGELOG.md` | `[Unreleased]` becomes `[0.2.0] — 2026-10-18`, with a new empty `[Unreleased]` above it. Add 006 entries. Update the known issues. |
| `tests/test_models.py`, `tests/test_tools.py`, `tests/test_deploy.py` (new) | AS-1 to AS-6. `test_deploy.py` parses the compose YAML with PyYAML, if the image has it; otherwise it skips with a reason. |
| `docs/specs/**` | FR-006-08 and FR-006-09 bookkeeping; the 005 contract version note; the 006 traceability table |

## Handling empty strings in Settings

`edit_output_resolution` isn't optional, so `""` has to mean "use the default". Implement this as a separate `mode="before"` validator that returns the field default for `""`, followed by the range and multiple-of-32 check. Don't make the field `int | None`.

## GPU window runbook (done by the orchestrator, not the implementer)

Run this only after the unit tests pass and the reviewer approves:
1. `docker stop qwen3-8-27b-nvfp4-engine`
2. **002 AS-6:** check `docker history --no-trunc` and `docker inspect` of the image for any `HF_TOKEN` value.
3. Rebuild with compose (FR-006-04 check: touch `server.py` and rebuild again, then look for `CACHED` on the pip step).
4. Make sure the compile-cache volume(s) don't exist yet, so the start is cold. Start with the `ai-bridge` override, and **time start → `Uvicorn running`** (FR-006-13, cold).
   - **002 AS-2:** poll the status while loading.
   - **002 AS-1:** wait for `healthy`.
   - **002 AS-4:** reachability from a throwaway container on `ai-bridge` (R-4).
5. **001 AS-2:** two generates with seed 42, using `num_inference_steps=20` to save time, then compare hashes. **The time of the first one is the cold first-generation time** (FR-006-13).
6. **001 AS-5:** time `list_aspect_ratios` while a generate is running.
7. **003 T007:** the save workflow.
8. **A/B at 2048 (E1–E3):** with `EDIT_OUTPUT_RESOLUTION=2048`, using the 005 smoke client, extended.
9. **002 AS-3 and 006 AS-7 (warm):**
   - Record the HF cache size and the compile-cache volume size, `docker compose down` (without `-v`), then `up` with `EDIT_OUTPUT_RESOLUTION=1024`.
   - Check the earlier PNGs are still on the volume and the HF cache size is unchanged.
   - **Time start → `Uvicorn running`**, then run the same 20-step seed-42 generate first and time it (warm).
   - Confirm the `ComputeCache` entries are still in the volume.
10. **A/B at 1024 (E1–E3).**
11. **002 AS-5:** `docker exec local-image-gen-mcp kill 1`, then confirm the container is exited and not restarting.
12. Decide the default (R-1 rule), update the setting default and the docs, and rerun the unit tests.
13. `docker start qwen3-8-27b-nvfp4-engine` and wait for `healthy`.
14. Record the results in 001, 002, 003 and 006 traceability, research R-1, the CHANGELOG `[0.2.0]` and the report.

## Risks

| Risk | Mitigation |
|------|------------|
| Neither resolution is clean | The decision rule escalates to the operator. F-1 stays open with the evidence. |
| `kill 1` leaves a stopped container that confuses the operator | The runbook ends with the image server stopped either way, the same as 005. |
| The A/B test costs about 30 min of GPU time | Accepted (D-3). The O-9 measurement adds about 10 min. |
| Docker Compose doesn't support `volume.subpath` | Fall back to two named volumes (see the Changes table) |
