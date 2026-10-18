# Feature 006 — Open Items Close-out and 0.2.0 Release

| | |
|---|---|
| **Status** | Implemented and GPU-verified 2026-10-18. Review approved after 3 rounds plus a final doc check. |
| **Constitution** | II, III, IV, VI, VII, VIII |
| **Depends on** | [001](../001-image-generation-server/spec.md) – [005](../005-qwen-image-2-1/spec.md) |
| **Out of scope** | 005 F-2 (memory footprint and coexisting with vLLM), by operator decision |
| **Next** | [research.md](research.md) → [plan.md](plan.md) → [tasks.md](tasks.md) |

> **Note (007):** statements here about 2K text-to-image sizes, or `edit_image` with `aspect_ratio` using the 2K table, are superseded by [007](../007-1k-default-resolution/spec.md): the default text-to-image size is now 1024 and the F-5 bug is fixed.

## Problem

After 005, these items were left open in `docs/specs/`:

| # | Item | Source |
|---|------|--------|
| O-1 | 2K edits show over-sharpening and HDR-like artifacts | 005 F-1, T022 |
| O-2 | The Dockerfile copies code before `pip install`, so every code change reinstalls all deps | 005 T023 |
| O-3 | `MCP_PATH` is logged but never passed to `FastMCP` | 001 FR-001-13, T017 |
| O-4 | Version mismatch: `pyproject.toml` says 1.0.0, the CHANGELOG history is 0.1.x, and the 005 contract assumes 0.2.0 | 005 contract |
| O-5 | Manual GPU acceptance never run: 001 AS-2, AS-5; 002 AS-1–AS-6; 003 T007 | 001 T020, 002 T005, 003 T007 |
| O-6 | The 003 acceptance tests are only partly covered | 003 T004 |
| O-7 | The 001 contract still lists `negative_defaults` and points nowhere for the 003/005 changes | 003 T005 |
| O-8 | Stale checkboxes and obsolete tasks: work done in 005 is not ticked, and 004 T005 measures a scenario that M-1 ruled out | 001 T014; 003 T004/T005/T008; 004 T003/T004/T005/T007 |
| O-10 | **F-3 (found in the GPU window, 2026-10-18):** `docker-compose.override.yml` declares the external network `ai-bridge` with no `name:`. No network has that name; the LiteLLM stack's network is `ai-litellm-proxy_ai-bridge`. The override has never worked. | 002 AS-4 |
| O-11 | **F-4 (found in the GPU window, 2026-10-18):** the MCP SDK rejects request bodies over 4 MiB (`DEFAULT_MAX_REQUEST_BODY_SIZE`) with HTTP 413, before the tool runs. So `edit_image` can't receive base64 references above ~3 MB, even though 005 allows 20 MB each and up to 10. Unit tests call the tool function directly and missed it. | 005 FR-005-13 |
| O-9 | GPU kernels compiled at runtime (~195 MB in `/root/.nv/ComputeCache`, plus `/root/.cache/torch/kernels`) live in the container's writable layer, so they're lost whenever the container is recreated (`compose down`/`up`, `up --build`) | Operator question, 2026-10-18 |

## Decisions (2026-10-18)

| # | Decision | Choice |
|---|----------|--------|
| D-1 | F-1 fix | Run an **A/B test** of `output_resolution` 1024 vs 2048 on the GPU. Expose it as the **`EDIT_OUTPUT_RESOLUTION` env setting**, defaulting to the cleaner value. The MCP tool contract is unchanged. |
| D-2 | Version | **0.2.0**: set it in `pyproject.toml`, turn `[Unreleased]` into `[0.2.0] — 2026-10-18`, and tag `v0.2.0` **when the branch merges** (the agents don't tag) |
| D-3 | GPU window | Cover **all** pending manual checks (O-5) plus the A/B test, in one window with vLLM stopped, restarting vLLM afterwards |
| D-4 | Git history | Leave the pushed commits as they are. New work goes on top of `feature/qwen-image-2-1-edit-rgba-upgrade`. |
| D-6 | F-4 fix | A new `MAX_REQUEST_BODY_MB` setting, passed as `FastMCP(max_request_body_size=…)`. Its default is computed so the documented worst case fits (operator, 2026-10-18). |
| D-5 | Compile caches | Persist both compile caches in a named Docker volume, and measure cold vs warm start (operator, 2026-10-18) |

## User stories

**US-1 (P1)** As an agent, `edit_image` returns clean edits without over-sharpening artifacts.

**US-2 (P2)** As an operator, I can set the edit resolution through the environment, and `get_model_info` shows the value in effect.

**US-3 (P2)** As an operator, rebuilding the image after a code-only change reuses the dependency layer and takes seconds.

**US-4 (P3)** As an operator, `MCP_PATH` really changes the endpoint path.

**US-6 (P2)** As an operator, recreating or rebuilding the container doesn't throw away the GPU kernels compiled on earlier runs. After a driver or library update, kernels still recompile automatically.

**US-5 (P2)** As a maintainer, the version, CHANGELOG and specs agree, and every acceptance scenario has a recorded result.

## Functional requirements

### Edit resolution (O-1)

- **FR-006-01** `Settings` MUST gain `edit_output_resolution: int`, read from `EDIT_OUTPUT_RESOLUTION`. It MUST be validated as a multiple of 32 in [512, 2048], and the default is the A/B winner (FR-006-03). It is passed to the backend through `load_model()`, and the backend uses it as the pipeline's `output_resolution` for edits. An empty string means the default.
- **FR-006-02** `get_model_info` MUST report `edit_output_resolution`. The `edit_image` docstring and the README MUST describe the edit size as "the last reference's aspect ratio at about `EDIT_OUTPUT_RESOLUTION`² pixel area", and give the default. Hard-coded "~2048²" wording for edits MUST be removed.
- **FR-006-03** The default MUST be chosen with the A/B protocol in [research.md R-1](research.md#r-1-f-1-ab-protocol). The protocol compares the same edits, with the same seeds, at 1024 and 2048. The results MUST be recorded in research.md: timings, peak memory, a visual assessment, and committed 512 px JPEG thumbnails in `docs/specs/006-open-items-closeout/ab/`. If neither value is clean, the default stays 2048, F-1 stays open with the new evidence, and the operator decides what to do next.

### Build (O-2)

- **FR-006-04** `docker/Dockerfile` MUST run the dependency `pip install` **before** copying any app source (`server.py`, `models/`, `config/`, `storage/`). After a change to only `server.py`, a rebuild MUST show the `pip install` step as `CACHED`.

### Compile caches (O-9)

- **FR-006-11** `docker/docker-compose.yml` MUST mount a named volume, `image-gen-compile-cache`, so that both the CUDA JIT cache (`/root/.nv/ComputeCache`) and the PyTorch kernel cache (`/root/.cache/torch`) survive container recreation. Use one volume with two subpaths, or two volumes; the plan picks. `restart: no` and all other compose settings stay the same. The existing `~/.cache/huggingface` bind mount MUST NOT be shadowed: mounting all of `/root/.cache` is not allowed.
- **FR-006-12** README and AGENTS.md MUST explain:
  - what the compile cache holds
  - that the CUDA cache is invalidated on NVIDIA driver upgrades, and that kernel caches are keyed by code and toolkit version, so updates recompile automatically ([research R-5](research.md#r-5-compile-caches))
  - how to force a full recompile: `docker volume rm …`, or `CUDA_CACHE_DISABLE=1` for one run
- **FR-006-13** The GPU window MUST measure the effect. Compare an empty volume (cold) with a populated one after `docker compose down && up` (warm), and record:
  - container start → `Uvicorn running`
  - the time of the first 20-step 1:1 `generate_image` after start
  - the volume's size

  Record the results in research R-5. If warm shows no measurable gain, keep the volume anyway (it costs little) and record that finding.

### Bridge network (O-10, F-3)

- **FR-006-14** The external network in `docker/docker-compose.override.yml` MUST declare `name: ${AI_BRIDGE_NETWORK:-ai-litellm-proxy_ai-bridge}`. *(Already applied by the orchestrator during the GPU window; 002 AS-4 passed with it.)* `test_deploy.py` MUST assert that it is external and has that default name. README and AGENTS.md MUST mention `AI_BRIDGE_NETWORK`.

### Request body size (O-11, F-4)

- **FR-006-15** `Settings` MUST gain `max_request_body_mb: int | None`, read from `MAX_REQUEST_BODY_MB`, with an empty string meaning `None`. `None` means the computed default: `ceil(10 × max_reference_image_mb × 4/3) + 1`, which is 268 MB at the default of 20. The value MUST be at least 4 (the SDK default). `server.py` MUST pass `max_request_body_size=<MB × 1024 × 1024>` to `FastMCP(...)`. The multiplier 10 is the transport-level worst case. Name it as a module constant in `server.py`, with a comment that it matches the largest reference limit of any backend. The server can't ask the model, because it isn't loaded when `FastMCP` is constructed.
- **FR-006-16** Unit tests MUST cover:
  - the computed default (20 → 268)
  - an explicit override
  - validation (`< 4` rejected; `""` gives the computed default)
  - that `FastMCP` receives the byte value

  Add an **HTTP-level** test that sends a request body over 4 MiB, but under the configured limit, through the app's ASGI stack (for example, an `httpx.AsyncClient` with `ASGITransport` against `mcp.streamable_http_app()`). It MUST NOT get 413. This closes the gap that let F-4 through.
- **FR-006-17** README, AGENTS.md, `.env.example` and the `edit_image` docstring MUST state the total request limit, and recommend `image_id`s over base64 for large or chained images.

### Transport path (O-3)

- **FR-006-05** `server.py` MUST pass `streamable_http_path=settings.mcp_path` to `FastMCP(...)` ([research R-2](research.md#r-2-fastmcp-path-setting)). A unit test MUST assert that `mcp.settings.streamable_http_path == settings.mcp_path`.

### Release (O-4)

- **FR-006-06** Set `pyproject.toml` `version = "0.2.0"`. In the CHANGELOG, rename `[Unreleased]` to `[0.2.0] — 2026-10-18` and add a new, empty `[Unreleased]` section above it. Changes made in this feature are listed under `[0.2.0]`. Update the version note in the 005 contract to say the release is 0.2.0. The `v0.2.0` tag is created at merge time, not in this feature.

### Tests and docs (O-6, O-7, O-8)

- **FR-006-07** `tests/test_tools.py` MUST cover 003 AS-1 to AS-4:
  - `get_latest_image_id` returns the newest of two images
  - it raises "Call generate_image first" when the store is empty
  - `base64.b64decode(get_image_base64(id))` is byte-identical to the stored file
  - an unknown ID raises a not-found error
- **FR-006-08** The 001 contract MUST:
  - mark `negative_defaults` and the 2512 values as superseded
  - link to the 003 contract additions and the 005 contract delta
  - list the current tool set

  Historical text stays, with strikethrough or a note.
- **FR-006-09** Bookkeeping:
  - tick the stale checkboxes in O-8, each with a short note of where the work was done
  - mark 004 T005 **obsolete**, with the reason (M-1: vLLM can't run at the same time)
  - update SPECS.md: the 006 row, close the gaps O-1 to O-8, and keep F-2 open

### Manual GPU acceptance (O-5)

These run in a single window with vLLM stopped, after the unit tests pass and the review loop approves.

- **FR-006-10** Run and record these:

  | Scenario | Check |
  |----------|-------|
  | 001 AS-2 | Two `generate_image` calls with the same prompt and `seed=42` produce pixel-identical PNGs |
  | 001 AS-5 | `list_aspect_ratios` answers within 2 s while a generation is running |
  | 002 AS-1 | The container reaches `healthy` |
  | 002 AS-2 | The status is `starting`, not `healthy`, while the model loads |
  | 002 AS-3 | After `docker compose down && up`, earlier PNGs are still on the `image-outputs` volume, and nothing is downloaded again (the HF cache size is unchanged) |
  | 002 AS-4 | With the `ai-bridge` override, another container on `ai-bridge` reaches `http://local-image-gen-mcp:8500/mcp` |
  | 002 AS-5 | After the server process exits, the container stays stopped |
  | 002 AS-6 | The image's history and config contain no `HF_TOKEN` value |
  | 003 T007 | `generate_image`, then `get_latest_image_id`, then `get_image_base64`; decode, write and open the result as a valid PNG |
  | 006 AS-4 | The A/B protocol (FR-006-03) |
| 006 AS-8 | **F-4 on GPU:** with the fixed server, `edit_image` with the 2K source PNG sent as **base64** (~12.7 MB of JSON) succeeds |
| 006 AS-7 | Cold vs warm compile cache (FR-006-13). After a recreate, the volume still contains the `ComputeCache` entries. |

  Each result goes in the owning spec's traceability table.

## Non-functional requirements

- **NFR-006-01** No change to the MCP tool contract (names, parameters, return types), except the new `get_model_info` field.
- **NFR-006-02** The full unit suite keeps passing in Docker without a GPU. New logic has tests.
- **NFR-006-03** The vLLM container `qwen3-8-27b-nvfp4-engine` is stopped only during the GPU window and restarted and confirmed healthy at the end. The image server is left stopped.

## Acceptance scenarios (unit)

- **AS-1** `EDIT_OUTPUT_RESOLUTION=1024` reaches the pipeline as `output_resolution=1024` for edits. Text-to-image calls never receive `output_resolution`.
- **AS-2** `EDIT_OUTPUT_RESOLUTION` values 1000 (not a multiple of 32), 256 and 4096 fail settings validation. An empty string gives the default.
- **AS-3** `get_model_info` includes `edit_output_resolution`.
- **AS-4** `FastMCP` is constructed with `streamable_http_path == settings.mcp_path`.
- **AS-5** 003 AS-1 to AS-4 are covered as in FR-006-07.
- **AS-6** A unit test parses `docker/docker-compose.yml` and checks:
  - the compile-cache volume is mounted at the CUDA and torch cache paths
  - it is declared under `volumes:`
  - the HF bind mount is still at `/root/.cache/huggingface`
  - `restart` is still `no`

## Traceability

| Requirement | Implementation | Test | Status |
|-------------|----------------|------|--------|
| FR-006-01 | `config/settings.py` (`edit_output_resolution`, validators); `models/__init__.py`; `models/qwen_image_2_1.py` | `test_edit_output_resolution_*`, `test_edit_output_resolution_reaches_pipeline_for_edits_only` | ✅ (default 1024, chosen by the A/B test, FR-006-03) |
| FR-006-02 | `server.py` (`get_model_info`, `edit_image` docstring, instructions); README, AGENTS.md, `.env.example` | `test_get_model_info` | ✅ |
| FR-006-03 | A/B protocol (research R-1) | — | ✅ default 1024 (A/B, see research R-1) |
| FR-006-04 | `docker/Dockerfile` (app `COPY` after `pip install`) | test-image rebuild after a `server.py` change showed the pip step `CACHED` | ✅ (cache re-check in the GPU window) |
| FR-006-05 | `server.py` `FastMCP(..., streamable_http_path=settings.mcp_path)` | `test_streamable_http_path_follows_settings` | ✅ |
| FR-006-06 | `pyproject.toml` 0.2.0; CHANGELOG `[0.2.0]`; 005 contract note | — | ✅ (tag at merge) |
| FR-006-07 | `tests/test_tools.py` | `test_latest_image_id_*`, `test_image_base64_*` | ✅ |
| FR-006-08 | 001 contract | review | ✅ |
| FR-006-09 | tasks.md of 001, 003, 004; SPECS.md | review | ✅ |
| FR-006-10 | GPU checks (see the table above) | manual, 2026-10-18 | ✅ 001 AS-2 (byte-identical) and AS-5 (0.02 s); 002 AS-1–AS-6 (after F-3); 003 T007 (valid PNG); A/B per R-1 |
| FR-006-11 | `docker/docker-compose.yml`: two named volumes, `image-gen-cuda-cache` and `image-gen-torch-cache` (not one volume with subpaths, since Docker fails to mount a subpath that doesn't exist yet) | `tests/test_deploy.py` | ✅ (mounts verified in the GPU window: both volumes mounted; CUDA cache survived a recreate) |
| FR-006-12 | README ("Rebuilds and compile caches"), AGENTS.md (Deployment notes) | review | ✅ |
| FR-006-13 | cold vs warm measurement | manual, 2026-10-18 | ✅ ready 194 s cold vs 197 s warm; first 20-step generate 136.9 s cold vs 131.0 s warm; CUDA cache 195 MB, kept across the recreate (research R-5) |
| FR-006-14 | `docker/docker-compose.override.yml` (`name: ${AI_BRIDGE_NETWORK:-ai-litellm-proxy_ai-bridge}`); README, AGENTS.md, `.env.example` | `test_override_network_is_external_with_default_name` | ✅ |
| FR-006-15 | `config/settings.py` (`max_request_body_mb`, `effective_max_request_body_mb()`); `server.py` (`MAX_REQUEST_REFERENCES`, `FastMCP(max_request_body_size=…)`) | `test_request_body_*`, `test_fastmcp_receives_body_limit_in_bytes` | ✅ |
| FR-006-16 | `tests/test_tools.py` | unit tests above and `test_body_over_4mib_is_not_rejected_with_413` (ASGI level, with a 413 control) | ✅ |
| FR-006-17 | README, AGENTS.md, `.env.example`, `edit_image` docstring, CHANGELOG | review | ✅; AS-8 on GPU ✅ (12.7 MB base64 2K reference edited in 61.3 s; was HTTP 413) |
| AS-1 – AS-6 | see above | `test_models.py`, `test_tools.py`, `test_deploy.py` | ✅ |
