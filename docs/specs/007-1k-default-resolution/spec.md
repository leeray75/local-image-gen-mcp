# Feature 007 — 1K Default Resolution and Output-Size Ceiling

| | |
|---|---|
| **Status** | Implemented and reviewed (approved in round 1). GPU checks AS-6 to AS-8 deferred (D-5). |
| **Constitution** | II, III, VI, VII, VIII |
| **Depends on** | [005](../005-qwen-image-2-1/spec.md), [006](../006-open-items-closeout/spec.md) |
| **Followed by** | 008 — weight quantization (torchao 8-bit / ModelOpt NVFP4), built on the 1K numbers measured here |
| **Next** | [research.md](research.md) → [plan.md](plan.md) → [tasks.md](tasks.md) |

## Problem

Text-to-image always produces the model card's native 2K sizes (005 decision P-4). Measured on the GB10, a 2K generate peaks at ~68 GB and takes ~256 s. With the vLLM Qwen3.8-27B engine at `gpu-memory-utilization 0.6`, only ~28–35 GB is free. So the image server can never run alongside vLLM, even with 4-bit weights, because about 36 GB of the 2K peak is activations rather than weights.

The 006 A/B test showed 1024 is the right size for edits. The pipeline's own default `output_resolution` is 1024 too. There is no way to keep the server out of 2K: the size table is hard-coded.

**Latent bug F-5:** `edit_image` with an explicit `aspect_ratio` still uses the **2K** table. That is exactly the configuration 006 showed produces artifacts, ignores instructions and fails extraction. Only edits with `aspect_ratio=null` use `EDIT_OUTPUT_RESOLUTION`.

## Decisions (operator, 2026-10-18)

| # | Decision | Choice |
|---|----------|--------|
| D-1 | How to limit size | New **`GENERATE_OUTPUT_RESOLUTION`** setting (multiple of 32, 512–2048, **default 1024**), mirroring `EDIT_OUTPUT_RESOLUTION`. The per-ratio sizes are computed from it. 2K is possible only when the operator configures it. |
| D-2 | Guardrail | New **`MAX_OUTPUT_RESOLUTION`** ceiling (**default 1024**). Startup fails if either resolution setting is above it, so reaching 2K takes two deliberate changes. |
| D-3 | Scope | Resolution only. Quantization is feature 008. |
| D-4 | Verification | A GPU A/B test of 1K vs 2K text-to-image (quality, time, memory). **Deferred by D-5.** |
| D-5 | vLLM stays up | **Do not stop the vLLM Qwen3.8 container in this cycle** (operator, 2026-10-18). So there's no GPU run: a 2K generate (~68 GB) and probably a BF16 1K generate (≤ ~40 GB estimated) don't fit in the ~28–37 GB free next to vLLM at 0.6, and running them risks swapping or out-of-memory kills on the host. **AS-6 to AS-8 are deferred** to a later GPU window chosen by the operator. The 1024 default stands on D-1 (the operator's choice) and on the 006 edit A/B. |

## User stories

**US-1 (P1)** As an operator, the server generates about 1-megapixel images by default, so its memory and time stay bounded (~40 GB estimated, ~60 s).

**US-2 (P1)** As an operator, nothing a client sends can make the server produce images above the configured ceiling. Exceeding it takes an explicit config change, and a startup check enforces that.

**US-3 (P2)** As an agent, `list_aspect_ratios` and `get_model_info` tell me the real sizes I'll get. Tool docstrings don't promise 2K.

**US-4 (P1)** As an agent, `edit_image` with an explicit `aspect_ratio` gives a clean result at the edit resolution, not a broken 2K one (F-5).

## Functional requirements

### Settings

- **FR-007-01** `Settings` MUST gain:
  - `generate_output_resolution: int`, from `GENERATE_OUTPUT_RESOLUTION`, default **1024**
  - `max_output_resolution: int`, from `MAX_OUTPUT_RESOLUTION`, default **1024**

  Both MUST be multiples of 32 in [512, 2048], and `""` means the default. This matches the existing `edit_output_resolution` rules.
- **FR-007-02** A model-level validator MUST reject the configuration, which stops startup, when `generate_output_resolution > max_output_resolution` or `edit_output_resolution > max_output_resolution`. The error MUST name the offending setting, its value and the ceiling. Example: `GENERATE_OUTPUT_RESOLUTION=2048 exceeds MAX_OUTPUT_RESOLUTION=1024; raise MAX_OUTPUT_RESOLUTION to allow it`.

### Size table (backend)

- **FR-007-03** The backend MUST provide its size table for a given resolution, through `sizes_for(resolution) -> dict[str, tuple[int, int]]`, with the same seven ratio keys as today:
  - at **2048**, the model card's official table, exactly as in 005 FR-005-04
  - at any other resolution, each ratio sized with the pipeline's `calculate_dimensions` rule: `w = round(sqrt(R² · r) / 32) · 32` and `h = round(w / r / 32) · 32`, where the height is computed from the unrounded width (R = resolution, r = ratio). See [research R-1](research.md#r-1-size-formula).

  At 1024 the table MUST be: 1:1 1024×1024 · 16:9 1376×768 · 9:16 768×1376 · 4:3 1184×896 · 3:4 896×1184 · 3:2 1248×832 · 2:3 832×1248.
- **FR-007-04** The backend's `aspect_ratios` property, used by `generate_image` and `list_aspect_ratios`, MUST return `sizes_for(generate_output_resolution)`. `load_model()` passes the setting to the backend, the same way it passes `edit_output_resolution`.

### Tools

- **FR-007-05 (F-5)** When `edit_image` gets an explicit `aspect_ratio`, it MUST size the output from `sizes_for(edit_output_resolution)`, not from the generate table.
- **FR-007-06** `get_model_info` MUST report `generate_output_resolution` and `max_output_resolution`, alongside the existing `edit_output_resolution`.
- **FR-007-07** The `generate_image` and `edit_image` docstrings, the server instructions and `list_aspect_ratios` MUST stop promising 2K. They describe sizes as "about `GENERATE_OUTPUT_RESOLUTION`² of area (default 1024; see `list_aspect_ratios`)" and, for edits, "`EDIT_OUTPUT_RESOLUTION`". The tool contract (names, parameters, return types) is unchanged.

### Docs and deployment

- **FR-007-08** The following MUST describe the new settings, the 1K default, the ceiling, how to opt into 2K (raise both settings), and the memory and time trade-off. Remove "native 2K" and "2K output" claims wherever they describe the default:
  - `.env.example`, README (environment table, tool parameters, memory table) and AGENTS.md
  - compose: pass both new settings through with `${VAR:-}`, as with `EDIT_OUTPUT_RESOLUTION`
- **FR-007-09** CHANGELOG `[Unreleased]` MUST record this as **breaking**: the default text-to-image size changes from 2K to 1K. It also records the F-5 fix and the new settings. The version bump happens when the release is cut.
- **FR-007-10** Bookkeeping:
  - annotate 005 P-4 / FR-005-04 / FR-005-10 and the 006 docs that state 2K defaults as "superseded by 007"
  - add the 007 row to SPECS.md, plus F-5 as a gap closed by 007

## Non-functional requirements

- **NFR-007-01** Unit tests MUST cover, without a GPU:
  - the 1024 table values and the 2048 card table
  - the formula at a non-default value (for example 1536)
  - settings defaults, ranges, empty strings, and the ceiling validator (both settings, plus the error message)
  - `generate_image` passing the generate-table size
  - `edit_image` with `aspect_ratio` using the edit-resolution table (F-5)
  - `get_model_info` fields
  - compose passthrough of the new variables
- **NFR-007-02** No new dependencies.
- **NFR-007-03** vLLM (`qwen3-8-27b-nvfp4-engine`) MUST NOT be stopped, and the image server MUST NOT be started, in this cycle (D-5). All verification is unit-level, in the test image, without `--gpus`.

## Acceptance scenarios

**Unit:**
- **AS-1** With default settings, `generate_image(aspect_ratio="16:9")` calls the pipeline with 1376×768.
- **AS-2** With `GENERATE_OUTPUT_RESOLUTION=2048` and `MAX_OUTPUT_RESOLUTION=2048`, `16:9` gives 2752×1536, the card table.
- **AS-3** `GENERATE_OUTPUT_RESOLUTION=2048` with the default ceiling fails settings validation with the message in FR-007-02. The same applies to `EDIT_OUTPUT_RESOLUTION=2048`.
- **AS-4** `edit_image(..., aspect_ratio="1:1")` with default settings passes 1024×1024 (F-5).
- **AS-5** `list_aspect_ratios` returns the 1024 table, and `get_model_info` includes all three resolution fields.

**GPU (manual, vLLM stopped), D-4: deferred by D-5 to a later window:**
- **AS-6** A/B test, text-to-image at 1024 vs 2048 (the server restarted between runs with both settings at 2048), using the same seeds. Cases:
  - **G1:** the photorealistic bicycle prompt from 005/006
  - **G2:** typography, the model card's neon-sign prompt with exact lettering
  - **G3:** a transparent sticker

  For each, record the output size, time, peak host RAM, a visual verdict, and for G2 the rendered-text accuracy. Commit 512 px thumbnails.
- **AS-7 (decision rule)** The **operator** reviews the A/B evidence. A 1024 default is kept unless the operator finds 1K quality unacceptable. In that case the operator picks the default, and 007 records it.
- **AS-8** The measured 1K peak memory is recorded as the baseline for 008.

## Out of scope

- Weight quantization (008).
- Changing the vLLM configuration (operator decision, in a different repo).
- Per-request size parameters on the tools.

## Traceability

| Requirement | Code | Test | Status |
|-------------|------|------|--------|
| FR-007-01 | `config/settings.py` (`generate_output_resolution`, `max_output_resolution`, shared `_empty_to_default` / `_check_resolution`) | `test_resolution_settings_defaults`, `test_new_resolution_settings_invalid`, `test_new_resolution_settings_empty_is_default` | ✅ |
| FR-007-02 | `Settings._check_ceiling` | `test_ceiling_rejects_resolution_above_max`, `test_ceiling_raised_allows_2048` | ✅ |
| FR-007-03 | `QwenImage21.sizes_for`, `CARD_SIZES_2048`, `RATIO_FACTORS`; abstract `BaseImageModel.sizes_for` | `test_aspect_ratios_default_is_1024_table_and_copy`, `test_sizes_for_2048_is_card_table_and_copy`, `test_sizes_for_formula_at_1536` | ✅ |
| FR-007-04 | `QwenImage21.aspect_ratios`; `load_model()` | `test_aspect_ratios_follow_generate_output_resolution`, `test_generate_output_resolution_reaches_backend` | ✅ |
| FR-007-05 | `server.edit_image` (`model.sizes_for(model.edit_output_resolution)`) | `test_edit_with_aspect_ratio_override`, `test_edit_aspect_ratio_uses_edit_table_not_generate_table` | ✅ |
| FR-007-06 | `server.get_model_info` | `test_get_model_info` | ✅ |
| FR-007-07 | `server.py` docstrings and instructions | `test_tool_docstrings_cover_contract` | ✅ |
| FR-007-08 | `.env.example`, README, AGENTS.md, `docker/docker-compose.yml` | `test_compose_passes_resolution_settings_through` (docs by review) | ✅ |
| FR-007-09 | CHANGELOG `[Unreleased]` | review | ✅ |
| FR-007-10 | 005 spec and contract, 006 spec note, SPECS.md | review | ✅ |
| NFR-007-01 | `tests/test_models.py`, `tests/test_tools.py`, `tests/test_deploy.py` | full suite, 120 passed | ✅ |
| NFR-007-02 | no dependency changes | review | ✅ |
| NFR-007-03 | vLLM untouched, no server start, no `--gpus` | process | ✅ |
| AS-1 | | `test_generate_uses_1024_table_by_default` | ✅ |
| AS-2 | | `test_generate_uses_card_table_at_2048` | ✅ |
| AS-3 | | `test_ceiling_rejects_resolution_above_max` (both settings) | ✅ |
| AS-4 | | `test_edit_aspect_ratio_uses_edit_table_not_generate_table` | ✅ |
| AS-5 | | `test_list_aspect_ratios_real_backend_is_1024_table`, `test_get_model_info` | ✅ |
| AS-6 | GPU A/B 1K vs 2K | | ⏳ deferred (D-5) |
| AS-7 | operator decision | | ⏳ deferred (D-5) |
| AS-8 | measured 1K peak memory | | ⏳ deferred (D-5) |
