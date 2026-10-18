# Feature 005 — Upgrade to Qwen-Image-2.1 (text-to-image, editing, RGBA)

| | |
|---|---|
| **Status** | Implemented. GPU smoke test passed 2026-10-18, with one open quality finding on edits (F-1). |
| **Constitution** | I, II, III, IV, V, VI, VII, VIII |
| **Depends on** | [001](../001-image-generation-server/spec.md), [002](../002-container-deployment/spec.md), [003](../003-image-retrieval-tools/spec.md), [004](../004-memory-bounded-loading/spec.md) |
| **Supersedes** | 001 R-5 (2512 pipeline facts and resolution table), the 2512 backend, `negative_defaults` prompt |
| **Next** | [research.md](research.md) → [data-model.md](data-model.md) · [contracts](contracts/mcp-interface.md) → [plan.md](plan.md) → [tasks.md](tasks.md) |

## Problem

[Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) supersedes Qwen-Image-2512. It is a single model for text-to-image *and* image editing, with:
- about 33 GB of weights (2512: about 58 GB), so it needs much less memory on the shared GB10 than 2512's ~62 GB footprint
- native 2K output
- native transparent (RGBA) generation
- editing that takes up to 10 reference images

Right now the server can only generate from text, at about 1.3K, and agents have no way to edit an image, change the background of an existing image, or make a transparent asset.

## Decisions already made (planning, 2026-10-18)

| # | Decision | Choice |
|---|----------|--------|
| P-1 | Approach | **Replace** 2512. Delete the 2512 backend. 2.1 becomes the only backend. |
| P-2 | Scope | **Full**: text-to-image, image editing, and RGBA output |
| P-3 | Defaults | **Backend-native**: tool params default to `null`, meaning "use the backend's default" |
| P-4 | T2I resolution | The model card's **native 2K table** |
| P-5 | Edit tool | A separate **`edit_image`** tool. `generate_image` stays text-only. |
| P-6 | Reference input | **Store image IDs and base64** strings, which can be mixed |
| P-7 | Edit output size | Same aspect ratio as the input, **2K-class** (`output_resolution=2048`), with an optional `aspect_ratio` override |
| M-1 | Memory | Measured 2026-10-18: 121 GiB total, ~28 GiB free with vLLM at `gpu-memory-utilization 0.6` (~73 GB). **The operator stops vLLM while the image server runs.** Quantization is a possible later feature. |
| P-8 | RGBA | A **`transparent: bool` flag** on both tools |

**License:** Qwen Research License, non-commercial use only. The operator has confirmed the use is personal (2026-10-18).

## Clarifications

All resolved on 2026-10-18.

- **C-1** *Reference ordering and shape.* **Resolved: one ordered `images: list[str]`.** An entry that parses as a UUID is a store `image_id`, and anything else is treated as base64. Caller order is kept exactly, so "Picture N" is entry N.
- **C-2** *Persisting uploaded references.* **Resolved: uploads are not stored.** Only the edit result is stored. Its metadata records the store IDs used and the total reference count.
- **C-3** *Prompt templates.* **Resolved: remove `negative_defaults`.** Add `transparent_asset` and `extract_subject`. Keep `photorealistic`, with model-neutral wording.
- **C-4** *Verification environment.* **Resolved:** the agents rebuild the project Docker image and run pytest inside it, with dev deps installed at run time and a mocked pipeline, so there is no model download. The GPU smoke test (AS-14, AS-15) runs only after the operator gives the go-ahead.

## Actors

As in 001: **Agent**, **Operator**, and **Backend author**.

## User stories

**US-1 (P1) — Generate at 2K.** As an agent, I call `generate_image` with a prompt and get a native 2K PNG from Qwen-Image-2.1, using the model's recommended settings unless I override them.

**US-2 (P1) — Edit an image.** As an agent, I call `edit_image` with an instruction and 1–10 reference images (images this server generated, my own images as base64, or both) and get the edited image back.

**US-3 (P1) — Chain edits.** As an agent, I can pass the ID returned for one result as the reference for the next edit, without moving bytes back and forth.

**US-4 (P2) — Transparent assets.** As an agent, I set `transparent=true` to get a PNG with a real alpha channel (stickers, icons, cut-outs). With `transparent=false`, the result is opaque RGB.

**US-5 (P2) — Extract a subject.** As an agent, I can pull the subject out of a photo onto a transparent background with `edit_image(..., transparent=true)` and a ready-made prompt template.

**US-6 (P2) — Know what the model supports.** As an agent, `get_model_info` and `list_aspect_ratios` tell me the active backend's sizes, defaults, reference-image limit, and whether it supports transparency.

**US-7 (P1) — Fit on the shared host.** As an operator, I know how much memory the server needs and that vLLM has to be stopped first. The docs don't promise that the cap makes the two coexist.

## Functional requirements

### Backend

- **FR-005-01** A `qwen_image_2_1` backend MUST load `Qwen/Qwen-Image-2.1` with `diffusers.QwenImage21Pipeline`. It keeps 001 R-6: `device_map` goes to `from_pretrained`, with no `.to()` afterwards. It also keeps 004: `device_map="balanced"`, plus `max_memory` when `IMAGE_GPU_MEMORY_LIMIT` is set. On CPU, no `device_map`.
- **FR-005-02** The Qwen-Image-2512 backend, its default negative prompt and its resolution table MUST be removed. `IMAGE_MODEL` MUST accept `qwen_image_2_1` and default to it. Any other value fails validation at startup.
- **FR-005-03** Model-specific facts MUST live on the backend, not in server or config code (Constitution II):
  - the aspect-ratio → size table
  - the default steps, CFG and negative prompt
  - the maximum number of reference images
  - transparency support
  - how a prompt is phrased for transparency

  The server reads these through `BaseImageModel`.
- **FR-005-04** The 2.1 text-to-image size table MUST be exactly: `1:1` 2048×2048, `4:3` 2400×1792, `3:4` 1792×2400, `3:2` 2528×1696, `2:3` 1696×2528, `16:9` 2752×1536, `9:16` 1536×2752. The ratio keys are the same seven the tool schema already accepts.
- **FR-005-05** 2.1 defaults MUST be: 40 steps, `true_cfg_scale` 1.0, and no negative prompt.

### Parameter resolution (both tools)

- **FR-005-06** `num_inference_steps`, `cfg_scale` and `negative_prompt` MUST default to `null`. A `null` value resolves to the env override if one is set (`MAX_INFERENCE_STEPS`, `DEFAULT_CFG_SCALE`), and otherwise to the backend default.
- **FR-005-07** After resolution, steps MUST be clamped to [20, 100] and `cfg_scale` to [1.0, 10.0]. *(Closes 001 FR-001-04.)*
- **FR-005-08** If a `negative_prompt` is given while the resolved `cfg_scale` ≤ 1.0, the call MUST still succeed. The server MUST send an info/warning message to the client saying the negative prompt is ignored unless `cfg_scale` > 1. *(Guidance doubles the compute, and 2.1 is tuned to run without it.)*
- **FR-005-09** Seed behavior is unchanged (001 FR-001-05). Determinism holds for fixed inputs, hardware, dtype and pipeline settings. The server MUST NOT change `use_kv_cache` between calls.

### `generate_image`

- **FR-005-10** The tool keeps its name and its `prompt`, `aspect_ratio` and `seed` params. It gains `transparent: bool = false`. Its output size is the backend's table entry for `aspect_ratio`.

### `edit_image` (new)

- **FR-005-11** `edit_image` MUST take:
  - a `prompt`, which is the edit instruction
  - 1 to `max_reference_images` (10 for 2.1) references, in the order the caller gives them, as one ordered `images` list (C-1)
  - `aspect_ratio: <ratio> | null = null`
  - `transparent: bool = false`
  - `num_inference_steps`, `cfg_scale`, `negative_prompt` and `seed`, resolved as above

  It returns the edited image as MCP PNG image content.
- **FR-005-12** A reference MUST be one of:
  - **(a)** an `image_id` in the image store, or
  - **(b)** a base64-encoded PNG, JPEG or WebP image, with or without a `data:image/...;base64,` prefix
- **FR-005-13** All references MUST be validated **before** any model work starts. The call is rejected with a clear error naming the offending reference (by its 1-based position) when:
  - there are no references
  - there are more than the maximum
  - an ID is unknown, or its file is missing
  - the base64 is invalid
  - the format is unsupported
  - the decoded size is over `MAX_REFERENCE_IMAGE_MB` (default 20)
  - the image is over Pillow's decompression-bomb pixel limit
- **FR-005-14** With `aspect_ratio` null, the output MUST match the aspect ratio of the **last** reference at about 2048² pixel area (`output_resolution=2048`). With `aspect_ratio` set, the output is that ratio's table size.
- **FR-005-15** References MUST be passed to the pipeline as PIL images in caller order, with alpha preserved. The pipeline itself flattens alpha only for the vision encoder.

### Transparency

- **FR-005-16** With `transparent=true`, the backend MUST wrap the prompt in the model card's RGBA phrasing: `"This is an RGBA image with transparency. {prompt} The image has alpha channel and the background is transparent."` The returned and stored PNG MUST be RGBA.
- **FR-005-17** With `transparent=false`, the returned and stored PNG MUST be RGB. The pipeline's 4-channel output is converted.

### Storage and retrieval

- **FR-005-18** Every result from either tool MUST be saved to the store. Metadata adds `mode` (`"generate"` | `"edit"`), `transparent`, `reference_image_ids` (the store IDs used, in order) and `reference_count` (total references, including uploads). For edits with no `aspect_ratio`, the stored `aspect_ratio` is `"input"`, and `width`/`height` are the actual output size.
- **FR-005-19** `images://list` and `get_latest_image_id` MUST include `mode` and `transparent`. Existing fields and behavior (003) are unchanged.
- **FR-005-20** Base64 references MUST NOT be persisted to the store (C-2).

### Discovery

- **FR-005-21** `list_aspect_ratios` MUST return the active backend's table.
- **FR-005-22** `get_model_info` MUST additionally report `max_reference_images`, `supports_transparency`, `supports_editing`, and the resolved defaults (steps, CFG, whether there is a negative prompt) after env overrides.

### Prompts

- **FR-005-23** Remove `negative_defaults` (C-3). Add:
  - `transparent_asset(subject, style="")`: a prompt for a standalone object on a transparent background, used with `generate_image(transparent=true)`
  - `extract_subject(subject="the main subject")`: an edit instruction to isolate the subject on a transparent background, used with `edit_image(transparent=true)`

  Keep `photorealistic`, with its wording made model-neutral (no model name).

### Configuration and deployment

- **FR-005-24** `IMAGE_GPU_MEMORY_LIMIT` MUST move into `Settings` (`int | None`) and be passed to the backend from `load_model()`. The module-level `os.getenv` goes away. *(Closes 004 FR-004-05.)* Add `MAX_REFERENCE_IMAGE_MB` (`int`, default 20). `MAX_INFERENCE_STEPS` and `DEFAULT_CFG_SCALE` become optional overrides, `None` by default.
- **FR-005-25** Dependencies MUST be `diffusers>=0.41.0` and `transformers>=5.17` in both `pyproject.toml` and `docker/Dockerfile`. torch stays as the NGC build in the container.
- **FR-005-26** `.env.example`, the Dockerfile `ENV` defaults, compose, README and AGENTS.md MUST describe the new variables and defaults, and no longer mention 2512.

## Non-functional requirements

- **NFR-005-01 (Serialization)** Generation and editing calls MUST be serialized through a single lock, so that only one pipeline call runs at a time. Non-generation tools stay responsive. *(Closes 001 NFR-001-03.)*
- **NFR-005-02 (Memory)** The server needs about 33 GB of weights plus 2K activations (**measured 2026-10-18:** ~30 GB at idle, peaking at ~68 GB during a 2K generate and ~76 GB during a 2K edit, out of 121 GB host RAM) in the GB10's unified memory. On this host that means it runs with the vLLM Qwen3.8-27B engine **stopped** (decision M-1, 2026-10-18). `IMAGE_GPU_MEMORY_LIMIT` only limits GPU allocator placement, because offloaded layers live in the same physical RAM, so it does **not** make the server fit alongside vLLM. The docs MUST say this.
- **NFR-005-03 (Testability)** All new logic (parameter resolution, reference decoding and validation, transparency handling, store metadata, tool registration) MUST have unit tests that pass without a GPU. The pipeline is mocked. `tests/test_tools.py` MUST no longer be empty.
- **NFR-005-04 (Payload)** Tool docstrings MUST warn that base64 references and results are large, and recommend using image IDs for chained edits.
- **NFR-005-05 (Contract clarity)** Docstrings for `generate_image` and `edit_image` MUST state:
  - the 2K sizes
  - the guidance-off default, and that `negative_prompt` needs `cfg_scale > 1`
  - the reference limit and ordering
  - what the `transparent` flag does

## Acceptance scenarios

Unit (mocked pipeline):
- **AS-1** `generate_image(prompt)`:
  - the pipeline gets `width=2048, height=2048, num_inference_steps=40, true_cfg_scale=1.0, negative_prompt=None`
  - the result is a 2048² RGB PNG
  - the stored `mode` is `"generate"`
- **AS-2** `generate_image(prompt, aspect_ratio="16:9", transparent=True)`: the pipeline gets 2752×1536 and the wrapped RGBA prompt, and the result is RGBA.
- **AS-3** `generate_image(prompt, num_inference_steps=5, cfg_scale=50)`: the pipeline gets steps 20 and CFG 10.0.
- **AS-4** `generate_image(prompt, negative_prompt="blurry")` with default CFG succeeds, and a message says the negative prompt is ignored.
- **AS-5** `edit_image(prompt, [id_A, <base64 B>])`:
  - the pipeline gets `image=[A, B]` in that order, `output_resolution=2048`, and no width or height
  - the stored `mode` is `"edit"`, `reference_image_ids == [id_A]`, `reference_count == 2`, and `aspect_ratio == "input"`
- **AS-6** `edit_image` with `aspect_ratio="1:1"` passes `width=height=2048`.
- **AS-7** `edit_image` with 0 references, 11 references, an unknown ID, bad base64, a GIF, or a 25 MB image is each rejected with an error naming the position, and the pipeline is **not** called.
- **AS-8** A `data:image/png;base64,` prefix is accepted.
- **AS-9** Two concurrent `generate_image` calls never run the pipeline at the same time.
- **AS-10** `list_aspect_ratios` equals the backend table, and the backend table keys equal the tool schema's ratio `Literal`.
- **AS-11** Tool registration lists `generate_image`, `edit_image`, `list_aspect_ratios`, `get_model_info`, `get_image_base64` and `get_latest_image_id`. Prompts list `photorealistic`, `transparent_asset` and `extract_subject`, and do not list `negative_defaults`.
- **AS-12** `IMAGE_MODEL=qwen_image_2512` fails settings validation.
- **AS-13** `IMAGE_GPU_MEMORY_LIMIT=30` from `Settings` reaches `from_pretrained` as `max_memory[0] == "30GiB"` with `device_map="balanced"`.

GPU smoke test (manual, after the operator gives the go-ahead, C-4):
- **AS-14** The container builds and becomes healthy. The log shows a single model load.
- **AS-15** A 1:1 generate, a transparent generate, and an edit that uses the first result's ID all succeed. The outputs look right and the transparent PNG has real alpha. Record the timings and memory use.

## Out of scope

- Painted, circled or mask-based local edits as separate parameters. These go in as ordinary reference images, described in the prompt. The pipeline has no mask argument.
- Per-step progress streaming (the pipeline's `callback_on_step_end`). Start and end progress stay as they are.
- LoRA loading, batching (`num_images_per_prompt` > 1), and model CPU offload modes other than the `max_memory` cap.
- Keeping 2512 as a fallback.

## Traceability

Tests: `M` = `tests/test_models.py`, `R` = `tests/test_references.py`, `T` = `tests/test_tools.py`. AS-14 and AS-15 (GPU) were run on 2026-10-18. See the results below.

| Requirement | Implementation | Test | Status |
|-------------|----------------|------|--------|
| FR-005-01 | `QwenImage21.load` (`QwenImage21Pipeline`, `device_map="balanced"`, `max_memory`, CPU branch, no `.to()`) | M `test_load_cpu_has_no_device_map`, `test_load_cuda_capped`, `test_load_cuda_uncapped`, `test_load_never_calls_to` | ✅ (real load ⏳ AS-14) |
| FR-005-02 | `models/qwen_image_2512.py` and `negative_defaults` deleted; `Settings.image_model` Literal | T `test_settings_reject_old_model`, `test_prompt_registration` | ✅ |
| FR-005-03 | Properties on `BaseImageModel` / `QwenImage21`; `server.py` reads them | T `test_list_aspect_ratios_equals_backend`, `test_get_model_info` | ✅ |
| FR-005-04 | `QwenImage21.ASPECT_RATIOS` | M `test_aspect_ratios_table_and_copy` | ✅ |
| FR-005-05 | `default_inference_steps`, `default_cfg_scale`, `default_negative_prompt` | M `test_model_name_and_properties` | ✅ |
| FR-005-06 | `_resolve_params` | T `test_resolve_defaults`, `test_resolve_env_overrides` | ✅ |
| FR-005-07 | `_resolve_params` clamps | T `test_resolve_clamps`, `test_generate_clamps` | ✅ |
| FR-005-08 | `_resolve_params` warnings, `ctx.warning` | T `test_resolve_negative_warning_and_empty_string`, `test_generate_negative_prompt_warns_but_succeeds` | ✅ |
| FR-005-09 | Seed resolution in both tools; `use_kv_cache` never passed | M `test_generate_t2i_kwargs` (determinism ⏳ AS-15) | ✅ / ⏳ |
| FR-005-10 | `generate_image` | T `test_generate_defaults_and_metadata`, `test_generate_transparent_16x9` | ✅ |
| FR-005-11 | `edit_image` | T `test_edit_mixed_references`, `test_edit_rejected_when_backend_cannot_edit` | ✅ |
| FR-005-12 | `storage/references.py` `resolve_references`, `decode_base64_image` | R `test_resolve_order_and_ids`, `test_decode_allowed_formats`, `test_decode_data_uri_prefix_and_whitespace` | ✅ |
| FR-005-13 | `resolve_references` (runs before the lock and the model); pixel limit checked explicitly (also in Pillow's warn-only 1x–2x band) | R `test_resolve_*`, `test_decode_*` incl. `test_decode_between_1x_and_2x_pixel_limit`; T `test_edit_bad_references_never_call_model`, `test_edit_gif_and_oversize_rejected`. The AS-7 "25 MB" case is exercised with tiny limits (`MAX_REFERENCE_IMAGE_MB=0` / 100 bytes), not a real 25 MB file | ✅ |
| FR-005-14 | `edit_image` size logic; backend `output_resolution=2048` | M `test_generate_edit_kwargs`; T `test_edit_with_aspect_ratio_override` (visual ⏳ AS-15) | ✅ / ⏳ |
| FR-005-15 | `resolve_references` keeps PIL images and alpha; backend passes `image=list(...)` | M `test_generate_edit_kwargs`; R `test_decode_preserves_alpha`, `test_resolve_order_and_ids` | ✅ |
| FR-005-16 | `QwenImage21._wrap_transparent`, `.convert("RGBA")` | M `test_generate_transparent_wraps_prompt_and_returns_rgba`, `test_transparent_prompt_keeps_terminal_punctuation` | ✅ |
| FR-005-17 | `.convert("RGB")` in `QwenImage21.generate` | M `test_generate_rgb_conversion`; T `test_generate_defaults_and_metadata` | ✅ |
| FR-005-18 | `_run_generation` metadata; `StoredImage` fields | T `test_generate_defaults_and_metadata`, `test_edit_mixed_references` | ✅ |
| FR-005-19 | `images://list`, `get_latest_image_id` | T `test_latest_id_and_list_include_mode_and_transparent` | ✅ |
| FR-005-20 | Uploads are never passed to `store.save` | T `test_edit_mixed_references` (store count) | ✅ |
| FR-005-21 | `list_aspect_ratios` | T `test_list_aspect_ratios_equals_backend` | ✅ |
| FR-005-22 | `get_model_info` | T `test_get_model_info`, `test_get_model_info_resolved_with_env` | ✅ |
| FR-005-23 | `photorealistic`, `transparent_asset`, `extract_subject` | T `test_prompt_registration`, `test_prompts_text` | ✅ |
| FR-005-24 | `Settings` fields and validator; `load_model()` | T `test_settings_defaults`, `test_settings_empty_strings_become_none`, `test_settings_memory_limit_reaches_backend` | ✅ |
| FR-005-25 | `pyproject.toml`, `docker/Dockerfile` | Container build; `from diffusers import QwenImage21Pipeline` (see research R-2) | ✅ |
| FR-005-26 | `.env.example`, Dockerfile, compose, README, AGENTS.md | Review (`grep 2512`) | ✅ |
| NFR-005-01 | `_generation_lock` in `_run_generation` | T `test_generation_calls_are_serialized` | ✅ |
| NFR-005-02 | README, AGENTS.md, `.env.example`, backend docstring and log messages | Review; measured use ⏳ AS-15 | ✅ docs / ⏳ measurement |
| NFR-005-03 | All new logic is unit-tested without a GPU | 77 tests in the Docker image | ✅ |
| NFR-005-04 | Tool docstrings | T `test_tool_docstrings_cover_contract` | ✅ |
| NFR-005-05 | Tool docstrings | T `test_tool_docstrings_cover_contract` | ✅ |
| AS-1 – AS-13 | See the rows above | M, R, T | ✅ |
| AS-14 | Container build + start | Built with diffusers 0.41.0 / transformers 5.20.1 / mcp 1.30.0. Healthy. One `Loading image model`. 32.44 GB allocated after load. | ✅ |
| AS-15 | MCP client over Streamable HTTP | generate 1:1: 2048×2048 RGB, 256.5 s ✅ · transparent: 2048×2048 RGBA, alpha 0–255, clean cut-out, 255.7 s ✅ · edit chained by `image_id`: 2048×2048 RGB, `mode=edit`, `aspect_ratio=input`, 372.1 s, layout kept and color changed, **but heavy over-sharpened/HDR-like artifacts** ⚠️ (F-1) | ⚠️ |

### Findings from the GPU smoke test

- **F-1 (edit quality)** The 2K edit output shows strong over-sharpening and saturation artifacts that the source image doesn't have. The leading hypothesis is `output_resolution=2048` (P-7). The pipeline default is 1024, which also sets the resize of the condition images. Suggested follow-up: an A/B test of the same edit at `output_resolution=1024`. If 1024 is clean, change `EDIT_OUTPUT_RESOLUTION` (or make it configurable) and revisit P-7.
- **F-2 (memory)** The measured peak (~68–76 GB) is well above the planning estimate (38–45 GB). This confirms M-1: vLLM at 0.6 (~73 GB) can't run at the same time.
