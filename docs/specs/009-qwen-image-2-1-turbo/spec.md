# Feature 009 — Qwen-Image-2.1-Turbo (8-step) Backend

| | |
|---|---|
| **Status** | Implemented and GPU-verified alongside vLLM (2026-10-18). **Turbo is the default** (D-5). Review approved in round 3. |
| **Constitution** | I, II, III, IV, VI, VII, VIII |
| **Depends on** | [005](../005-qwen-image-2-1/spec.md), [007](../007-1k-default-resolution/spec.md), [008](../008-load-time-quantization/spec.md) |
| **Next** | [research.md](research.md) → [plan.md](plan.md) → [tasks.md](tasks.md) |

## Problem

With the 008 default (int8, running next to vLLM), a 1K image takes ~62 s and a 1024 edit ~72 s. [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) is Qwen's official accelerated checkpoint of the same model. It has the same 7B architecture and loads through the same `QwenImage21Pipeline`, but uses an **8-step** sampling schedule baked into the checkpoint, versus 40 steps. It should be **~5× faster at the same memory**. The cost is the usual distillation risk to quality and instruction-following, so it has to be measured.

## Decisions (orchestrator recommendations; the operator delegated "do whatever you think is best", 2026-10-18. This spec is the approval gate.)

| # | Decision | Choice |
|---|----------|--------|
| D-1 | How to select it | A new **`IMAGE_MODEL=qwen_image_2_1_turbo`** value: the same `QwenImage21` backend class, a different checkpoint ID, and its own step rules. This follows the existing backend-selection pattern (Constitution II). |
| D-2 | Default | Stays **`qwen_image_2_1`** until the GPU A/B test. The operator then decides whether Turbo becomes the default. |
| D-3 | Verification | A GPU A/B test of Turbo against the base model, **both int8**, alongside vLLM under the 8 GB watchdog (008's method). It uses the same G1/G2/G3 prompts and seeds and the 006 E1 edit, compared with 008's int8 results. vLLM is not stopped. |
| D-4 | Download | The first Turbo start downloads ~32 GB into the shared HF cache (text encoder 17.5, transformer 14.2, VAE 0.7). Disk has 2.9 TB free. The download happens in the image server container, alongside vLLM. |
| D-5 | Default (AS-7) | **Turbo becomes the default** (`IMAGE_MODEL=qwen_image_2_1_turbo`), chosen by the operator from the GPU A/B (2026-10-18). It was ~4.5× faster with equal quality on all four checks, at the same memory. The base model stays selectable with `IMAGE_MODEL=qwen_image_2_1`. |
| — | Standing rule | If memory pressure appears, stop and report. Don't retry or work around it (operator, 2026-10-18). |

## User stories

**US-1 (P1)** As an operator, I can switch the server to Turbo with one environment variable and get images about 5× faster, with the same memory, quantization and tools.

**US-2 (P1)** As an agent, the tools behave the same on Turbo. The docstrings and `get_model_info` tell me which checkpoint is active and that it runs a fixed 8-step schedule.

**US-3 (P2)** As an operator, I get measured evidence (speed, memory, quality, lettering, edit fidelity) to decide whether Turbo should be the default.

## Functional requirements

- **FR-009-01** `Settings.image_model` MUST accept `qwen_image_2_1_turbo`. `load_model()` constructs the Qwen-Image-2.1 backend with `MODEL_ID="Qwen/Qwen-Image-2.1-Turbo"` and the Turbo step rules. All other settings (quantization, resolutions, limits) apply unchanged.
- **FR-009-02** Step limits MUST come from the **backend**, not from the server-wide `MIN_STEPS, MAX_STEPS = 20, 100` constants:
  - The backend exposes `default_inference_steps`, `min_inference_steps` and `max_inference_steps`.
  - Base: 40 / 20 / 100 (unchanged behavior).
  - Turbo: **8 / 8 / 8**. The checkpoint's `sample_sigmas` fixes the schedule, and the pipeline ignores `num_inference_steps` when they are set ([research R-1](research.md#r-1-turbo-checkpoint-facts)).
  - The server clamps to the backend's range. A non-default `num_inference_steps` (or `MAX_INFERENCE_STEPS`) on Turbo MUST produce a warning in the client log saying Turbo uses its fixed 8-step schedule.
- **FR-009-03** The backend MUST NOT pass `sigmas` and MUST NOT override the scheduler for Turbo, so the checkpoint's saved schedule is used. CFG defaults stay at 1.0 (the model card says Turbo uses CFG = 1), and the negative-prompt behavior is unchanged.
- **FR-009-04** `get_model_info` MUST report `model_id` (the checkpoint) and the step range `min_inference_steps` / `max_inference_steps`, in addition to the existing fields. `model_name` becomes `Qwen-Image-2.1-Turbo` on Turbo.
- **FR-009-05** Tool docstrings and server instructions MUST NOT state "40 steps" as universal. They describe steps as "the backend default (40 for Qwen-Image-2.1, a fixed 8 for Turbo; see `get_model_info`)".
- **FR-009-06** README, AGENTS.md, `.env.example` and compose MUST document the new `IMAGE_MODEL` value, the ~32 GB first download, the fixed 8 steps, and the A/B results (after T-GPU). Compose already passes `IMAGE_MODEL` through. If it doesn't, add `${IMAGE_MODEL:-qwen_image_2_1}`.
- **FR-009-07** The CHANGELOG `[Unreleased]` MUST get an `Added` entry. If the default changes after the A/B test, it also gets a `Changed` entry.

## Non-functional requirements

- **NFR-009-01** No new dependencies. diffusers 0.41.0 already supports pipeline-configured `sample_sigmas` (R-1).
- **NFR-009-02** GPU-free unit tests MUST cover:
  - settings validation, including the new value
  - the factory building the Turbo backend with the right `MODEL_ID` and step rules
  - server step clamping that follows the backend range, both base and Turbo, including the Turbo warning
  - no `sigmas` and no `num_inference_steps` override on the Turbo pipeline call (only what the pipeline needs)
  - `get_model_info` fields
  - the unchanged base behavior
- **NFR-009-03** vLLM is never stopped. The Turbo GPU run uses int8 under the watchdog, and the standing rule applies.

## Acceptance scenarios

**Unit:**
- **AS-1** `IMAGE_MODEL=qwen_image_2_1_turbo` produces a backend with `MODEL_ID="Qwen/Qwen-Image-2.1-Turbo"`, `default_inference_steps == 8`, and step range [8, 8].
- **AS-2** On Turbo, `generate_image(num_inference_steps=40)` runs with 8 steps and sends a warning to the client log.
- **AS-3** On the base model, steps clamp to [20, 100] and default to 40, as before.
- **AS-4** `get_model_info` reports `model_id`, `model_name` and the step range for each backend.

**GPU (manual, alongside vLLM, under the watchdog, both int8):**
- **AS-5** Turbo loads within the watchdog limits. Record the load time, the per-step memory log, and the minimum available host memory.
- **AS-6** Turbo G1/G2/G3 (seed 42/42/7) and E1 (the 006 source as base64): record time, size, visual verdict, G2 lettering and G3 alpha. Commit thumbnails next to 008's int8 base results.
- **AS-7** The operator reviews the evidence and decides the default (D-2, resolved by D-5).

## Out of scope

- Running both checkpoints at once.
- Custom sigma grids and per-request step overrides on Turbo.
- Pre-quantizing to disk, and the vLLM KV-cache right-sizing (a separate repo; that change was blocked by permissions and is pending the operator).

## Traceability

Status after T001-T006 (unit tests run in the project image; 191 passed). GPU items are for the orchestrator (T008).

| Requirement / scenario | Implementation | Test | Status |
|---|---|---|---|
| FR-009-01 | `config/settings.py` Literal; `models/__init__.py` factory; `QwenImage21.VARIANTS` | `test_settings_accept_turbo`, `test_factory_builds_turbo_backend`, `test_turbo_loads_turbo_checkpoint` | ✅ |
| FR-009-02 | backend `min/max/default_inference_steps`; `server._resolve_params` | `test_resolve_params_*`, `test_turbo_generate_clamps_and_warns_client` | ✅ |
| FR-009-03 | no `sigmas`/scheduler override in `generate` (research R-1 addendum) | `test_turbo_generate_kwargs_use_checkpoint_schedule` | ✅ |
| FR-009-04 | `get_model_info` adds `model_id`, `min/max_inference_steps` | `test_get_model_info_reports_checkpoint_and_range` | ✅ |
| FR-009-05 | tool docstrings, server instructions, module docstring | `test_docstrings_do_not_claim_universal_40_steps` | ✅ |
| FR-009-06 | README, AGENTS.md, `.env.example`, compose (`${IMAGE_MODEL:-qwen_image_2_1_turbo}` after D-5) | `test_compose_passes_image_model_through` | ✅ (A/B results in README and research R-4) |
| FR-009-07 | CHANGELOG `Added` and `Changed` (default change) entries | n/a | ✅ |
| NFR-009-01 | no new dependencies | n/a | ✅ |
| NFR-009-02 | tests listed above | `tests/test_models.py`, `tests/test_tools.py` | ✅ |
| NFR-009-03 | no vLLM stop in the implementation or the GPU run | manual | ✅ vLLM was never stopped, the watchdog never fired, and the minimum available memory was 13.8 GB |
| AS-1 | factory + backend properties | `test_turbo_variant_properties`, `test_factory_builds_turbo_backend` | ✅ |
| AS-2 | step clamp plus client warning | `test_turbo_generate_clamps_and_warns_client` | ✅ |
| AS-3 | base range 20-100, default 40 | `test_resolve_params_base_range_unchanged`, `test_base_variant_properties`, the existing clamp tests | ✅ |
| AS-4 | `get_model_info` | `test_get_model_info_reports_checkpoint_and_range` | ✅ |
| AS-5 | GPU run (R-4) | manual | ✅ 18.54 GB; 441 s first load including the download; minimum available 13.8 GB |
| AS-6 | GPU run (R-4, `ab/`) | manual | ✅ 14.5 / 13.5 / 17.2 s; exact lettering; sticker 63.5% transparent; edit ≈ base |
| AS-7 | default changed to Turbo in `config/settings.py`, compose, Dockerfile, `.env.example` | `test_settings_defaults`, `test_factory_builds_turbo_backend_by_default`, `test_compose_passes_image_model_through` | ✅ (D-5) |
