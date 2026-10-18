# Feature 008 — Load-Time Weight Quantization

| | |
|---|---|
| **Status** | Implemented and GPU-verified alongside vLLM (2026-10-18). Default `auto` resolves to int8/int8 on CUDA (D-10, D-11). F-2 resolved. Review approved in round 5. |
| **Constitution** | I, II, III, IV, VI, VII |
| **Depends on** | [005](../005-qwen-image-2-1/spec.md), [007](../007-1k-default-resolution/spec.md) |
| **Addresses** | 005 F-2: the image server can't run next to vLLM. |
| **Next** | [research.md](research.md) → [plan.md](plan.md) → [tasks.md](tasks.md) |

## Problem

The BF16 weights take 32.4 GB: text encoder 17.5 GB, image model (DiT) 14.2 GB, VAE 0.7 GB. With vLLM at `gpu-memory-utilization 0.6`, about 37 GB is available on the host. An estimated ~40 GB peak for 1K generation in BF16 (007) leaves no safe margin.

Quantizing the weights should bring the 1K peak to about 20–25 GB, which would let both run together.

## Decisions (operator, 2026-10-18)

| # | Decision | Choice |
|---|----------|--------|
| D-1 | Approaches | Build **both 8-bit (torchao INT8 and FP8) and NVFP4 (NVIDIA ModelOpt)**, measure them on the GPU, and **pick the default from the data**. Quantize the **official Qwen weights at load time**; no third-party weights are downloaded. |
| D-2 | Components | Quantize the **image model and the text encoder separately**, each with its own setting. |
| D-3 | GPU verification | Run **alongside vLLM, guarded**. A memory watchdog kills the image server and the run is reported if available host memory drops below 8 GB. vLLM is never stopped. There is no BF16 1K baseline, because it doesn't fit. Quality is judged on its own terms and against existing BF16 evidence (the 006 A/B edits at 1024). |
| D-4 | Default | **Off (BF16) until measured.** The operator flips the default only after the GPU results show acceptable quality and real savings. |
| D-5 | After F-6 | **Mitigations first, then pre-quantize.** Phase A: load-time mitigations (FR-008-10 to 12), tested alongside vLLM under the watchdog. Phase B, **only if** Phase A still trips the watchdog: quantize once in a short vLLM-down window, save the quantized components to disk, and load those from then on (FR-008-13). (Operator, 2026-10-18.) |
| D-6 | vLLM window | vLLM may be stopped for **up to ~30 min, with notice** to the operator right before, then restarted and health-checked (operator, 2026-10-18) |
| D-7 | Modes | Keep **all four** configurations from AS-7 (operator, 2026-10-18) |
| D-8 | After F-7 | **Drop NVFP4 from the GPU matrix and test 8-bit now**: `fp8`/`fp8`, then `int8`/`int8`, alongside vLLM under the watchdog (operator, 2026-10-18) |
| D-9 | `nvfp4` option | **Keep it, documented as unsupported on GB10** with the current library versions. Selecting it logs a startup warning that it does not reduce memory there (FR-008-14) (operator, 2026-10-18). |
| D-10 | Default (AS-8) | **`int8` / `int8`**, chosen by the operator from the GPU evidence (2026-10-18). Same memory as fp8, about BF16 speed, exact lettering, and an edit nearly identical to BF16. |
| D-11 | CPU behavior | The default value becomes **`auto`**: `int8` when CUDA is available, `none` on CPU, with a log line naming the resolved mode. Explicitly setting `int8`, `fp8` or `nvfp4` on CPU still fails (FR-008-03) (operator, 2026-10-18). |
| — | Standing rule | If anything fails because of memory pressure, **stop and report**. Don't retry or work around it (operator, 2026-10-18). |

## Findings

- **F-6 (GPU run 1, 2026-10-18).** `nvfp4`/`nvfp4` tripped the watchdog **during loading**, and vLLM was unaffected.
  - Available memory was ~37 GB before the load.
  - It fell to 24 GB after the image model loaded (ModelOpt NVFP4, 2 min 47 s). That's ~13 GB consumed, against ~4.2 GB expected once quantized.
  - It fell to **8.2 GB** with the text encoder (torchao prototype NVFP4) only **45–49%** loaded, and the watchdog killed the server.
  - **Load-time memory is far above the steady state.** Likely causes, unconfirmed:
    - (a) the PyTorch CUDA caching allocator keeps the released BF16 staging buffers, and on unified memory they are host RAM;
    - (b) the transformers/torchao path stages more full-precision weights than one module at a time.

- **F-7 (GPU run 2, 2026-10-18; staged loading with per-step memory logs).** Start: CUDA allocated 0.0 GB, host available 41.4 GB.
  - After the **transformer with ModelOpt NVFP4**: CUDA **allocated 14.23 GB** (reserved 14.24), host available 27.3 GB.
    - **14.23 GB is exactly the BF16 size of the transformer**, and it is allocated tensors, not allocator cache.
    - So on diffusers 0.41 + ModelOpt 0.42 on GB10, the "compressed" NVFP4 weights still take full BF16 size, and allocator retention (F-6 cause a) is ruled out.
  - The **text encoder with the torchao prototype NVFP4** took ~19 GB of host memory by **50%** of its weights (BF16 for that half would be ~9 GB), and the watchdog fired at 7.98 GB.
  - **Conclusion:** neither NVFP4 path reduces memory with the installed libraries. vLLM was unaffected in both runs.

## User stories

**US-1 (P1)** As an operator, I can set the image model and the text encoder each to `none`, `int8`, `fp8` or `nvfp4`. The server then loads the official weights quantized, cutting resident memory.

**US-2 (P1)** As an operator, I can see in `get_model_info` and the startup logs which quantization is active, and how much memory the loaded model really uses.

**US-3 (P1)** As an operator, I get measured evidence (memory, speed, quality, text accuracy) for each mode on the GB10 with vLLM running, so I can choose a default.

**US-4 (P2)** As an operator, if a mode is unsupported or fails to load, startup fails with a clear message. It doesn't silently fall back, and it doesn't produce wrong images.

## Functional requirements

### Settings

- **FR-008-01** `Settings` MUST gain `quantize_transformer` and `quantize_text_encoder`, each a `Literal["none", "int8", "fp8", "nvfp4"]`, read from `QUANTIZE_TRANSFORMER` and `QUANTIZE_TEXT_ENCODER`. Both default to **`none`**, and an empty string means `none`. Invalid values fail at startup.

### Loading

- **FR-008-02** The backend MUST build one quantization config for the whole pipeline (`diffusers.PipelineQuantizationConfig` with `quant_mapping`), with one entry per quantized component. Components set to `none` are left out and stay BF16. The VAE is never quantized.

  | Mode | Image model (`transformer`, diffusers) | Text encoder (`text_encoder`, transformers) |
  |------|-------------------------------------|---------------------------------------------|
  | `int8` | `diffusers.TorchAoConfig(Int8WeightOnlyConfig())` | `transformers.TorchAoConfig(Int8WeightOnlyConfig())` |
  | `fp8` | `diffusers.TorchAoConfig(Float8WeightOnlyConfig())` | `transformers.TorchAoConfig(Float8WeightOnlyConfig())` |
  | `nvfp4` | `diffusers.NVIDIAModelOptConfig(quant_type="NVFP4")`, weight-only, so no calibration loop | `transformers.TorchAoConfig(NVFP4WeightOnlyConfig())` (torchao prototype) |

  The exact constructor arguments MUST be verified against the installed versions (diffusers 0.41.0, transformers 5.20.1, torchao 0.17.0, nvidia-modelopt 0.42.0). Record the verified calls in research.md (Constitution IV).
- **FR-008-03** The quantization config MUST be passed to `QwenImage21Pipeline.from_pretrained` together with the existing `device_map="balanced"` and the optional `max_memory`, on the CUDA path only. On CPU, any value other than `none` MUST fail at startup with a clear error.
- **FR-008-04** If a configured mode raises an error while loading (an unsupported kernel, a missing library, an unsupported layer), startup MUST fail. The error MUST name the component and the mode, and include the original error. The server MUST NOT fall back to BF16 silently.
- **FR-008-05** Layers the libraries can't or shouldn't quantize MUST stay BF16 through the backends' `modules_to_not_convert`. At minimum, that is the text encoder's **vision tower and the embedding and output heads**, and the image model's input and output projections, timestep and modulation layers, and norms. The exact module names are verified in research.md. This matches what both published NVFP4 conversions kept in BF16 (005/F-2 research).

### Load-time memory mitigations (D-5 Phase A, F-6)

- **FR-008-10** When any quantization is enabled, the backend MUST load the quantized components **one at a time**, before building the pipeline:
  1. the `transformer`, through `QwenImage21Transformer2DModel.from_pretrained(subfolder="transformer", quantization_config=…)`;
  2. **free memory**: `gc.collect()` then `torch.cuda.empty_cache()`;
  3. the `text_encoder`, through `Qwen3VLForConditionalGeneration.from_pretrained(subfolder="text_encoder", quantization_config=…)`;
  4. free memory again;
  5. build the pipeline with `QwenImage21Pipeline.from_pretrained(..., transformer=..., text_encoder=...)`, so it loads only the remaining components.

  Components set to `none` go through the same sequence in BF16, with no quantization config. Placement: each component loads with `device_map={"": 0}` and the final pipeline call with `device_map="cuda"`. This supersedes the pipeline-level `"balanced"` / `max_memory` on the quantized path (research R-6), and `IMAGE_GPU_MEMORY_LIMIT` isn't applied there; a warning is logged. With both modes `none`, loading stays exactly as before (AS-1).
- **FR-008-11** The container MUST set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` by default, overridable via compose (`${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}`). This reduces caching-allocator fragmentation and retention.
- **FR-008-12** After each component loads, the backend MUST log `torch.cuda.memory_allocated`, `torch.cuda.memory_reserved` and the host's available memory (from `/proc/meminfo` `MemAvailable`). That pins down F-6's cause on the next run.

### Pre-quantized components (D-5 Phase B — implement only if Phase A fails on the GPU)

- **FR-008-13** *(Contingent.)* An operator-run command, for example `python -m tools.prequantize`, quantizes the configured components from the official weights and saves them, with the library's supported serialization, to a local directory (`QUANTIZED_WEIGHTS_DIR`, a volume). When saved components that match the configured modes exist, the backend loads them instead of quantizing at startup. The command runs only in a vLLM-down window, per D-6. Its requirements get their own spec addendum before implementation, if Phase B is triggered.

### NVFP4 status (D-9)

- **FR-008-14** If either quantization setting is `nvfp4`, the backend MUST log a startup **warning**: on the installed diffusers, ModelOpt and torchao versions on GB10, NVFP4 does not reduce resident memory (F-7) and may use more than BF16 while loading. README, AGENTS.md and `.env.example` MUST say the same. The value stays accepted, so it can be re-tested when the libraries update.

### Default (D-10, D-11)

- **FR-008-15** Both settings MUST accept `auto` and default to it. `""` also means `auto`. At load, `auto` resolves to **`int8`** when CUDA is available and **`none`** otherwise. The resolved modes are logged, and `get_model_info` reports them. That value is the effective mode, with the configured value alongside if they differ. An explicit `int8`, `fp8` or `nvfp4` on CPU keeps failing at startup (FR-008-03). A deployment that wants BF16 sets `none` explicitly.

### Observability

- **FR-008-06** After loading, the backend MUST log the mode of each component and the real resident memory: `torch.cuda.memory_allocated` and `memory_reserved`, plus the per-component parameter bytes as stored.
- **FR-008-07** `get_model_info` MUST report `quantize_transformer`, `quantize_text_encoder`, and the measured per-component weight memory in GB.

### Docs

- **FR-008-08** README, AGENTS.md, `.env.example` and compose MUST document both settings, the modes, `none` as the default, the measured results table (once T-GPU is done), and the warning that `nvfp4` on the text encoder uses a torchao **prototype** API. Compose passes both through with `${VAR:-}`.
- **FR-008-09** CHANGELOG `[Unreleased]`: an `Added` entry. If the default flips after the GPU test (D-4), a `Changed` entry too.

## Non-functional requirements

- **NFR-008-01** No new dependencies; torchao and nvidia-modelopt are already in the image. No third-party weights.
- **NFR-008-02** GPU-free unit tests MUST cover:
  - settings validation and empty strings
  - the quantization config built for each combination of modes, checked by type and mapping, with library classes patched where needed
  - `none`/`none` passing no quantization config
  - CPU plus quantization failing
  - load errors being wrapped with the component and mode names
  - the `get_model_info` fields
  - compose passthrough
- **NFR-008-03** vLLM MUST NOT be stopped. Every GPU run happens under the watchdog (D-3), and the standing rule applies: on memory pressure, stop and report.

## Acceptance scenarios

**Unit:**
- **AS-1** Defaults pass no `quantization_config`, so behavior is the same as today.
- **AS-2** `QUANTIZE_TRANSFORMER=fp8 QUANTIZE_TEXT_ENCODER=int8` builds a mapping with the diffusers torchao FP8 config for `transformer` and the transformers torchao INT8 config for `text_encoder`.
- **AS-3** `QUANTIZE_TRANSFORMER=nvfp4` maps to the ModelOpt NVFP4 config.
- **AS-4** `QUANTIZE_TEXT_ENCODER=bf16`, an invalid value, fails settings validation.
- **AS-5** A simulated load error from a quantization backend makes `load()` raise with the component and the mode in the message.
- **AS-5b** With any quantization enabled, the components load in the FR-008-10 order: transformer → free memory → text encoder → free memory → pipeline with both passed in. Memory is logged after each step (FR-008-12). With `none`/`none`, the call sequence is the same as before.
- **AS-5d** With default settings, CUDA available gives `int8`/`int8` (staged load, torchao INT8 configs), and CPU gives `none`/`none` (the unchanged single-call path), each with a log line. Explicit `int8` on CPU fails.
- **AS-5c** Compose sets `PYTORCH_CUDA_ALLOC_CONF`, defaulting to `expandable_segments:True`.

**GPU (manual, alongside vLLM, under the watchdog; done by the orchestrator):**
- **AS-6** For each tested configuration, record:
  - load time
  - steady-state resident memory (allocated and reserved) and peak host RAM during load
  - for 1K generate G1/G2/G3 (photo, typography, transparent, seed 42): time and peak host RAM
  - one 1024 edit (006 E1 on the 006 source image, sent as base64) for a like-for-like comparison with the BF16 006 result
  - the minimum available host memory seen during the run
  - a visual verdict, and G2 lettering accuracy
  - thumbnails committed to `docs/specs/008-load-time-quantization/ab/`
- **AS-7** *(Amended by D-8: the NVFP4 configurations are dropped after F-7, so `fp8`/`fp8` then `int8`/`int8`.)* Run configurations in order of **lowest expected memory first**:
  1. `nvfp4` / `nvfp4`
  2. `nvfp4` / `fp8`
  3. `fp8` / `fp8`
  4. `int8` / `int8`

  If the watchdog fires on a configuration, record it and skip the configurations above it, which need more memory. Stop if it fires on the first one.
- **AS-8** Present the evidence to the operator, who picks the default per D-4. Record the decision and update the default in Settings and the docs.

## Out of scope

- Turbo (8-step) checkpoints, tiled VAE decoding, and changing the vLLM configuration. These are possible follow-ups.
- Speed optimizations through activation quantization (W8A8 / W4A4). This feature is weight-only, targeting memory.
- Pre-quantized third-party checkpoints, such as the Unsloth builds. They're held in reserve if load-time quantization disappoints.

## Traceability

Status: ✅ met. Unit tests run in the project image; 174 passed.

| Requirement | Status | Where |
|---|---|---|
| FR-008-01 settings | ✅ | `config/settings.py`; `test_quantize_settings_*` |
| FR-008-02 pipeline config and mapping | ✅ (constructors verified in R-5; int8 and fp8 loads verified on the GPU, R-7; nvfp4 loads but saves nothing, F-7) | `QwenImage21._build_quantization_config`; `test_quantization_mapping_*` |
| FR-008-03 wiring, CPU guard | ✅ (superseded for the quantized path by FR-008-10: per-component configs, see R-6) | `QwenImage21.load`; `test_staged_load_order_and_arguments`, `test_quantization_on_cpu_fails` |
| FR-008-04 loud load errors | ✅ | `test_staged_load_error_names_component_and_modes`, `test_quantization_build_error_names_modes` |
| FR-008-05 BF16 module lists | ✅ names verified (R-5); real loads keep the image model at 7.27 GB and the text encoder at 10.6 GB at 8-bit, consistent with the BF16 skips (R-7) | `TRANSFORMER_SKIP_MODULES`, `TEXT_ENCODER_SKIP_MODULES` |
| FR-008-06 load logging | ✅ code and unit test; real numbers logged on the GPU (R-7) | `_measure_weights` |
| FR-008-07 `get_model_info` | ✅ | `quantization_info`; `test_get_model_info_reports_quantization` |
| FR-008-08 docs, compose passthrough | ✅ (results table is a placeholder until T009) | README, AGENTS.md, `.env.example`, compose; `test_compose_passes_quantization_settings_through` |
| FR-008-09 CHANGELOG | ✅ `Added` (a `Changed` entry only if the default flips) | CHANGELOG.md |
| NFR-008-01 no new dependencies | ✅ | no change to `pyproject.toml` or the Dockerfile |
| NFR-008-02 GPU-free tests | ✅ | `tests/test_models.py`, `test_tools.py`, `test_deploy.py` |
| NFR-008-03 vLLM never stopped | ✅ all GPU runs ran next to vLLM; the watchdog fired twice (nvfp4, F-6/F-7) and protected vLLM; vLLM stayed healthy | |
| AS-1 | ✅ | `test_quantization_default_passes_no_config` |
| AS-2 | ✅ | `test_quantization_mapping_fp8_int8` |
| AS-3 | ✅ | `test_quantization_mapping_nvfp4_transformer_only` |
| AS-4 | ✅ | `test_quantize_settings_invalid_value_fails` |
| AS-5 | ✅ | `test_staged_load_error_names_component_and_modes` |
| FR-008-10 staged loading | ✅ code and unit tests; GPU: per-step logs settled F-6/F-7, and the 8-bit staged load peaked with ~10 GB still available (R-7) | `QwenImage21._load_staged`; `test_staged_load_order_and_arguments`, `test_staged_load_none_component_loads_bf16`; placement in R-6 |
| FR-008-11 allocator conf | ✅ | compose; `test_compose_sets_expandable_segments_default` |
| FR-008-12 per-step memory log | ✅ | `_log_memory`, `_host_available_gb`; `test_staged_load_logs_memory_after_each_step`, `test_host_available_gb_reads_meminfo` |
| FR-008-15 `auto` default | ✅ | `Settings`, `QwenImage21.__init__`/`quantization_info`; `test_quantize_settings_default_auto`, `test_get_model_info_shows_configured_only_when_different`; README results table |
| AS-5d | ✅ | `test_auto_resolves_to_int8_on_cuda_with_staged_load`, `test_auto_resolves_to_none_on_cpu_single_call`, `test_auto_with_explicit_int8_on_cpu_still_fails` |
| FR-008-14 nvfp4 warning | ✅ | `QwenImage21.load`; `test_nvfp4_logs_startup_warning`; README, AGENTS.md, `.env.example` |
| FR-008-13 pre-quantized components | not started (contingent on Phase B) | |
| AS-5b | ✅ | `test_staged_load_order_and_arguments`, `test_quantization_default_passes_no_config` (none/none unchanged) |
| AS-5c | ✅ | `test_compose_sets_expandable_segments_default` |
| AS-6 | ✅ measured for int8/int8 and fp8/fp8 (R-7, `ab/`); nvfp4 stopped by the watchdog | |
| AS-7 | ✅ run order followed; nvfp4 dropped after F-7 (D-8) | |
| AS-8 | ✅ operator chose int8/int8 (D-10), with `auto` (D-11) | |
