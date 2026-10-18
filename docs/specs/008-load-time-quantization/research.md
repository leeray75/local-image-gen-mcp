# Feature 008 — Research

Checked on 2026-10-18 in the project image (`local-image-gen-mcp:qwen21-test`). Versions: diffusers 0.41.0, transformers 5.20.1, torchao 0.17.0+git42bcdc49, nvidia-modelopt 0.42.0, accelerate 1.15.0, torch 2.12.0a0 (NGC 26.04).

## R-1 What's available

| API | Present | Notes |
|-----|---------|-------|
| `diffusers.PipelineQuantizationConfig` | ✅ | One config per component (`quant_mapping`), passed to `Pipeline.from_pretrained` |
| `diffusers.TorchAoConfig` | ✅ | `quant_type` is an **`AOBaseConfig` instance**, not a string (e.g. `Int8WeightOnlyConfig()`) |
| `diffusers.NVIDIAModelOptConfig` | ✅ | `quant_type` ∈ {FP8, INT8, INT4, NF4, **NVFP4**} |
| `transformers.TorchAoConfig` | ✅ | For the Qwen3-VL text encoder |
| `transformers.FineGrainedFP8Config` | ✅ | An alternative FP8 path for the text encoder (not chosen; torchao keeps one library for 8-bit) |
| torchao configs | ✅ | `Int8WeightOnlyConfig`, `Float8WeightOnlyConfig`, `Int8DynamicActivationInt8WeightConfig`, `Float8DynamicActivationFloat8WeightConfig`, `Int4WeightOnlyConfig`, … |
| torchao NVFP4 | ⚠️ prototype | `torchao.prototype.mx_formats`: `NVFP4WeightOnlyConfig`, `NVFP4DynamicActivationNVFP4WeightConfig` |

## R-2 Does ModelOpt really store 4-bit weights?

Yes, in diffusers 0.41.0. The ModelOpt quantizer (`diffusers/quantizers/modelopt/modelopt_quantizer.py`) does this for each module as it loads:

```python
set_module_tensor_to_device(model, param_name, target_device, param_value, dtype)
mtq.calibrate(module, algorithm, forward_loop)
mtq.compress(module)
module.weight.requires_grad = False
```

`mtq.compress` replaces the simulated-quantization weights with **packed low-bit storage**. Because this runs one module at a time during loading, the startup peak should stay near the quantized size, not the BF16 size. Both points need confirming on the GPU (AS-6).

With no `forward_loop`, ModelOpt uses static scales computed from the weights alone. That's appropriate for **weight-only** quantization, which is what 008 uses.

## R-3 Expected savings (estimates from published file sizes)

| Component | BF16 | 8-bit | NVFP4 |
|-----------|------|-------|-------|
| Image model | 14.2 GB | ~7.1 | ~4.2 |
| Text encoder | 17.5 GB | ~9.4 | ~7.6 (vision tower and embeddings kept BF16) |
| VAE | 0.7 | 0.7 | 0.7 |
| **Total** | **32.4** | **~17** | **~12.5** |

Expected 1K peak: the weights plus ~8 GB of activations, so **~25 GB (8-bit) and ~20 GB (NVFP4)**, against ~37 GB available next to vLLM.

## R-4 Risks to verify on the GPU

- **NVFP4 kernels on GB10 (sm_121):** the published NVFP4 builds were tested on sm_120 (RTX 5090 / RTX PRO 6000). ModelOpt and torchao kernel coverage for sm_121 is unverified.
- **torchao prototype NVFP4 in transformers:** this is a prototype API and may not load into Qwen3-VL, or may lack fast kernels. FR-008-04 makes a failure loud.
- **`device_map="balanced"` combined with quantization and `max_memory`:** with quantized weights far below the 40 GB cap, nothing should be offloaded. If offloading does happen, check that quantized tensors survive the accelerate hooks.
- **Speed:** weight-only quantization dequantizes on the fly and may be **slower** than BF16. Record the timings; speed isn't a goal of 008.
- **Quality:** text-encoder quantization affects prompt understanding and lettering (005/F-2 research), so G2 lettering accuracy and the E1 edit are the key checks.

## R-5 Verified calls and module names (T001)

Checked on 2026-10-18 in `local-image-gen-mcp:qwen21-test` (no GPU, no weights). Components were instantiated on the **meta device** from the cached configs (`accelerate.init_empty_weights()` + `QwenImage21Transformer2DModel.from_config` / `Qwen3VLForConditionalGeneration._from_config`), and the real quantizer predicates were run on those meta models. The scripts are throwaway introspection and are not committed.

### Constructor calls (all accepted by the installed versions)

| Mode | Image model (`transformer`) | Text encoder (`text_encoder`) |
|------|------------------------------|-------------------------------|
| `int8` | `diffusers.TorchAoConfig(torchao.quantization.Int8WeightOnlyConfig(), modules_to_not_convert=[...])` | `transformers.TorchAoConfig(Int8WeightOnlyConfig(), modules_to_not_convert=[...])` |
| `fp8` | `diffusers.TorchAoConfig(torchao.quantization.Float8WeightOnlyConfig(), modules_to_not_convert=[...])` | `transformers.TorchAoConfig(Float8WeightOnlyConfig(), modules_to_not_convert=[...])` |
| `nvfp4` | `diffusers.NVIDIAModelOptConfig(quant_type="NVFP4", weight_only=True, modules_to_not_convert=[...])` | `transformers.TorchAoConfig(torchao.prototype.mx_formats.NVFP4WeightOnlyConfig(), modules_to_not_convert=[...])` |

- Signatures: `diffusers.TorchAoConfig(quant_type: AOBaseConfig, modules_to_not_convert=None, **kw)`; `NVIDIAModelOptConfig(quant_type: str, modules_to_not_convert=None, weight_only=True, ..., disable_conv_quantization=False)` (`weight_only=True` is already the default; it is passed explicitly for clarity); `transformers.TorchAoConfig(quant_type, modules_to_not_convert=None, include_input_output_embeddings=False, untie_embedding_weights=False)`.
- `Int8WeightOnlyConfig(group_size=None, granularity=PerRow(dim=-1), version=1)`, `Float8WeightOnlyConfig(weight_dtype=float8_e4m3fn, version=2)`, `NVFP4WeightOnlyConfig(use_dynamic_per_tensor_scale=True)`. Default arguments are used.
- `PipelineQuantizationConfig(quant_mapping={"transformer": <diffusers cfg>, "text_encoder": <transformers cfg>})` accepts the mixed library classes.
- `NVFP4WeightOnlyConfig` imports from `torchao.prototype.mx_formats`. `transformers.TorchAoConfig` **accepts** it at construction time. Whether it loads into Qwen3-VL and has working kernels on sm_121 is still a GPU question (R-4, AS-6).
- ModelOpt NVFP4 config: `quant_cfg["*weight_quantizer"] = {"fake_quant": False, "num_bits": (2, 1), ...}` with input/output quantizers disabled (weight-only), algorithm `max`.
- Both `modules_to_not_convert` lists are **mutated in place** by the diffusers quantizers (they `extend` them), so the backend builds fresh lists on every call.

### Module structure (meta device)

Image model: `QwenImage21Transformer2DModel`, 7.12 B params. Top level: `pos_embed`, `time_text_embed` (`timestep_embedder.linear_1/2`), `txt_in` (`text_norm`, `in_layer`, `out_layer`), `img_in`, `modulation` (Sequential, one shared Linear at `modulation.1`), 32 `transformer_blocks` (`attn.to_q/k/v/to_out.0`, `attn.norm_q/norm_k`, `img_norm1/2`, `img_mlp.proj/out/gate_layer`), `norm_out` (`.linear`, `.norm`), `proj_out`. There are no per-block modulation layers.

Text encoder: `Qwen3VLForConditionalGeneration`, 8.77 B params. `model.visual` (27 vision blocks, `patch_embed`, `pos_embed`, `merger`, 3 `deepstack_merger_list`), `model.language_model` (`embed_tokens`, 36 layers with `self_attn.q/k/v/o_proj`, `mlp.gate/up/down_proj`, `q_norm`/`k_norm`/layer norms, `norm`, `rotary_emb`), and an **untied** `lm_head` (`tie_word_embeddings=False`).

### Matching rules in the installed quantizers

- diffusers torchao: a parameter is skipped if `key + "."` is a **substring** of its name (or equals it); it only quantizes `nn.Linear` weights, so norms, embeddings and convs are never touched.
- transformers torchao: `should_convert_module` skips on a regex prefix match (`key.`), a regex match, or `endswith(key)`; only `nn.Linear` weights are quantized (embeddings only with `include_input_output_embeddings=True`). **When `modules_to_not_convert` is given, the library's default skips (`lm_head` etc.) are not added**, so `lm_head` MUST be listed explicitly.
- ModelOpt: each entry becomes `quant_cfg["*<entry>*"] = {"enable": False}` (a wildcard match on module names).

### Chosen `modules_to_not_convert`

| Component | List | Effect (measured on the meta model with the real quantizer predicates) |
|-----------|------|--------------------------------------------------------------------|
| `transformer` | `img_in`, `txt_in`, `time_text_embed`, `modulation`, `norm_out`, `proj_out` | 6.979 B params converted; 0.136 B kept BF16 (input/output projections, timestep embedder, modulation, `norm_out.linear`, and the 64 non-Linear norm weights). No block layer is skipped, so there is no name clash with `*img_in*`-style wildcards in the ModelOpt config (`img_norm*`, `img_mlp*` do not match). |
| `text_encoder` | `model.visual`, `lm_head` | 6.946 B params converted (the 36 language-model layers' Linears); 1.821 B kept BF16: vision tower, `lm_head`, `embed_tokens` (an Embedding, never converted), and all RMSNorm weights. |

### Differences from the spec's mapping table

None in the mapping. Two clarifications: (1) `weight_only=True` is passed explicitly to ModelOpt; (2) `lm_head` has to be listed by name because the transformers default skips are replaced by an explicit list. Estimated BF16-kept text encoder footprint is 1.82 B params (3.6 GB), so the NVFP4 text-encoder estimate in R-3 (~7.6 GB) is about right.

## R-6 Staged loading and device placement (T007b, F-6 mitigation)

Checked against the installed diffusers 0.41.0 / transformers 5.20.1 sources (no weights loaded).

- `QwenImage21Transformer2DModel.from_pretrained(MODEL_ID, subfolder="transformer", quantization_config=<diffusers cfg>, dtype=..., device_map=...)` and `Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, subfolder="text_encoder", quantization_config=<transformers cfg>, dtype=..., device_map=...)` are the per-component loaders. diffusers models accept `dtype` (and the older `torch_dtype`); transformers 5 prefers `dtype`, so `dtype=` is used for all three calls.
- `QwenImage21Pipeline.from_pretrained(MODEL_ID, transformer=<loaded>, text_encoder=<loaded>, ...)` uses passed modules as they are (`passed_class_obj`) and loads only the rest (VAE, processor, scheduler). The per-component quantization config is **not** repeated at pipeline level.
- **Placement decision.** Pipeline-level `device_map="balanced"` is only defined for the whole pipeline (`_get_final_device_map` sizes every component on the meta device and splits them over GPUs and `max_memory`); it does not apply to a single model. In the staged path:
  - each of the transformer and text encoder loads with `device_map={"": 0}` (the whole model on GPU 0, loaded module by module by accelerate, so no `.to()` is ever needed);
  - the final pipeline call uses `device_map="cuda"` (a supported string in diffusers' `SUPPORTED_DEVICE_MAP`) for the remaining VAE, instead of `"balanced"`, so a balanced map cannot re-place or offload the already loaded components;
  - `IMAGE_GPU_MEMORY_LIMIT` / `max_memory` is **not applied** when quantizing (logged as a warning). Quantized weights are 12 to 17 GB, far below the 40 GB cap, and on unified memory the cap never reduced total RAM use anyway.
  - `none`/`none` keeps the single `from_pretrained(device_map="balanced", max_memory=...)` call, unchanged.
- Between components the backend runs `gc.collect()` and `torch.cuda.empty_cache()`, and logs CUDA `memory_allocated` / `memory_reserved` plus host `MemAvailable` (from `/proc/meminfo`) after each step, to pin down the F-6 cause on the next GPU run.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set in compose (FR-008-11).
- Not verifiable without the GPU: whether staging actually lowers the load-time peak, and whether `device_map="cuda"` on the final pipeline call with pre-loaded modules behaves the same on the real pipeline as the source reading suggests.

## R-7 GPU results (T009, 2026-10-18, alongside vLLM at `gpu-memory-utilization 0.6`)

All runs were made with the vLLM Qwen3.8-27B engine **running**, under a watchdog that kills the image server if host `MemAvailable` drops below 8 GB. Host available memory before each load was ~37–43 GB. 1K = the 007 defaults. G1 is a photo, G2 typography, G3 a transparent sticker (seed 42 / 7, 40 steps). E1 is the 006 recolor edit at 1024, with the 006 source sent as base64.

| Config | Image model | Text encoder | VAE | **Total allocated** | Load | Min available, load / gen | G1 / G2 / G3 / E1 | Quality |
|--------|-----------|--------------|-----|---------------------|------|----------------------------|-------------------|---------|
| BF16 (005/006, vLLM stopped) | 14.23 | 17.5 | 0.68 | **32.4 GB** | ~195 s | doesn't fit next to vLLM | — / — / — / 61 s | reference |
| nvfp4 / nvfp4 (run 1) | — | — | — | — | watchdog fired at 8.2 GB, text encoder ~45% loaded | — | — | — (F-6) |
| nvfp4 / nvfp4 (run 2, staged) | **14.23 (no saving)** | — | — | — | watchdog fired at 7.98 GB, text encoder ~50% loaded | — | — | — (F-7) |
| fp8 / fp8 | 7.26 | 10.59 | 0.68 | **18.53 GB** | 191 s | 9.8 / 15.5 GB | 176 / 86 / 87 / 121 s | exact "QWEN IMAGE 2.1" lettering; sticker 60.9% transparent; E1 ≈ BF16 |
| **int8 / int8** | 7.27 | 10.60 | 0.68 | **18.54 GB** | 194 s | 10.2 / 14.6 GB | **77.5 / 62 / 62 / 71.6 s** | exact lettering; sticker 61.1% transparent; E1 ≈ BF16 |

Thumbnails are in [`ab/`](ab/): `fp8_fp8_*`, `int8_int8_*`, and `bf16_E1_006.jpg`, the BF16 reference for E1.

**Findings:**
- 8-bit weight-only quantization cuts resident weights by **43%** (32.4 → 18.5 GB) with no visible quality loss in these cases.
- INT8 runs at close to BF16 speed (62 s per 1K image; the edit took 72 s against 61 s). FP8 weight-only is 1.4–2× slower on this stack.
- **The tightest moment is loading**, when ~10 GB is still available while the text encoder streams in. Generation leaves ~15 GB.
- Starting the image server while vLLM is under heavy load could cut that margin.
- NVFP4 is not usable with the installed diffusers/ModelOpt/torchao on GB10 (F-7).
