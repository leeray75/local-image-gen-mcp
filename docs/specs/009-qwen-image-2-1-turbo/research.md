# Feature 009 — Research

## R-1 Turbo checkpoint facts

Checked 2026-10-18 via the Hugging Face Hub (`Qwen/Qwen-Image-2.1-Turbo`) and the diffusers `v0.41.0` source.

| Fact | Value |
|------|-------|
| Publisher | **Qwen** (official). 970 likes, Qwen Research License (the same as the base model; the operator confirmed personal use). |
| Pipeline | `QwenImage21Pipeline` (`model_index.json` `_class_name`), with the same components as the base: Qwen3-VL text encoder, `QwenImage21Transformer2DModel` (7.1 B), `AutoencoderKLQwenImage21`, `FlowMatchEulerDiscreteScheduler` |
| Schedule | `model_index.json` carries **`sample_sigmas`** with 8 values: `[1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]`, so **8 denoising steps** |
| diffusers support | The `v0.41.0` pipeline documents `sample_sigmas` as "default sampling sigmas configured by the model author … their length determines the number of denoising steps". `num_inference_steps` is ignored when the configured `sample_sigmas` are used. The model card's requirement (PR #14950) is in 0.41.0. |
| Guidance | The model card says "Generation uses CFG=1 by default", the same as our default |
| KV cache | `use_kv_cache=True`, the same default as the base pipeline |
| Model-card examples | Text-to-image at 1696×2528 and editing at 2048×2048. We stay at 1K (007) for memory and for edit quality (006 F-1). Turbo at 2K is not tested here. |
| Files | `text_encoder/model.safetensors` 17.53 GB (one file); `transformer/` 2 shards, 14.23 GB; `vae/` 0.68 GB; tokenizer and processor. **About 32.4 GB in total, with no files shared with the base checkpoint**, so it's a separate download. |

### R-1 addendum: `num_inference_steps` on Turbo (T001)

Read from `diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py` (v0.41.0, inside the test image):

- `__call__(…, num_inference_steps=40, sigmas=None, …)` (line ~519). The docstring says `num_inference_steps` is "Ignored when `sigmas` or the pipeline's configured `sample_sigmas` is used".
- Lines ~732-735: `if sigmas is None: sigmas = self.config.sample_sigmas; if sigmas is None: sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)`. Then `retrieve_timesteps(scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)` takes the `sigmas` branch and sets `num_inference_steps = len(timesteps)`. The progress bar and loop use that length.

So on Turbo the checkpoint's 8 `sample_sigmas` always win and the argument has no effect. **Decision: the backend still passes `num_inference_steps` (always 8 on Turbo, because the server clamps to the backend range 8-8), and never passes `sigmas`, `timesteps` or a scheduler.** Reasons: one code path for both variants (base behavior unchanged), the request and logs carry the real step count, and if a checkpoint ever lacked `sample_sigmas` the pipeline would fall back to an 8-step linspace rather than the default 40. Omitting the argument would silently run 40 steps in that case.

## R-2 Memory expectation

The architecture and parameter counts are the same as the base, so int8 weights are about **18.5 GB** (008 R-7) and the load-time margin is the same (~10 GB available at its lowest next to vLLM at 0.6). Activations per step are the same, and fewer steps don't raise the peak. **The expected peak is the same as base int8.**

## R-3 Speed expectation

The base int8 1K generation took 62 s for 40 steps. Assuming ~1.2 s per step plus fixed overhead (text encoding, VAE decode, PNG), **8 steps should take roughly 12–18 s**. AS-6 measures it.

## R-4 GPU A/B results (T008, 2026-10-18, alongside vLLM at `gpu-memory-utilization 0.54`, int8)

The base int8 server was stopped for the comparison; vLLM stayed up. Turbo was started with `IMAGE_MODEL=qwen_image_2_1_turbo` and the default `auto` quantization (resolved to int8/int8), under the 8 GB host-memory watchdog. The prompts and seeds are the same as 008 R-7. E1 is the 006 recolor edit at 1024, with the 006 source image sent as base64.

| | Base int8 (008 R-7, vLLM at 0.6) | **Turbo int8** |
|---|---|---|
| Weights allocated | 18.54 GB (7.27 / 10.60 / 0.68) | 18.54 GB (7.27 / 10.60 / 0.68) |
| Load | 194 s | 441 s first start, including the ~32 GB download (later starts should match base) |
| Min host available, load / generation | 10.2 / 14.6 GB | **13.8 / 17.7 GB** (also helped by vLLM 0.6 → 0.54) |
| G1 photo | 77.5 s (first) | **29.2 s** (first, includes warm-up) |
| G2 typography | 62 s | **14.5 s** |
| G3 transparent sticker | 62 s | **13.5 s** |
| E1 edit 1024 | 71.6 s | **17.2 s** |
| Quality | exact lettering, sticker 61.1% transparent, edit ≈ BF16 | **exact "QWEN IMAGE 2.1" lettering; sticker 63.5% transparent (alpha 0–255); edit nearly identical to base; G1 photorealistic with a different but equally good composition** |

The thumbnails are in [`ab/`](ab/) (`turbo_int8_*`). The base int8 equivalents are in [`../008-load-time-quantization/ab/`](../008-load-time-quantization/ab/) (`int8_int8_*`).

**Result:** Turbo is about **4.3–4.6× faster** than base at the same memory, with no visible quality loss on these checks. The operator made Turbo the default (spec D-5). The watchdog never fired, and vLLM stayed healthy throughout. The new 32 GB of Turbo files in the HF cache sit alongside the base checkpoint (31 GB), for 62 GB of blobs in total.
