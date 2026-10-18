# Feature 005 — Research

Checked on 2026-10-18 against these sources:
- the Hugging Face repo `Qwen/Qwen-Image-2.1`: README, LICENSE, `model_index.json` and the file listing
- the diffusers source `src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py` on `main`
- PyPI

## R-1 Pipeline and components

`model_index.json`:

| Component | Class |
|-----------|-------|
| pipeline | `QwenImage21Pipeline` (diffusers) |
| transformer | `QwenImage21Transformer2DModel` (7B; 32 single-stream DiT layers) |
| vae | `AutoencoderKLQwenImage21` (16× spatial compression, RGBA latents) |
| text_encoder | `Qwen3VLForConditionalGeneration` (transformers) |
| processor | `Qwen3VLProcessor` |
| scheduler | `FlowMatchEulerDiscreteScheduler` |

Import: `from diffusers import QwenImage21Pipeline`. `DiffusionPipeline.from_pretrained` also resolves it from `model_index.json`, but import the specific class so that tests can patch it.

## R-2 Dependency versions

| Package | Requirement | Evidence |
|---------|-------------|----------|
| diffusers | **≥ 0.41.0** | 0.41.0 was released 2026-10-06, and the `v0.41.0` tag contains `pipelines/qwenimage21/`. The model card's `pip install git+…` is no longer needed. |
| transformers | **≥ 5.17** | Model card. Needed for `Qwen3VL*` classes. The latest on PyPI is 5.20.1. |
| torch | ≥ 2.4 | Model card. The NGC 26.04 build already satisfies this. |

**Result (T001, container build 2026-10-18):** the NGC `pytorch:26.04-py3` build resolves to diffusers 0.41.0, transformers 5.20.1, accelerate 1.15.0, huggingface-hub 1.33.0, torch 2.12.0a0 (NGC build kept), and `from diffusers import QwenImage21Pipeline` works. One extra pin was needed: **`mcp[cli]>=1.27.0,<2`**. Unpinned, pip picked mcp 2.3.0, which has no `mcp.server.fastmcp` (renamed to `MCPServer`); with the pin it installs mcp 1.30.0. The only remaining pip warning is `spin 0.18 requires click<8.4` vs click 8.5.0 (an NGC dev tool, unused here).

**Risk (original):** going from transformers 4.x to 5.x is a major-version jump. Only this project uses it, but `accelerate` and `huggingface-hub` may also need updating. Let the solver choose versions compatible with transformers 5.17 or later, and check this in the container build (task T001).

## R-3 `__call__` signature (relevant subset)

```python
pipe(
    prompt, image=None, negative_prompt=None, true_cfg_scale=1.0,
    height=None, width=None, num_inference_steps=40, generator=None,
    output_type="pil", output_resolution=1024, use_kv_cache=True, ...
)
```

- **Guidance off by default.** CFG is enabled only when `true_cfg_scale > 1` *and* a negative prompt is given. Otherwise the pipeline logs a warning and ignores the unused input. The docstring says "Qwen-Image 2.1 is meant to be sampled without guidance". CFG also doubles the transformer passes.
- **Size.** `height` and `width` are rounded down to multiples of 32. When they are omitted and `image` is given, they are taken from the **last** condition image's aspect ratio at `output_resolution²` area. With no image, they default to `output_resolution`.
- **`output_resolution`** also sets the size that condition images are resized to before encoding. 2048 gives the 2K output class (decision P-7).
- **`use_kv_cache`** caches condition and text K/V after the first step. Turning it on or off gives visibly different (equally valid) samples, so keep it fixed at the default `True` for determinism (FR-005-09).

## R-4 Condition (reference) images

- `image` is a PIL image or ndarray, or a flat list of them. Nested lists raise `ValueError`.
- The prompt template numbers images "Picture 1…N" in list order, so **order is meaningful**.
- RGBA references are kept: the VAE reads all four channels. Only the copy that goes to the vision encoder is composited over white. Non-RGBA inputs are converted to RGBA internally.
- The pipeline has no mask parameter. Per the model card, local edits marked by circles, painted annotations or separate masks are given as **images** and referred to in the prompt.
- The maximum is **10 reference images** (model card). The pipeline doesn't enforce it, so the server must (FR-005-13).

## R-5 Output

- The VAE decodes 4 channels, so `postprocess(..., "pil")` returns **RGBA** PIL images whether or not the prompt asked for transparency. Convert to RGB for opaque output (FR-005-17).
- The model card's recommended phrasing for transparent output: `"This is an RGBA image with transparency. <description>. The image has alpha channel and the background is transparent."`

## R-6 Supported sizes (model card)

| Ratio | W × H | Megapixels |
|-------|-------|-----------|
| 1:1 | 2048 × 2048 | 4.19 |
| 4:3 | 2400 × 1792 | 4.30 |
| 3:4 | 1792 × 2400 | 4.30 |
| 3:2 | 2528 × 1696 | 4.29 |
| 2:3 | 1696 × 2528 | 4.29 |
| 16:9 | 2752 × 1536 | 4.23 |
| 9:16 | 1536 × 2752 | 4.23 |

All of these are multiples of 32. Compared with 2512 (1328² = 1.76 MP), that is about 2.4× the pixels, so expect more activation memory and longer generation despite the smaller transformer.

## R-7 Memory

| Component | Weights on disk (BF16) |
|-----------|-------------------|
| text_encoder (Qwen3-VL) | 17.5 GB |
| transformer | 14.2 GB |
| vae | 1.35 GB |
| **Total** | **~33 GB** (2512: ~58 GB) |

**Unified memory caveat (measured 2026-10-18):** on GB10, CPU offload lands in the same physical memory, so `max_memory` changes placement but not the total footprint. The whole ~33 GB plus activations must fit in free RAM. With vLLM at 0.6 (~73 GB), only ~28 GiB was free, which is not enough. With vLLM stopped, ~106 GiB is free. Under the existing 40 GB cap (004), all the weights still fit on the device. The model card's own low-memory option is `pipe.enable_model_cpu_offload()`. We keep the `device_map="balanced"` plus `max_memory` approach instead, for consistency with 004. Measure in the GPU smoke test (AS-15).

## R-8 Sampling defaults

From the model card examples and the pipeline defaults: **40 steps, `true_cfg_scale=1.0`, no negative prompt**.

## R-9 License

Qwen Research License (released 2026-09-20): **non-commercial (research/evaluation) use only**, governed by PRC law. Redistribution requires shipping the license and the attribution notice. We don't redistribute weights; they are downloaded from HF at run time. The operator has confirmed the use is personal. The README must state the license.

## R-10 Testing without a GPU

Patch `models.qwen_image_2_1.QwenImage21Pipeline`. `from_pretrained` should return a callable mock whose `.images` is `[PIL.Image.new("RGBA", (w, h))]`. Assert on the kwargs passed to the pipeline call. Importing the backend module still needs `diffusers ≥ 0.41` and `torch` installed. The host has no Python ML environment, so tests run in the project Docker image (C-4).
