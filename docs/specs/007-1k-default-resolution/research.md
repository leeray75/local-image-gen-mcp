# Feature 007 — Research

## R-1 Size formula

The diffusers `QwenImage21Pipeline` sizes edits with `calculate_dimensions(target_area, ratio)` (pipeline source, line 149):

```python
width = math.sqrt(target_area * ratio)
height = width / ratio
width = round(width / 32) * 32
height = round(height / 32) * 32
```

`height` comes from the **unrounded** width. Applied with `target_area = R²`:

| Ratio | R = 1024 | R = 2048 (formula) | R = 2048 (model card) |
|-------|----------|--------------------|-----------------------|
| 1:1 | 1024×1024 | 2048×2048 | 2048×2048 |
| 16:9 | 1376×768 | 2720×1536 | **2752×1536** |
| 9:16 | 768×1376 | 1536×2720 | **1536×2752** |
| 4:3 | 1184×896 | 2368×1760 | **2400×1792** |
| 3:4 | 896×1184 | 1760×2368 | **1792×2400** |
| 3:2 | 1248×832 | 2496×1664 | **2528×1696** |
| 2:3 | 832×1248 | 1664×2496 | **1696×2528** |

The card's 2K sizes are slightly larger than the formula's, about 4.3 MP against 4.2 MP. **Decision:** at 2048, use the card table exactly, so opting back into 2K reproduces 005/006 behavior bit for bit. At every other resolution, use the formula. That keeps text-to-image consistent with how the pipeline sizes edits.

## R-2 Memory and time expectations at 1K

There is no direct measurement yet; AS-6 and AS-8 provide it. From 005 and 006:
- An edit at 1024 peaked at ~40 GB for the server (32.4 GB of weights plus ~8 GB of activations) and took ~61 s.
- A 2K generate peaked at ~68 GB and took ~256 s.

The 1K text-to-image path has no reference images to encode, so its peak should be **at or below ~40 GB**, and its time around 60 s. That still doesn't fit next to vLLM at 0.6 in BF16 (~28–35 GB free). With 008's quantized weights (~17 GB at 8-bit, ~12.5 GB at NVFP4) it should: about 20–25 GB.

## R-3 Settings validation in pydantic-settings

The current `Settings` uses per-field `field_validator`s, with a `mode="before"` validator turning `""` into the default. The ceiling check involves several fields, so it belongs in a `model_validator(mode="after")`, which runs after field validation and defaults. A `ValueError` raised there surfaces as a pydantic `ValidationError` when `Settings()` is created. That happens at import, so the server refuses to start, which is the required behavior.

## R-4 Prior art for text quality at 1K

The model card shows 2K typography examples. The 006 A/B tested only edits. BennyDaBall's NVFP4 validation package (005/F-2 research) used 1024×1024 for most of its comparisons and reported correct lettering at 1K, but that's a different runtime and only weak evidence. G2 in AS-6 tests it directly.
