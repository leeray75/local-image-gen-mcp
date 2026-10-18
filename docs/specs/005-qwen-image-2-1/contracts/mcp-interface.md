# Feature 005 — MCP Interface Contract (delta)

This document lists only what changes from the [001 contract](../../001-image-generation-server/contracts/mcp-interface.md) and the [003 additions](../../003-image-retrieval-tools/plan.md#contract-additions). It is a **breaking** release (pre-1.0 minor bump → **0.2.0**), for two reasons:
- defaults and output sizes change
- `negative_defaults` is removed

> The release is **0.2.0** (set in feature 006: `pyproject.toml` and the CHANGELOG `[0.2.0]` section). The `v0.2.0` tag is created at merge time.

The server `instructions` MUST also mention `edit_image` and the `transparent` flag.

`Ratio` = `1:1 | 16:9 | 9:16 | 4:3 | 3:4 | 3:2 | 2:3`

## `generate_image` (changed)

| Param | Type | Default | Notes |
|-------|------|---------|-------|
| `prompt` | string | required | |
| `aspect_ratio` | Ratio | `1:1` | Uses the 2K table (1:1 = 2048²); *superseded by 007: the table for `GENERATE_OUTPUT_RESOLUTION`, default 1024* |
| `transparent` | boolean | `false` | **new**. RGBA output with the model's transparency phrasing. |
| `negative_prompt` | string \| null | `null` | **was** the 2512 Chinese default. Only takes effect when `cfg_scale > 1`. |
| `num_inference_steps` | integer \| null | `null` | **was** 50. `null` means the env override, or else 40. Clamped to [20, 100]. |
| `cfg_scale` | number \| null | `null` | **was** 4.0. `null` means the env override, or else 1.0 (guidance off). Clamped to [1.0, 10.0]. |
| `seed` | integer \| null | `null` | unchanged |

**Returns:** one `image/png` content block, RGB or RGBA. Side effects are as in 001. The info log also states the mode and, for transparent output, that it is transparent.

## `edit_image` (new)

| Param | Type | Default | Notes |
|-------|------|---------|-------|
| `prompt` | string | required | The edit instruction. It can refer to "Picture 1…N" by position. |
| `images` | string[] (1–10) | required | An ordered list. Each entry is either a store `image_id` (a UUID from `generate_image`/`edit_image`/`images://list`) or a base64 PNG/JPEG/WebP, optionally with a `data:` URI prefix. Each one is ≤ `MAX_REFERENCE_IMAGE_MB` once decoded. |
| `aspect_ratio` | Ratio \| null | `null` | `null` means the last reference's aspect ratio at about `EDIT_OUTPUT_RESOLUTION`² area (default 1024 since feature 006; originally 2048). Otherwise that ratio's size at `EDIT_OUTPUT_RESOLUTION` (*was the 2K table; fixed by 007, F-5*). |
| `transparent` | boolean | `false` | e.g. extract a subject onto a transparent background |
| `negative_prompt`, `num_inference_steps`, `cfg_scale`, `seed` | | | as for `generate_image` |

**Returns:** one `image/png` content block. The result is stored and gets a new `image_id`, which can be used in a chained `edit_image`.

**Errors (raised before any model work):**
- `images must contain 1–10 references`
- `reference N: image '<id>' not found …`
- `reference N: invalid base64`
- `reference N: unsupported format GIF (PNG, JPEG, WebP only)`
- `reference N: decoded size X MB exceeds MAX_REFERENCE_IMAGE_MB=20`
- `reference N: image too large (decompression limit)`

## `list_aspect_ratios` (changed)

The shape is unchanged. It now returns the backend's 2K table (*superseded by 007: the table for `GENERATE_OUTPUT_RESOLUTION`, default 1024*).

## `get_model_info` (changed)

Adds:

```
max_reference_images: int, supports_editing: bool, supports_transparency: bool,
default_inference_steps: int, default_cfg_scale: float, has_default_negative_prompt: bool
```

`default_inference_steps` and `default_cfg_scale` are now the **resolved** values: the env override if set, otherwise the backend default.

## `images://list` and `get_latest_image_id` (changed)

Each entry adds `mode` (`"generate"` | `"edit"`) and `transparent` (bool). `aspect_ratio` can be `"input"`.

## Prompts (changed)

| Prompt | Args | Change |
|--------|------|--------|
| `photorealistic` | `subject`, `setting=""`, `style=""` | wording made model-neutral |
| `negative_defaults` | — | **removed** |
| `transparent_asset` | `subject`, `style=""` | **new**. A clean standalone-object prompt. Use it with `generate_image(transparent=true)`. |
| `extract_subject` | `subject="the main subject"` | **new**. An instruction to isolate the subject from Picture 1 onto a transparent background. Use it with `edit_image(transparent=true)`. |

## Environment (changed)

| Var | Default | Change |
|-----|---------|--------|
| `IMAGE_MODEL` | `qwen_image_2_1` | the only accepted value |
| `MAX_INFERENCE_STEPS` | unset | optional override |
| `DEFAULT_CFG_SCALE` | unset | optional override |
| `IMAGE_GPU_MEMORY_LIMIT` | unset (compose sets 40) | now validated by Settings |
| `MAX_REFERENCE_IMAGE_MB` | `20` | new |
