# Feature 005 — Data Model Changes

This document describes only what changes from the [001 data model](../001-image-generation-server/data-model.md).

## GenerationRequest  (`models/base.py`)

| Field | Type | Change |
|-------|------|--------|
| `prompt` | `str` | The user's prompt, before any wrapping. The backend applies the transparency phrasing. |
| `negative_prompt` | `str \| None` | **was `str`**. `None` means no negative prompt. |
| `width`, `height` | `int \| None` | **was `int`**. `None` (edit only) means the backend takes the size from the last reference image. |
| `num_inference_steps` | `int` | Already resolved and clamped to [20, 100] |
| `cfg_scale` | `float` | Already resolved and clamped to [1.0, 10.0] |
| `seed` | `int` | unchanged |
| `reference_images` | `list[PIL.Image.Image]` | **new**, default `[]`. An empty list means text-to-image. Order is significant. |
| `transparent` | `bool` | **new**, default `False` |

Invariant: `width`/`height` are either both set or both `None`. They can only be `None` when `reference_images` is non-empty.

## GenerationResult  (`models/base.py`)

| Field | Type | Change |
|-------|------|--------|
| `image` | `PIL.Image.Image` | mode `RGBA` if `transparent`, otherwise `RGB` |
| `seed` | `int` | unchanged |
| `generation_time` | `float` | unchanged |
| `width`, `height` | `int` | **new**: the actual output size, which matters for edits that size from the input |

## BaseImageModel  (`models/base.py`)

These members are added. They are abstract unless a default is given.

| Member | Type | 2.1 value |
|--------|------|-----------|
| `aspect_ratios` | property → `dict[str, tuple[int, int]]` | the R-6 table |
| `default_inference_steps` | property → `int` | `40` |
| `default_cfg_scale` | property → `float` | `1.0` |
| `default_negative_prompt` | property → `str \| None` | `None` |
| `max_reference_images` | property → `int`, default `0` (meaning editing is unsupported) | `10` |
| `supports_transparency` | property → `bool`, default `False` | `True` |

`supports_editing` is derived as `max_reference_images > 0`.

The existing members are unchanged: `load`, `unload`, `generate`, `is_loaded`, `model_name`.

## QwenImage21  (`models/qwen_image_2_1.py`, replaces `qwen_image_2512.py`)

- `MODEL_ID = "Qwen/Qwen-Image-2.1"`, `model_name = "Qwen-Image-2.1"`
- `__init__(torch_dtype_str="bfloat16", memory_limit_gb: int | None = None)`
- `EDIT_OUTPUT_RESOLUTION = 2048`
- `TRANSPARENT_PROMPT = "This is an RGBA image with transparency. {prompt} The image has alpha channel and the background is transparent."`

## StoredImage  (`storage/image_store.py`)

| Field | Type | Change |
|-------|------|--------|
| `negative_prompt` | `str` | `""` when there was none |
| `aspect_ratio` | `str` | a ratio key, or `"input"` for an edit sized from its reference |
| `mode` | `str` | **new**: `"generate"` \| `"edit"`, default `"generate"` |
| `transparent` | `bool` | **new**, default `False` |
| `reference_image_ids` | `list[str]` | **new**: the store IDs used as references, in caller order, default `[]` |
| `reference_count` | `int` | **new**: all references, including base64 uploads, default `0` |

`ImageStore.save(image, metadata)` keeps its signature. The new keys are read from `metadata` with the defaults above. The PNG is saved in the image's own mode, RGB or RGBA.

A new method, **`ImageStore.load_image(image_id) -> PIL.Image.Image`**, opens the stored PNG fully loaded, not lazily. It raises the same `KeyError` / `FileNotFoundError` as `get_image_bytes`.

## Reference decoding  (`storage/references.py`, new)

`decode_base64_image(data: str, max_bytes: int) -> PIL.Image.Image`:
1. Strip an optional `data:image/<fmt>;base64,` prefix and any whitespace.
2. Decode strictly (`base64.b64decode(..., validate=True)`).
3. Reject the input if it is larger than `max_bytes`.
4. Open it with Pillow, check the format is one of {PNG, JPEG, WEBP}, and call `.load()`. Pillow's `DecompressionBombError` is turned into a `ValueError`.

It raises `ValueError` with a human-readable reason. The caller adds the reference's position.

## Settings  (`config/settings.py`)

| Field | Env var | Type | Default | Change |
|-------|---------|------|---------|--------|
| `image_model` | `IMAGE_MODEL` | `Literal["qwen_image_2_1"]` | `qwen_image_2_1` | **was** `qwen_image_2512` |
| `max_inference_steps` | `MAX_INFERENCE_STEPS` | `int \| None` | `None` | **was** `50`. Overrides the backend default steps. |
| `default_cfg_scale` | `DEFAULT_CFG_SCALE` | `float \| None` | `None` | **was** `4.0`. Overrides the backend default CFG. |
| `image_gpu_memory_limit` | `IMAGE_GPU_MEMORY_LIMIT` | `int \| None` | `None` | **new in Settings**. It used to be read with `os.getenv` in the model module. |
| `max_reference_image_mb` | `MAX_REFERENCE_IMAGE_MB` | `int` | `20` | **new** |

An empty string in `IMAGE_GPU_MEMORY_LIMIT` (as compose may pass) MUST be treated as `None`.

## Aspect-ratio table

`config/aspect_ratios.py` is **removed**. The table moves to the backend (`QwenImage21.aspect_ratios`). The `aspect_ratio` `Literal` in the tool signatures keeps the same seven keys, and a test asserts they equal the backend's table keys.
