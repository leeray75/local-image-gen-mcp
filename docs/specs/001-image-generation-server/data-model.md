# Feature 001 — Data Model

All entities are in-process Python objects. Nothing uses a database. The only persistent artifact is the PNG file on disk.

## GenerationRequest  (`models/base.py`)

Backend-neutral input to `BaseImageModel.generate()`. The server builds it after resolving defaults, so every field is concrete: no `None`s, and no aspect-ratio strings.

| Field | Type | Invariant |
|-------|------|-----------|
| `prompt` | `str` | non-empty |
| `negative_prompt` | `str` | may be empty |
| `width` | `int` | from the aspect-ratio table |
| `height` | `int` | from the aspect-ratio table |
| `num_inference_steps` | `int` | 20 ≤ n ≤ 100 after clamping (FR-001-03) |
| `cfg_scale` | `float` | 1.0 ≤ c ≤ 10.0 (FR-001-04) |
| `seed` | `int` | 0 ≤ s < 2³², already resolved |

## GenerationResult  (`models/base.py`)

| Field | Type | Notes |
|-------|------|-------|
| `image` | `PIL.Image.Image` | |
| `seed` | `int` | Echoes the request seed |
| `generation_time` | `float` | Seconds, rounded to 2 decimal places |

## StoredImage  (`storage/image_store.py`)

One entry in the image index.

| Field | Type | Source |
|-------|------|--------|
| `image_id` | `str` | `uuid4()` assigned on save; primary key |
| `prompt` | `str` | request |
| `negative_prompt` | `str` | request |
| `aspect_ratio` | `str` | tool argument (e.g. `"16:9"`) |
| `width`, `height` | `int` | resolved dimensions |
| `seed` | `int` | resolved seed |
| `generation_time` | `float` | from the result |
| `model_name` | `str` | `BaseImageModel.model_name` |
| `disk_path` | `Path` | `IMAGE_OUTPUT_DIR / f"{image_id}.png"` |

**Lifecycle:**
1. `save()` writes the PNG first, then inserts the index entry while holding the lock.
2. Entries are never updated or deleted.
3. The index is lost when the process exits. PNG files remain on the `image-outputs` volume but are orphaned from the index.

**Ordering:** `list_all()` returns entries newest first (dict insertion order, reversed).

## ImageStore

A process-wide singleton, `storage.store`, created at import time. On init it creates `IMAGE_OUTPUT_DIR` with `mkdir(parents=True, exist_ok=True)`, so importing `storage` (and therefore `server`) needs a writable output directory.

| Operation | Behavior | Errors |
|-----------|----------|--------|
| `save(image, metadata)` | write PNG, index it, return `StoredImage` | I/O errors propagate |
| `get(id)` | look up under the lock | `KeyError` (not found; lifetime-scoped) |
| `get_image_bytes(id)` | `get()`, then read the file | `KeyError`; `FileNotFoundError` if the file was deleted |
| `list_all()` | snapshot, newest first | — |
| `count()` | number of entries | — |

## Settings  (`config/settings.py`)

A `pydantic-settings` singleton, `config.settings.settings`. Environment variable names match the field names, case-insensitive. It also reads `.env`.

| Field | Env var | Type | Default |
|-------|---------|------|---------|
| `mcp_host` | `MCP_HOST` | `str` | `0.0.0.0` |
| `mcp_port` | `MCP_PORT` | `int` | `8500` |
| `mcp_path` | `MCP_PATH` | `str` | `/mcp` |
| `image_model` | `IMAGE_MODEL` | `Literal["qwen_image_2512"]` | `qwen_image_2512` |
| `image_output_dir` | `IMAGE_OUTPUT_DIR` | `str` | `/outputs` |
| `max_inference_steps` | `MAX_INFERENCE_STEPS` | `int` | `50` |
| `default_cfg_scale` | `DEFAULT_CFG_SCALE` | `float` | `4.0` |
| `torch_dtype` | `TORCH_DTYPE` | `Literal["bfloat16","float32"]` | `bfloat16` |
| `hf_token` | `HF_TOKEN` | `str \| None` | `None` |

> `max_inference_steps` and `default_cfg_scale` are reported by `get_model_info`, but the `generate_image` signature hard-codes the defaults `50` / `4.0` instead of reading these settings. Keep them in sync, or wire the settings through.

## Aspect-ratio table  (`config/aspect_ratios.py`)

`ASPECT_RATIOS: dict[str, tuple[int, int]]`. See [research R-5](research.md#r-5-qwen-image-2512-pipeline-parameters) for the values. The `Literal[...]` type on `generate_image.aspect_ratio` duplicates the keys and **must be kept in sync**.
