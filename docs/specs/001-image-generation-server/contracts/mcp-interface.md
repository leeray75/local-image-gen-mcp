# Feature 001 — MCP Interface Contract

Server name: `local-image-gen`. Transport: Streamable HTTP, `http://<host>:8500/mcp`, with JSON responses.

Server `instructions` (sent to clients on initialize) MUST mention `generate_image`, `list_aspect_ratios`, `get_model_info`, and the `image://{image_id}` resource.

## Tools

### `generate_image` (async)

| Param | Type | Default | Constraint |
|-------|------|---------|------------|
| `prompt` | `string` | required | English or Chinese |
| `aspect_ratio` | enum `1:1 \| 16:9 \| 9:16 \| 4:3 \| 3:4 \| 3:2 \| 2:3` | `1:1` | rejected by schema if not in the enum |
| `negative_prompt` | `string` | backend default | |
| `num_inference_steps` | `integer` | `50` | clamped to [20, 100] |
| `cfg_scale` | `number` | `4.0` | [1.0, 10.0] |
| `seed` | `integer \| null` | `null` | `null` means a random seed |

**Returns:** a single `image` content block, `mimeType: image/png`.

**Side effects:**
- progress `0/N`, then `N/N`
- an `info` log: `Generating <ratio> (<w>x<h>) image, steps=…, cfg=…, seed=…`
- an `info` log: `Image saved: id=<uuid> (<w>x<h>, seed=<s>, <t>s) — retrieve via image://<uuid>`
- a PNG written to disk

**Errors:** "model not loaded" (startup incomplete); errors from the backend pipeline.

### `list_aspect_ratios`

No params. Returns `{ "<ratio>": { "width": int, "height": int }, ... }` for every supported ratio.

### `get_model_info`

No params. Returns an object containing:

```
model_name, backend, is_loaded, device ("cuda"|"cpu"), torch_dtype,
default_inference_steps, default_cfg_scale, supported_aspect_ratios[],
images_generated, output_directory
```

When CUDA is available, it also includes `gpu_name`, `gpu_memory_total_gb`, `gpu_memory_allocated_gb`, and `gpu_memory_reserved_gb`.

## Resources

### `image://{image_id}`

Returns the stored PNG (`image/png`).

**Errors:**
- not found: the message mentions "Images are stored for the server's lifetime"
- file missing on disk: the message mentions the output directory may have been cleared

### `images://list`

Returns a JSON string: an array, newest first.

```json
[{
  "image_id": "uuid",
  "prompt": "first 100 chars…",
  "aspect_ratio": "16:9",
  "width": 1664, "height": 928,
  "seed": 123,
  "model": "Qwen-Image-2512",
  "generation_time_seconds": 241.7
}]
```

## Prompts

| Prompt | Args | Returns |
|--------|------|---------|
| `photorealistic` | `subject` (req), `setting` = "", `style` = "" | `"<subject>[, in <setting>]. Ultra-realistic photography … [<style>. ]Photojournalistic composition. No AI-generated artifacts."` |
| `negative_defaults` | — | the backend's default negative prompt |

## Environment contract

See [data-model.md → Settings](../data-model.md#settings--configsettingspy). Invalid `IMAGE_MODEL` or `TORCH_DTYPE` values MUST stop startup.

## Compatibility rules

- Adding optional params or new tools: non-breaking.
- Renaming, removing, or changing the type of a param, tool, resource URI or return shape: breaking. Requires a minor-version bump while the project is below 1.0 and a CHANGELOG entry.
