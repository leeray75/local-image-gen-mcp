# Feature 005 — Implementation Plan

**Inputs:** [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/mcp-interface.md](contracts/mcp-interface.md)
**Branch:** `feature/qwen-image-2-1`, branched from `feature/memory-cap-retrieval-tools-sdd-specs`

## Constitution check

| Principle | How this plan complies |
|-----------|------------------------|
| II Model-agnostic | The size table, defaults, reference limit, transparency phrasing and the RGB/RGBA conversion all move onto the backend. `server.py` imports nothing model-specific. The `DEFAULT_NEGATIVE_PROMPT` deviation noted in 001 goes away. *Accepted deviation:* tool docstrings and server `instructions` name the active model and its sizes/limits, because NFR-005-05 requires the calling agent to see them; the code itself reads everything from the backend. |
| III Env-only config | `IMAGE_GPU_MEMORY_LIMIT` moves into `Settings`. `MAX_REFERENCE_IMAGE_MB` is added. |
| IV Verify the SDK | The pipeline signature and behavior were confirmed from the diffusers source ([R-3](research.md#r-3-__call__-signature-relevant-subset) to [R-5](research.md#r-5-output)). The dependency pins are verified by the container build (T001). |
| V Non-blocking | Decoding references and calling the pipeline both run in `asyncio.to_thread`, behind an `asyncio.Lock`. |
| VI Shared host | The 40 GB cap is kept. The weights are about 25 GB smaller. |
| VII GPU-free tests | The pipeline is mocked. The tests run inside the Docker image without `--gpus`. |
| VIII Self-describing tools | The docstring content is listed in NFR-005-05. |

## Files

| File | Action |
|------|--------|
| `models/base.py` | Extend `GenerationRequest`, `GenerationResult` and `BaseImageModel` (see the data model) |
| `models/qwen_image_2_1.py` | **New**: the `QwenImage21` backend |
| `models/qwen_image_2512.py` | **Delete** |
| `models/__init__.py` | The factory builds `QwenImage21(torch_dtype_str=…, memory_limit_gb=settings.image_gpu_memory_limit)` |
| `config/settings.py` | Field changes from the data model, plus a validator that turns an empty string into `None` for the optional ints/floats |
| `config/aspect_ratios.py` | **Delete**. Its table moves to the backend. |
| `storage/image_store.py` | New `StoredImage` fields and `load_image()`. Fix the docstring that claims a rebuild from disk (001 T019). |
| `storage/references.py` | **New**: `decode_base64_image()` and `resolve_references()` |
| `storage/__init__.py` | Export the new names |
| `server.py` | Parameter resolution, the lock, the `transparent` flag, `edit_image`, the discovery changes, the prompt changes, and updated server instructions |
| `tests/test_models.py` | Rewrite for `QwenImage21` |
| `tests/test_tools.py` | **Fill in** (it is empty today) |
| `tests/test_references.py` | **New** |
| `pyproject.toml`, `docker/Dockerfile` | `diffusers>=0.41.0`, `transformers>=5.17`. Remove `ENV MAX_INFERENCE_STEPS` / `ENV DEFAULT_CFG_SCALE`, because they would override the backend defaults. Set `ENV IMAGE_MODEL=qwen_image_2_1`. |
| `docker/docker-compose.yml` | `IMAGE_MODEL: qwen_image_2_1`. Remove `MAX_INFERENCE_STEPS` and `DEFAULT_CFG_SCALE`. Keep `IMAGE_GPU_MEMORY_LIMIT: "40"`. |
| `.env.example`, `README.md`, `AGENTS.md`, `CHANGELOG.md` | Docs (see T012) |
| `docs/specs/SPECS.md`, `001/*`, `004/*` | Add the 005 row, and mark what 005 supersedes |

## Design

### D-1 Backend: `QwenImage21`

```python
from diffusers import QwenImage21Pipeline   # module-level, so tests can patch models.qwen_image_2_1.QwenImage21Pipeline

class QwenImage21(BaseImageModel):
    MODEL_ID = "Qwen/Qwen-Image-2.1"
    EDIT_OUTPUT_RESOLUTION = 2048
    ASPECT_RATIOS = {"1:1": (2048, 2048), "4:3": (2400, 1792), "3:4": (1792, 2400),
                     "3:2": (2528, 1696), "2:3": (1696, 2528), "16:9": (2752, 1536), "9:16": (1536, 2752)}
    TRANSPARENT_PROMPT = ("This is an RGBA image with transparency. {prompt} "
                          "The image has alpha channel and the background is transparent.")
```

- **`load()`:** the same structure as the current 2512 `load()`:
  - CPU branch: no `device_map`.
  - CUDA branch: `device_map="balanced"`, plus `max_memory={0: f"{N}GiB", "cpu": "60GiB"}` when `memory_limit_gb` is set.
  - Same log messages, with the "~62GB" wording updated to "~33GB weights".
  - Never call `.to()`.
- **`generate(req)`:**
  1. Work out the prompt. If `req.transparent`, use `TRANSPARENT_PROMPT.format(prompt=p)`, where `p` is the prompt stripped and given a trailing `.` if it doesn't already end in `.`, `!` or `?`.
  2. Build the kwargs: `prompt`, `num_inference_steps`, `true_cfg_scale=req.cfg_scale`, `generator=torch.Generator(device=self._device).manual_seed(req.seed)`.
  3. Add `negative_prompt` **only if** it is not `None`.
  4. If `req.reference_images`, add `image=list(req.reference_images)` and `output_resolution=EDIT_OUTPUT_RESOLUTION`.
  5. If `req.width`/`height` are set, add them. Leave `use_kv_cache` at its default.
  6. Call the pipeline. Take `img = output.images[0]`, then `img = img.convert("RGBA" if req.transparent else "RGB")`.
  7. Call `empty_cache()` and return `GenerationResult(image, seed, time, width=img.width, height=img.height)`.
- **Properties:** `aspect_ratios` returns a **copy** of `ASPECT_RATIOS`. `default_inference_steps` is 40, `default_cfg_scale` is 1.0, `default_negative_prompt` is `None`, `max_reference_images` is 10, and `supports_transparency` is `True`.
- **`generate()` with `transparent=True` on a backend where `supports_transparency` is false:** the server rejects the request first (generic check), so the backend never sees it.

### D-2 Parameter resolution (server)

A pure helper, `_resolve_params(model, num_inference_steps, cfg_scale, negative_prompt) -> (steps, cfg, neg, warnings)`:

```
steps = num_inference_steps ?? settings.max_inference_steps ?? model.default_inference_steps  → clamp [20, 100]
cfg   = cfg_scale           ?? settings.default_cfg_scale   ?? model.default_cfg_scale        → clamp [1.0, 10.0]
neg   = negative_prompt     ?? model.default_negative_prompt
if neg and cfg <= 1.0: warnings.append("negative_prompt is ignored unless cfg_scale > 1 …")
```

An empty string for `negative_prompt` is treated as `None`. The warnings are sent with `await ctx.warning(...)`. Because the helper is pure, it can be unit-tested directly (AS-3, AS-4).

### D-3 Reference resolution (`storage/references.py`)

```python
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)

def resolve_references(entries, store, max_images, max_bytes) -> tuple[list[PILImage], list[str]]:
    # returns (images in caller order, store ids used in order)
```

- **ID vs base64:** only the canonical hyphenated 36-character UUID form counts as an ID. A bare 32-hex string is valid base64 and must not be mistaken for an ID.
- **Count:** check `1 ≤ len(entries) ≤ max_images` first.
- **Errors:** every error is a `ValueError(f"reference {i}: …")`, with `i` 1-based. A store `KeyError` or `FileNotFoundError` is re-raised as a `ValueError` with the position added.
- **Size:** `max_bytes = settings.max_reference_image_mb * 1024 * 1024`.
- **Formats:** `{"PNG", "JPEG", "WEBP"}`. Call `img.load()` inside a try block so that truncated or corrupt data fails here, not inside the pipeline.
- **Execution:** call it with `await asyncio.to_thread(resolve_references, …)` **before** taking the generation lock, so bad input never waits behind a running generation, and never reaches the model (AS-7).

### D-4 Tools

Both tools share a private coroutine, `_run_generation(ctx, model, request, metadata) -> Image`:
- report progress 0/N and send an info log
- `async with _generation_lock: result = await asyncio.to_thread(model.generate, request)`
- report progress N/N
- `store.save(result.image, metadata | {"width": result.width, "height": result.height, …})`
- send an info log with the ID
- PNG-encode the image and return `Image(data=…, format="png")`

`generate_image(prompt, aspect_ratio="1:1", transparent=False, negative_prompt=None, num_inference_steps=None, cfg_scale=None, seed=None, ctx=None)`:
- Ratio → size comes from `model.aspect_ratios[aspect_ratio]`. Raise a `ValueError` if the backend lacks that key.
- Metadata: `mode="generate"`, `reference_image_ids=[]`, `reference_count=0`.

`edit_image(prompt, images: list[str], aspect_ratio: Ratio | None = None, transparent=False, negative_prompt=None, num_inference_steps=None, cfg_scale=None, seed=None, ctx=None)`:
- Reject if `model.max_reference_images == 0`.
- Resolve the references (D-3).
- Size: `aspect_ratio` set → the table size; otherwise `width = height = None`.
- Metadata: `mode="edit"`, `aspect_ratio=aspect_ratio or "input"`, `reference_image_ids`, `reference_count=len(images)`.

Both tools:
- reject `transparent=True` when `not model.supports_transparency`
- resolve the seed as in 001
- put `transparent` in the metadata

`_generation_lock = asyncio.Lock()` lives at module level, next to `_model`.

`Ratio = Literal["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]` is defined once and used by both tools.

### D-5 Discovery, retrieval, prompts

- **`list_aspect_ratios`** comes from `get_model().aspect_ratios`.
- **`get_model_info`** adds the fields from the contract. `supported_aspect_ratios` is the sorted keys of the backend table. `default_*` are the resolved values (D-2 with no tool args).
- **`images://list` and `get_latest_image_id`** add `mode` and `transparent`.
- **Prompts:**
  - Delete `negative_defaults`.
  - `photorealistic` loses "Qwen-Image-2512" from its docstring.
  - Add `transparent_asset(subject, style="")`, which returns e.g. `"{subject}, a single standalone object, centered, clean edges, no background elements{, style}."`, and whose docstring says to use it with `generate_image(transparent=true)`.
  - Add `extract_subject(subject="the main subject")`, which returns `"Extract {subject} from Picture 1 and place it on a fully transparent background. Keep its edges, colors and details exactly; remove everything else."`, and whose docstring says to use it with `edit_image(images=[photo], transparent=true)`.
- **Server `instructions`:** mention `edit_image`, chaining by `image_id`, and `transparent`.

### D-6 Settings

```python
image_model: Literal["qwen_image_2_1"] = "qwen_image_2_1"
max_inference_steps: int | None = None
default_cfg_scale: float | None = None
image_gpu_memory_limit: int | None = None
max_reference_image_mb: int = 20

@field_validator("max_inference_steps", "default_cfg_scale", "image_gpu_memory_limit", mode="before")
def _empty_to_none(cls, v): return None if v == "" else v
```

Keep the existing `class Config` style. Don't migrate to `model_config`: that would be scope creep.

## Test plan

Tests run inside the Docker image (C-4). They are all GPU-free. Use `tmp_path` for `IMAGE_OUTPUT_DIR`. Set `os.environ["IMAGE_OUTPUT_DIR"]` in a `tests/conftest.py` **before** `storage` or `server` is imported, and monkeypatch `server.store` to a fresh `ImageStore(tmp_path)` for each test.

| File | Covers |
|------|--------|
| `test_models.py` | `QwenImage21` lifecycle; load kwargs on CPU, CUDA-capped and CUDA-uncapped (patch `torch.cuda.is_available`) (AS-13); generate kwargs for t2i (AS-1), transparent prompt wrapping and RGBA output (AS-2), `negative_prompt` omitted when `None`, edit kwargs (`image` order, `output_resolution`, no width/height) (AS-5, AS-6); RGB conversion; generate before load raises |
| `test_references.py` | UUID vs base64 detection, including a 32-hex string treated as base64; data-URI prefix (AS-8); the count limits; unknown ID; invalid base64; GIF rejected; oversize; truncated image; order preserved (AS-7) |
| `test_tools.py` | Registration of tools and prompts (AS-11); ratio `Literal` == backend keys (AS-10); `_resolve_params` clamps and env overrides (AS-3, AS-4); `generate_image` and `edit_image` end to end with a fake `BaseImageModel` set as `server._model` and a mock `ctx` (`AsyncMock`), checking the stored metadata (AS-1, AS-5); bad references never call `model.generate` (AS-7); the lock serializes overlapping calls (AS-9, using a fake whose `generate` sleeps and records overlap); settings reject `qwen_image_2512` (AS-12); `get_latest_image_id` and `images://list` include `mode` and `transparent` |

**Test command:**

```bash
docker build -f docker/Dockerfile -t local-image-gen-mcp:qwen21-test .
docker run --rm --entrypoint bash -v "$PWD":/app -w /app local-image-gen-mcp:qwen21-test \
  -c 'pip install -q "pytest>=8" "pytest-asyncio>=0.25" && IMAGE_OUTPUT_DIR=/tmp/out python -m pytest tests -v'
```

Use a separate tag, so the operator's `docker-local-image-gen-mcp` image isn't overwritten. Don't pass `--gpus`, don't start compose, and never download model weights.

## Risks

| Risk | Mitigation |
|------|------------|
| transformers 5.x conflicts with the NGC image's preinstalled packages | T001 runs the build first. If pip reports conflicts, fix them with explicit pins and record the pins in research.md. |
| The transparency phrasing is only documented for t2i; wrapping an edit instruction might work less well | Validate in AS-15. If needed, a follow-up can use a different phrasing for edits. The `extract_subject` template already asks for a transparent background explicitly. |
| A 2K edit with 10 references needs a lot of memory | NFR-005-02 is checked by hand. The cap limits weights, not activations. Document this in the `edit_image` docstring. |
| Removing `config/aspect_ratios.py` breaks something that imports it | `grep -r aspect_ratios` and update every import |
| `@mcp.tool()` changes the function, so tests can't call it directly | In mcp 1.27 the decorator returns the original function. Confirm in the test, and fall back to `mcp.call_tool` if not. |
