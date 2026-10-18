# Feature 001 — Implementation Plan

**Inputs:** [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/mcp-interface.md](contracts/mcp-interface.md)

## Constitution check

| Principle | How this plan complies |
|-----------|------------------------|
| II Model-agnostic | `server.py` imports only `models.load_model` and `BaseImageModel`. The one exception is `DEFAULT_NEGATIVE_PROMPT`, imported from the Qwen module and used as the tool default. *Deviation accepted for v0.1; move it to a backend property when a second backend lands.* |
| III Env-only config | Everything goes through `config.settings.settings`. `HF_TOKEN` is never logged. |
| IV Verify SDK | R-1 to R-4 confirmed against `mcp` 1.27.0. |
| V Non-blocking | `asyncio.to_thread(model.generate, …)`. |
| VII GPU-free tests | `DiffusionPipeline` is patched in unit tests. |

## Architecture

```
MCP client (Claude Code / Cline)
        │  Streamable HTTP, json_response
        ▼
server.py ── FastMCP("local-image-gen", host, port)
  │  tools:     generate_image · list_aspect_ratios · get_model_info
  │  resources: image://{id} · images://list
  │  prompts:   photorealistic · negative_defaults
  │
  ├── get_model() ──► module singleton _model  (loaded in __main__ before run())
  │                        │
  │                        ▼
  │                models/  load_model() ─► BaseImageModel
  │                                          └─ QwenImage2512 (diffusers)
  ├── store ──────► storage/ ImageStore (lock + dict index, PNGs on disk)
  └── settings ───► config/  Settings (pydantic-settings), ASPECT_RATIOS
```

**Request flow for `generate_image`:**
1. Resolve the seed. Map the ratio to (w, h). Clamp the steps.
2. `get_model()`.
3. Report progress 0/N and send an info log.
4. Build a `GenerationRequest` and run `await asyncio.to_thread(model.generate, req)`.
5. Report progress N/N.
6. `store.save(result.image, metadata)` and send an info log with the image ID.
7. Encode the PNG into a `BytesIO` and return `Image(data=bytes, format="png")`.

## Design decisions

### D-1 Package layout

The work is split into `config/`, `models/` and `storage/`. Each layer can be tested on its own, and only `models/` depends on torch or diffusers at import time.

### D-2 Model lifecycle: module singleton, not lifespan

The lifespan context runs per session in this SDK version ([R-2](research.md#r-2-lifespan-scope-in-mcpserverfastmcp-1270-found-late)). `_startup_load_model()` is called in `if __name__ == "__main__":` before `mcp.run(transport="streamable-http")`. Because uvicorn hasn't started yet, no connection can arrive before the model is ready (FR-001-14).

Importing `server` (in tests or REPL checks) does **not** load the model. `get_model()` raises `RuntimeError` until load completes (FR-001-15).

Trade-off: `unload()` is never called on shutdown. That is acceptable, because the process exit frees GPU memory.

### D-3 Host and port on the constructor

Per [R-3](research.md#r-3-fastmcprun-signature-found-late), host and port go to `FastMCP(...)` and `run()` takes only `transport`. **TODO:** pass `MCP_PATH` via the constructor's path setting.

### D-4 Backend interface and factory

The backend is an ABC with `load`, `unload`, `generate` (synchronous), `is_loaded`, and a `model_name` property. `load_model()` imports the backend module lazily inside its `if` branch, so torch/diffusers load only for the selected backend. To register a backend, add three things: the module, a branch in the factory, and the ID in the `Settings.image_model` Literal (which provides the startup validation, AS-9).

### D-5 Qwen backend specifics

- Use `true_cfg_scale` ([R-5](research.md#r-5-qwen-image-2512-pipeline-parameters)).
- Pass `device_map` to `from_pretrained`, never `.to()` ([R-6](research.md#r-6-meta-tensor-initialization-found-late)).
- Create the Generator on `self._device`.
- Call `torch.cuda.empty_cache()` after each generation to release activation buffers.
- `unload()` deletes the pipeline, then calls `empty_cache()` and `synchronize()`.

### D-6 Storage

The index is a `dict` guarded by `threading.Lock`. The PNG is written before indexing, so an indexed entry always has a file at save time. There is no rebuild from disk, by design: the catalog only lasts as long as the process (FR-001-10). Fix the misleading docstring.

### D-7 Validation boundaries

The `aspect_ratio` enum is enforced by the tool schema (`Literal`). Steps are clamped server-side. `cfg_scale` is currently passed through unchanged; add a clamp to meet FR-001-04.

## Risks

| Risk | Mitigation |
|------|------------|
| SDK behavior differs from docs/examples | Constitution IV: research and verify against the installed version |
| Concurrent generations on one pipeline (NFR-001-03) | Add an `asyncio.Lock` around the `to_thread` call; until then, document that callers should not generate in parallel |
| `Literal` ratio list drifts from `ASPECT_RATIOS` | Add a test asserting the two are equal |
| Importing `storage` creates `/outputs` | Set `IMAGE_OUTPUT_DIR` in tests and local dev |

## Verification

```bash
python -c "from server import mcp; print([t.name for t in mcp._tool_manager.list_tools()])"
uv run pytest tests/ -v
```

Then run the GPU acceptance checks from spec.md: AS-1, AS-2, AS-5, AS-6.
