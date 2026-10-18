# Feature 001 — Research

Facts that the design depends on, each verified against the pinned dependency or the primary source. **Every item here should be confirmed before writing `plan.md`.** Items marked *(found late)* were discovered at runtime during v0.1.1 rather than during research. They are recorded here so they inform future features (Constitution IV).

## R-1 Which FastMCP

**Question:** `mcp.server.fastmcp.FastMCP` (official SDK) or the standalone `fastmcp` package?

**Finding:** We use the official SDK, `mcp[cli] >= 1.27.0`, and import `from mcp.server.fastmcp import FastMCP, Context, Image`. The two packages have different constructor and `run()` signatures, and examples for one do not apply to the other.

## R-2 Lifespan scope in `mcp.server.fastmcp` 1.27.0 *(found late)*

**Question:** Does `FastMCP(lifespan=...)` run once per process?

**Finding:** No. With Streamable HTTP and `json_response=True`, the lifespan context ran **per session**. Each new transport session, including every Docker healthcheck probe, logged `Created new transport with session ID` and then `Loading image model`. That reloaded about 60 GB of weights over and over.

**Decision:** Do not load the model in the lifespan. Use a module-level singleton, loaded in `__main__` before `mcp.run()`, and give tools access through `get_model()`. See [plan D-2](plan.md#d-2-model-lifecycle-module-singleton-not-lifespan).

**Source:** `ai-workspace/plans/bug-fix.md`. The original plan's "lifespan runs ONCE" gotcha (#7) was wrong for this SDK version.

## R-3 `FastMCP.run()` signature *(found late)*

**Question:** Where do host and port go?

**Finding:** In 1.27.0, `run()` accepts only `transport`. Passing `host=`/`port=`/`path=` raises `TypeError: FastMCP.run() got an unexpected keyword argument 'host'`. Host and port are constructor arguments: `FastMCP(name, host=..., port=...)`.

**Open:** The path is also a constructor setting (`streamable_http_path`, default `/mcp`). The current code doesn't pass it, so `MCP_PATH` has no effect. Confirm the exact keyword against the installed SDK before wiring it up.

## R-4 Returning images

**Finding:** `Image(data=<bytes>, format="png")`. Here `data` is raw PNG bytes and the SDK does the base64 encoding. Passing bytes that are already base64-encoded double-encodes them and produces an image the client can't decode.

## R-5 Qwen-Image-2512 pipeline parameters

> **Superseded by [005](../005-qwen-image-2-1/spec.md):** the 2512 pipeline facts, resolution table and default negative prompt below no longer apply. The backend is now Qwen-Image-2.1.

**Source:** the Hugging Face model card for `Qwen/Qwen-Image-2512`.

| Fact | Value |
|------|-------|
| Loader | `diffusers.DiffusionPipeline.from_pretrained("Qwen/Qwen-Image-2512", torch_dtype=...)` |
| CFG kwarg | **`true_cfg_scale`**, not `guidance_scale` (the two are different parameters) |
| Default CFG | 4.0 |
| Default steps | 50 |
| Negative prompt | Official default includes Chinese terms; the model responds to them even with English prompts |
| Seeding | `torch.Generator(device=<model device>).manual_seed(seed)` |

**Official resolutions:**

| Ratio | W × H |
|-------|-------|
| 1:1 | 1328 × 1328 |
| 16:9 | 1664 × 928 |
| 9:16 | 928 × 1664 |
| 4:3 | 1472 × 1104 |
| 3:4 | 1104 × 1472 |
| 3:2 | 1584 × 1056 |
| 2:3 | 1056 × 1584 |

## R-6 Meta-tensor initialization *(found late)*

**Finding:** The pipeline initializes on accelerate's `meta` device. Calling `pipeline.to("cuda")` after `from_pretrained()` raises `NotImplementedError: Cannot copy out of meta tensor`, because meta tensors have no data to copy.

**Decision:** Pass `device_map` to `from_pretrained()` so placement happens while weights load. Never call `.to()` afterwards. v0.1.1 used `"cuda"`; feature 004 changes it to `"balanced"` to support `max_memory`. On CPU, omit `device_map`.

## R-7 Blocking behavior

**Finding:** `pipeline(...)` is synchronous and runs for minutes on GB10. Calling it straight from an `async def` tool stalls the event loop for every session. `asyncio.to_thread(model.generate, request)` is enough. No process pool is needed, because torch releases the GIL during kernels.

## R-8 Settings library

**Finding:** `BaseSettings` lives in the separate `pydantic-settings` package, not in `pydantic` v2. Both are needed. `Literal[...]` fields give startup validation for `IMAGE_MODEL` and `TORCH_DTYPE`.

## R-9 Memory footprint on GB10

**Finding:** BF16 with all weights on the device uses about 62 GB of the 128 GB unified pool. That matters on this host because vLLM is also resident. See feature [004](../004-memory-bounded-loading/spec.md).
