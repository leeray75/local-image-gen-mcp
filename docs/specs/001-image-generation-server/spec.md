# Feature 001 — Image Generation MCP Server

| | |
|---|---|
| **Status** | Implemented (v0.1.0, corrected in v0.1.1) |
| **Constitution** | I, II, III, IV, V, VII, VIII |
| **Depends on** | — |
| **Next** | [research.md](research.md) → [data-model.md](data-model.md) · [contracts](contracts/mcp-interface.md) → [plan.md](plan.md) → [tasks.md](tasks.md) |

## Problem

Agents running in Claude Code and Cline need to generate images during a task: illustrations, mockups, reference photos. Hosted image APIs cost money per image, send prompts off-box, and can't use the GPU that already sits idle on the DGX Spark. The existing prototype `server.py` was a single file: the model was hardcoded, config came from scattered `os.getenv` calls, the pipeline loaded lazily on the first request, and images lived in a global dict. Adding a second model would have meant rewriting the server.

## Goal

An MCP server that exposes local text-to-image generation as tools, resources and prompts any MCP client can use. The diffusion model is a pluggable backend chosen by configuration. The first backend is Qwen-Image-2512.

## Non-goals

- Image-to-image, inpainting, upscaling, or LoRA loading.
- Multi-GPU or multi-host scheduling.
- Authentication or multi-tenant isolation. The server is reached on a trusted host or bridge network.
- Durable image catalog across restarts (see FR-001-10).
- Supporting the standalone `fastmcp` package.

## Actors

- **Agent:** an LLM in an MCP client that calls tools.
- **Operator:** the person who deploys and configures the server.
- **Backend author:** a developer adding a new diffusion model.

## User stories

**US-1 (P1) — Generate an image.** As an agent, I can call one tool with a prompt and get a PNG back in the same response, so I can use it straight away.

**US-2 (P1) — Choose the shape.** As an agent, I can find out which aspect ratios are supported and ask for one, so the output fits its purpose (wallpaper, phone, avatar).

**US-3 (P2) — Reproduce a result.** As an agent, I can pass a seed and get the same image for the same inputs, and I'm told which seed was used when I don't pass one.

**US-4 (P2) — Retrieve earlier images.** As an agent, I can list the images generated this session and fetch one by ID without regenerating it.

**US-5 (P2) — Check readiness.** As an agent or operator, I can ask the server which model is loaded and how much GPU memory it is using.

**US-6 (P3) — Better prompts.** As an agent, I can use prompt templates that turn a short subject into a prompt the model handles well, and fetch the recommended negative prompt.

**US-7 (P2) — Swap the model.** As a backend author, I can add a new model by implementing one interface and registering it, without touching tool code.

**US-8 (P1) — Configure via environment.** As an operator, I set everything through environment variables, and the server refuses to start on invalid values.

## Functional requirements

### Generation

- **FR-001-01** The server MUST expose a `generate_image` tool that takes `prompt` (required), `aspect_ratio`, `negative_prompt`, `num_inference_steps`, `cfg_scale`, and `seed`, and returns the image as MCP image content (PNG).
- **FR-001-02** `aspect_ratio` MUST be one of the ratios defined by the active configuration (`1:1`, `16:9`, `9:16`, `4:3`, `3:4`, `3:2`, `2:3` for Qwen-Image-2512). Each ratio maps to the exact pixel dimensions published on the model card. The default is `1:1`.
- **FR-001-03** `num_inference_steps` MUST be clamped to the range 20–100. The default is 50.
- **FR-001-04** `cfg_scale` MUST be accepted in the range 1.0–10.0 and default to 4.0. Values outside the range SHOULD be clamped or rejected with a clear error.
- **FR-001-05** When `seed` is omitted, the server MUST pick a seed in `[0, 2³²)` and report it to the client. Identical inputs with the same seed MUST produce the same image on the same hardware and dtype.
- **FR-001-06** When `negative_prompt` is omitted, the backend's default negative prompt MUST be used.
- **FR-001-07** The server MUST report progress at the start and end of generation, and send an info log message that includes the image ID, dimensions, seed and duration.

### Discovery and status

- **FR-001-08** The server MUST expose a `list_aspect_ratios` tool that returns every supported ratio with its width and height.
- **FR-001-09** The server MUST expose a `get_model_info` tool that returns:
  - the model name, backend ID, whether it is loaded, the device and the dtype
  - the default steps and CFG
  - the supported ratios, the count of images generated, and the output directory
  - when CUDA is present: the GPU name and total, allocated and reserved memory (GB)

### Storage and retrieval

- **FR-001-10** Every generated image MUST be written to `IMAGE_OUTPUT_DIR` as `<uuid>.png`. Its metadata (prompt, negative prompt, ratio, width, height, seed, generation time, model name) is kept in an in-process index. The index is not required to survive a restart, and docs MUST NOT claim that it does.
- **FR-001-11** The server MUST expose an `image://{image_id}` resource that returns the stored PNG, and an `images://list` resource that returns a JSON array of image metadata, newest first. Prompts in the list are truncated to 100 characters with an ellipsis.
- **FR-001-12** Asking for an unknown `image_id` MUST give an error that says the image was not found and that images last only for the server's lifetime. If the index entry exists but the file is gone, the error MUST say so.

### Server lifecycle and transport

- **FR-001-13** The server MUST serve MCP over Streamable HTTP at `MCP_HOST:MCP_PORT` on the path `MCP_PATH`, with `json_response=True`.
- **FR-001-14** The model MUST be loaded **exactly once per process**, before the HTTP listener accepts connections. New MCP sessions, including healthcheck probes, MUST NOT trigger a reload (see [research R-2](research.md#r-2-lifespan-scope-in-mcpserverfastmcp-1270-found-late)).
- **FR-001-15** If a tool needs the model before it is loaded, the call MUST fail with a clear "model not loaded" error.

### Prompts

- **FR-001-16** The server MUST expose these prompts:
  - `photorealistic(subject, setting="", style="")`, which returns an enhanced photorealism prompt.
  - `negative_defaults()`, which returns the backend's default negative prompt.

### Extensibility and configuration

- **FR-001-17** Model backends MUST implement `BaseImageModel` (`load`, `unload`, `generate`, `is_loaded`, `model_name`). A factory `load_model()` MUST select the backend from `IMAGE_MODEL`. Unknown values MUST fail validation at startup.
- **FR-001-18** All settings MUST come from environment variables (and an optional `.env`), with the types validated: `MCP_HOST`, `MCP_PORT`, `MCP_PATH`, `IMAGE_MODEL`, `IMAGE_OUTPUT_DIR`, `MAX_INFERENCE_STEPS`, `DEFAULT_CFG_SCALE`, `TORCH_DTYPE` (`bfloat16` | `float32`), `HF_TOKEN`.
- **FR-001-19** If CUDA is unavailable, the backend MUST still load on CPU and log a warning that generation will be very slow.

## Non-functional requirements

- **NFR-001-01 (Responsiveness)** While a generation runs, the server MUST keep answering other MCP requests (list, info, resources). Blocking work runs off the event loop.
- **NFR-001-02 (Startup)** On a warm Hugging Face cache, the server SHOULD be accepting connections within the container healthcheck's start period (180 s).
- **NFR-001-03 (Concurrency)** Concurrent writes to the image index MUST be thread-safe. Concurrent `generate_image` calls SHOULD be serialized, or proven safe on a shared pipeline. *(Open: not currently serialized.)*
- **NFR-001-04 (Secrets)** `HF_TOKEN` MUST NOT appear in logs, tool output, or tracked files.
- **NFR-001-05 (Testability)** The model, storage and config layers MUST have unit tests that pass without a GPU.

## Acceptance scenarios

- **AS-1** Given the server is running with the model loaded, when an agent calls `generate_image(prompt="a red bicycle")`, then it gets a 1328×1328 PNG, a file `<id>.png` exists in `IMAGE_OUTPUT_DIR`, and an info message names `image://<id>` and the seed used.
- **AS-2** Given the same prompt, ratio, steps, cfg and `seed=42`, when `generate_image` is called twice, then the two PNGs are pixel-identical.
- **AS-3** When `generate_image(num_inference_steps=5)` is called, then generation runs with 20 steps.
- **AS-4** When `generate_image(aspect_ratio="5:4")` is called, then the call is rejected by schema validation before generation starts.
- **AS-5** Given a generation is in progress, when another client calls `list_aspect_ratios`, then it gets a response without waiting for the generation to finish.
- **AS-6** Given the server has been up through N healthcheck probes and M client sessions, then the log contains exactly one `Loading image model` line.
- **AS-7** Given images A then B were generated, when `images://list` is read, then B comes before A.
- **AS-8** When `image://does-not-exist` is read, then the error says the image was not found.
- **AS-9** Given `IMAGE_MODEL=unknown`, when the server starts, then it exits with a settings validation error.
- **AS-10** Given no CUDA device, when the server starts, then it loads on CPU and logs a warning.

## Traceability

| Requirement | Implementation | Test | Status |
|-------------|----------------|------|--------|
| FR-001-01, 02 | `server.py` `generate_image`; `config/aspect_ratios.py` | — | ✅ (no test) |
| FR-001-03 | `server.py` clamp | — | ✅ (no test) |
| FR-001-04 | `server.py` | — | ⚠️ no clamp or validation |
| FR-001-05 | `server.py` seed resolution; `qwen_image_2512.generate` Generator | `test_models::test_load_and_generate` (seed echo only) | ✅ / ⏳ determinism unverified |
| FR-001-06 | `DEFAULT_NEGATIVE_PROMPT` default | `test_default_negative_prompt_not_empty` | ✅ |
| FR-001-07 | `ctx.report_progress`, `ctx.info` | — | ✅ |
| FR-001-08, 09 | `list_aspect_ratios`, `get_model_info` | — | ✅ (no test) |
| FR-001-10 | `storage/image_store.py` | — | ⚠️ docstring wrongly says the index is rebuilt from disk |
| FR-001-11, 12 | resources in `server.py`; `ImageStore.get*` | — | ✅ (no test) |
| FR-001-13 | `FastMCP(..., host, port, json_response=True)` | — | ⚠️ `MCP_PATH` not passed through |
| FR-001-14, 15 | `_startup_load_model()`, `get_model()` | — | ✅ (v0.1.1) |
| FR-001-16 | `photorealistic`, `negative_defaults` | — | ✅ |
| FR-001-17 | `models/base.py`, `models/__init__.py`, `Settings.image_model` Literal | `test_is_loaded_before_load`, `test_model_name` | ✅ |
| FR-001-18 | `config/settings.py` | — | ✅ |
| FR-001-19 | CPU branch in `QwenImage2512.load` | — | ✅ |
| NFR-001-01 | `asyncio.to_thread(model.generate, …)` | — | ✅ |
| NFR-001-03 | `threading.Lock` in `ImageStore` | — | ⚠️ generation not serialized |
| NFR-001-05 | `tests/test_models.py` | 7 tests | ⚠️ `tests/test_tools.py` empty |
