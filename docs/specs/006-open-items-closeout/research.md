# Feature 006 — Research

## R-1 F-1 A/B protocol

**Hypothesis.** `output_resolution` does two jobs in `QwenImage21Pipeline`:
1. It sets the output area for edits.
2. It sets the size every condition image is resized to before it goes to the VAE and the vision encoder (005 research R-3).

The pipeline's default is 1024. The model card's edit example passes no size, so it runs at 1024. Running at 2048 means encoding references and denoising at a resolution the edit path may not have been tuned for. That fits the over-sharpened, HDR-like look seen in 005 AS-15.

**Protocol.** Use the same server image. Restart it with `EDIT_OUTPUT_RESOLUTION=2048`, then with `1024`. Use the same seeds and the same inputs:

| Case | Input | Instruction | Flags |
|------|-------|-------------|-------|
| E1 | the 005 bicycle photo, regenerated with `generate_image` (seed 42) | "Change the bicycle's color to bright blue; keep everything else the same" | — |
| E2 | the same photo | "Make it dusk with warm street-lamp light" | — |
| E3 | the same photo | the `extract_subject` prompt (subject "the bicycle") | `transparent=true` |

For each case and resolution, record:
- the output size
- the time taken
- peak host RAM, sampled with `free -m` every 2 s
- a visual verdict (clean / artifacts), with a short note

Save 512 px JPEG thumbnails, with E3 composited on a checkerboard, to `ab/<case>_<res>.jpg`.

**Decision rule:**
- If 1024 is clean and 2048 shows artifacts in E1 and E2, the default becomes **1024**.
- If both are clean, keep **2048**.
- If neither is clean, keep 2048, leave F-1 open, and escalate to the operator.

### Results (GPU window, 2026-10-18, DGX Spark GB10, vLLM stopped)

The source image was the 40-step seed-42 bicycle. It was regenerated after the restart and was **byte-identical** (sha256 match), so both resolutions edited the same input. Edits used seed 42, 40 steps, CFG 1.0, and one reference passed by `image_id`.

| Case | 2048: size, time | 2048: verdict | 1024: size, time | 1024: verdict |
|------|------------------|---------------|------------------|---------------|
| E1 recolor | 2048², 342.9 s | Recolored, but heavy over-sharpening and HDR artifacts | 1024², 61.4 s | **Clean.** Bright blue; wall, lighting and composition kept |
| E2 dusk | 2048², 336.3 s | **Instruction ignored** (still daylight), same artifacts | 1024², 60.5 s | **Clean.** Convincing dusk, street lamp added |
| E3 extract (transparent) | 2048², 340.0 s | **Extraction failed.** Background kept; alpha range 77–255, 0.0% transparent pixels | 1024², 60.6 s | **Clean cut-out.** Alpha range 0–255, 85.5% transparent pixels |
| Peak host RAM (system "used") | ~90.2 GB | | ~48.7 GB | |

Thumbnails (512 px; E3 is composited on a checkerboard): [source](ab/source.jpg) · 2048: [E1](ab/E1_2048.jpg) [E2](ab/E2_2048.jpg) [E3](ab/E3_2048.jpg) · 1024: [E1](ab/E1_1024.jpg) [E2](ab/E2_1024.jpg) [E3](ab/E3_1024.jpg)

**Decision:** 1024 is clean and 2048 is not, so **the default is 1024** (decision rule above). The hypothesis holds: the edit path breaks down when `output_resolution` (which also sets the reference resize) is 2048. 1024 is also about 5.5× faster and needs about 40 GB less peak memory. F-1 is closed.

## R-2 FastMCP path setting

Checked in the project image (`mcp` 1.30.0): `FastMCP.__init__` accepts `streamable_http_path`, along with `mount_path`, `sse_path` and `message_path`. The SDK default is `/mcp`, which is why `MCP_PATH` appeared to work. Pass `streamable_http_path=settings.mcp_path` to the constructor, next to `host` and `port`. `run()` still takes only `transport` (001 R-3).

## R-3 Dockerfile layer caching

The current order is: `COPY pyproject.toml`, `COPY server.py`, `COPY models/ config/ storage/`, then `RUN pip install …` (with the package list written inline). Docker invalidates every layer after the first changed `COPY`, so any source change re-runs the multi-minute `pip install` of diffusers, transformers and the rest.

The pip command lists its packages inline and doesn't read `pyproject.toml`. So the fix is to move all app `COPY` lines below the `RUN pip install`. `pyproject.toml` isn't used by that step and can be copied with the app files.

**Check:** `touch server.py`, rebuild, and look for `CACHED` on the pip step in the build output.

## R-4 ai-bridge reachability check (002 AS-4)

`ai-bridge` is an external network created by the LiteLLM stack. Start the server with the override file. From a throwaway container on that network, POST an MCP `initialize` request to `http://local-image-gen-mcp:8500/mcp`. Use the project image with Python's `urllib`, so no extra image is pulled. A JSON-RPC result means it is reachable. Don't touch the other stack containers.

## R-5 Compile caches

**Observed** (005 GPU run, `docker diff` of the stopped container):
- `/root/.nv/ComputeCache`: 49 entries, **195 MB**. This is CUDA driver JIT output (PTX compiled to native code for GB10, `sm_121`), because the libraries don't ship prebuilt native code for every kernel on this architecture.
- `/root/.cache/torch/kernels/reduction_prod_arch12.1_nvrtc13.2_sass_…`: a PyTorch NVRTC runtime-compiled kernel.
- Both live in the container's writable layer. They survive `docker stop`/`start` and are lost when the container is recreated.

**Startup timeline** (005 run, from timestamped logs): the weight download took about 4.5 min (one-time, now cached). Loading weights from disk to memory took about 3¼ min, on every start, and can't be cached. The JIT cost hasn't been measured. The first and second 2K generations both took about 256 s, so JIT is either small or happened during load. FR-006-13 measures it.

**Invalidation and safety:**
- **CUDA JIT cache:** the CUDA Programming Guide (JIT compilation section) says the compute cache is invalidated automatically when the device driver is upgraded, so kernels recompile with the new driver's JIT. Entries are keyed by the PTX being compiled, so new library code compiles fresh. `CUDA_CACHE_MAXSIZE` is 1 GiB by default on desktop/server platforms (maximum 4 GiB), with the oldest entries evicted first ([CUDA env vars](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)). `CUDA_CACHE_PATH` defaults to `~/.nv/ComputeCache`. `CUDA_CACHE_DISABLE=1` forces recompilation.
- **PyTorch kernel cache:** file names encode the architecture, the NVRTC version and a source hash. A torch or CUDA toolkit update gives a different key and compiles fresh. Compiled native code stays valid on newer drivers.
- **Linux kernel updates** don't affect either cache.

**Measured** (GPU window, 2026-10-18), following FR-006-13:

| | Cold (new, empty volumes) | Warm (after `compose down && up`) |
|---|---|---|
| Container start → `Uvicorn running` | 194 s (healthy at 212 s) | 197 s |
| First 20-step 1:1 `generate_image` | 136.9 s (the second run took 131.8 s) | 131.0 s |
| `image-gen-cuda-cache` / `image-gen-torch-cache` | populated during the run | 195 MB / 104 KB; the CUDA cache entries survived the recreate |

**Finding:** the compile cache saves about **6 s** on the first generation after a recreate. It has no measurable effect on startup, which is dominated by the ~3¼ min weight load. The volumes are kept anyway (FR-006-13), because they're cheap and will matter more after code or library changes that bring new kernels.

**Mount choice:** `/root/.cache/huggingface` is already a host bind mount. Mounting a volume over all of `/root/.cache` would shadow it, so mount only `/root/.nv` and `/root/.cache/torch`.
