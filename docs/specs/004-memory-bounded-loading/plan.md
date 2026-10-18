# Feature 004 — Implementation Plan

**Inputs:** [spec.md](spec.md), [001 research R-6, R-9](../001-image-generation-server/research.md)

## Research

- **accelerate placement.** `max_memory` is only honored when `device_map` is a strategy string such as `"auto"`, `"balanced"`, or `"balanced_low_0"`. With `device_map="cuda"`, everything goes onto device 0 and the cap is ignored. So the cap requires changing from `"cuda"` to `"balanced"`.
- **Unified memory.** On GB10, "CPU offload" lands in the same physical pool. The cap still helps, because it bounds what the CUDA allocator reserves, and the reserved amount is what competes with vLLM's preallocated KV cache. Offloaded layers are streamed to the GPU on each forward pass, and that streaming is where the slowdown comes from.
- **Why not a cgroup limit.** `deploy.resources.limits.memory` doesn't change the placement decision. When the model goes over the limit, the kernel OOM-kills it.

## Design

In `models/qwen_image_2512.py`:

```python
_env_limit = os.getenv("IMAGE_GPU_MEMORY_LIMIT")
MEMORY_LIMIT_GB: int | None = int(_env_limit) if _env_limit else None

# load(), CUDA branch:
kwargs = {"torch_dtype": dtype, "device_map": "balanced"}
if MEMORY_LIMIT_GB is not None:
    kwargs["max_memory"] = {0: f"{MEMORY_LIMIT_GB}GiB", "cpu": "60GiB"}
pipeline = DiffusionPipeline.from_pretrained(MODEL_ID, **kwargs)
```

The CPU branch is unchanged. Log lines are added for the selected mode and for post-load CUDA allocated/reserved.

In `docker/docker-compose.yml`: add `IMAGE_GPU_MEMORY_LIMIT: "40"` and remove `deploy.resources.limits.memory`.

### Follow-up (FR-004-05)

Move the setting into `Settings` as `image_gpu_memory_limit: int | None = None`. Pass it to `QwenImage2512.__init__` from `load_model()`, in the same way as `torch_dtype`. Drop the module-level `os.getenv`. This also makes AS-4 testable without reloading the module, because the value is no longer read at import time.

## Risks

| Risk | Mitigation |
|------|------------|
| Generation becomes several times slower | Document the trade-off table. The operator chooses the cap. |
| Non-integer env value (`"40GB"`) crashes at import with `ValueError` | Validation through `Settings` (follow-up) gives a clear startup error |
| The `"cpu": "60GiB"` ceiling is hard-coded | Acceptable for GB10. Make it configurable if a different host needs it. |
| The speed figures are estimates | Measure them (task T005) |
