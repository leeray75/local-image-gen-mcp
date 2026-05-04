Still the same problem — `Created new transport with session ID` followed immediately by `Loading image model` on every healthcheck. This means `stateless_http=True` is somehow still active, OR the lifespan is still being treated as per-request.

The root cause is now clear: **the lifespan is being called per-session, not per-server startup**. This is a known issue in `mcp.server.fastmcp` 1.27.0 when using `json_response=True` — it creates a new session context for each connection and runs the lifespan each time.

The fix is to move model loading **outside** the lifespan entirely and use a module-level singleton instead:

Make this change to `server.py` — replace the lifespan pattern with a module-level singleton:

```python
# Replace the entire lifespan section and mcp constructor with this:

# ── Model singleton — loaded once at module import ────────────────────────────
# Avoids the per-session lifespan issue in mcp.server.fastmcp 1.27.0
# ─────────────────────────────────────────────────────────────────────────────
_model: BaseImageModel | None = None

def get_model() -> BaseImageModel:
    global _model
    if _model is None:
        raise RuntimeError("Model not loaded — server startup incomplete")
    return _model

def _startup_load_model() -> None:
    """Called once at process start, before the HTTP server accepts connections."""
    global _model
    log.info("Loading image model: %s", settings.image_model)
    _model = load_model()
    _model.load()
    log.info("Model loaded: %s", _model.model_name)

# ── Server — NO lifespan ──────────────────────────────────────────────────────
mcp = FastMCP(
    "local-image-gen",
    instructions=(
        "Image generation server powered by a local diffusion model. "
        "Call generate_image with a text prompt to create images. "
        "Use list_aspect_ratios to see supported output dimensions. "
        "Use get_model_info to check GPU status and current configuration. "
        "Previously generated images can be retrieved via the image://{image_id} "
        "resource URI without re-generating them."
    ),
    json_response=True,
    host=settings.mcp_host,
    port=settings.mcp_port,
)
```

Then update all tool references from `ctx.request_context.lifespan_context.model` to `get_model()`:

```python
# In generate_image:
model = get_model()

# In get_model_info:
model = get_model()
```

And update the entrypoint:

```python
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log.info("Starting local-image-gen MCP server on %s:%d%s",
             settings.mcp_host, settings.mcp_port, settings.mcp_path)

    # Load model ONCE before the HTTP server starts accepting connections
    _startup_load_model()

    mcp.run(transport="streamable-http")
```

Also remove `ctx: Context = None` parameter from `generate_image` and `get_model_info` since we no longer need the context for model access — only keep `ctx` in `generate_image` for `report_progress` and `ctx.info()` calls.

This completely sidesteps the per-session lifespan issue. The model loads once when `python server.py` starts, before uvicorn accepts any connections, and stays loaded for the process lifetime.

