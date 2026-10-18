Add two new tools to the existing MCP server at:
~/vscode-workspace/local-image-gen-mcp/server.py

## Context
The MCP server uses mcp 1.27.0 FastMCP with a module-level singleton pattern
(no lifespan). Tools access the image store via `store` imported from `storage`.
The `store` object has these methods:
  - store.list_all() → list[StoredImage]
  - store.get(image_id: str) → StoredImage  (raises KeyError if not found)
  - store.get_image_bytes(image_id: str) → bytes
  - StoredImage has: image_id, prompt, aspect_ratio, width, height, seed, generation_time, model_name, disk_path

## Task
Add exactly two new tools to server.py. Insert them AFTER the existing
`get_model_info` tool and BEFORE the `# ── Resources` section comment.
Do not modify any existing code.

## Tool 1: get_image_base64

```python
@mcp.tool()
def get_image_base64(image_id: str) -> str:
    """
    Return a previously generated image as a base64-encoded PNG string.

    Use this when you need to save the image to a client-side filesystem.
    The returned string is plain base64 — decode it to binary before writing
    to a .png file. Writing the raw string to a .png file will NOT work.

    Correct client-side usage (Python):
        import base64
        data = get_image_base64(image_id)
        with open('output.png', 'wb') as f:
            f.write(base64.b64decode(data))

    Args:
        image_id: UUID returned by generate_image or from images://list resource.

    Returns:
        Base64-encoded PNG string (plain text, not a data URI).
    """
    import base64 as b64
    img_bytes = store.get_image_bytes(image_id)
    return b64.b64encode(img_bytes).decode("utf-8")
```

## Tool 2: get_latest_image_id

```python
@mcp.tool()
def get_latest_image_id() -> dict:
    """
    Return the image_id of the most recently generated image.

    Use this immediately after generate_image to get the image_id without
    having to parse the images://list resource. Combines with get_image_base64
    for a two-step save workflow:
        1. id = get_latest_image_id()["image_id"]
        2. b64 = get_image_base64(id)
        3. decode and write to .png on client filesystem

    Returns:
        dict with keys:
          image_id:     UUID of the most recent image
          prompt:       First 100 chars of the prompt used
          aspect_ratio: e.g. "16:9"
          width:        Pixel width
          height:       Pixel height
          seed:         Seed used for generation
          model_name:   Model that generated it

    Raises:
        RuntimeError if no images have been generated yet this session.
    """
    images = store.list_all()
    if not images:
        raise RuntimeError(
            "No images have been generated in this session. "
            "Call generate_image first."
        )
    latest = images[0]  # list_all() returns newest first
    return {
        "image_id": latest.image_id,
        "prompt": latest.prompt[:100],
        "aspect_ratio": latest.aspect_ratio,
        "width": latest.width,
        "height": latest.height,
        "seed": latest.seed,
        "model_name": latest.model_name,
    }
```

## Verification
After editing, run:
  python3 -c "from server import mcp; tools = [t.name for t in mcp._tool_manager.list_tools()]; print(tools); assert 'get_image_base64' in tools; assert 'get_latest_image_id' in tools; print('OK')"

Expected output includes both new tool names and prints OK.

## Then rebuild and restart
cd ~/vscode-workspace/local-image-gen-mcp
docker compose -f ./docker/docker-compose.yml build --no-cache
docker compose -f ./docker/docker-compose.yml up -d
docker logs -f local-image-gen-mcp

## Do not
- Do not modify any existing tools, resources, prompts, or imports
- Do not change server startup or entrypoint code
- Do not add any new imports at the top of the file (both tools only need
  `import base64 as b64` inline inside get_image_base64, which is already
  handled in the tool body above)
- Do not change docker-compose.yml or any other file