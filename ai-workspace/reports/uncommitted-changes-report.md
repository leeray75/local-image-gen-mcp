# Uncommitted Changes Report

> **Historical snapshot.** This report describes the repo as of its date. For current state see `CHANGELOG.md`, `docs/specs/SPECS.md` and later reports.

**Date:** 2026-05-04  
**Branch:** `bugfix/meta-tensor-and-lifespan-fixes`  
**Last Committed:** `a42846d` - add memory limit 40g to docker compose  
**Repository:** `local-image-gen-mcp`

---

## Summary

Two changes are present but not yet committed:

| Status | File | Description |
|--------|------|-------------|
| Modified | `server.py` | Added two new MCP tools |
| Untracked | `ai-workspace/plans/add-tools.md` | Plan/specification document for the new tools |

---

## Detailed Changes

### 1. `server.py` — Modified

**Location:** After `get_model_info()` tool, before `# ── Resources` section  
**Lines added:** +75

Two new MCP tools were added to the server:

#### Tool 1: `get_image_base64(image_id: str) -> str`
- **Purpose:** Returns a previously generated image as a base64-encoded PNG string.
- **Use case:** Enables client-side saving of images to a filesystem without requiring direct file access.
- **Implementation:** Uses `store.get_image_bytes(image_id)` then encodes with `base64.b64encode()`.
- **Note:** The docstring explicitly warns that the raw base64 string must be decoded before writing to a `.png` file.

#### Tool 2: `get_latest_image_id() -> dict`
- **Purpose:** Returns metadata about the most recently generated image.
- **Use case:** Provides a convenient way to fetch the latest image ID without parsing the `images://list` resource.
- **Returns:** Dict with keys: `image_id`, `prompt` (truncated to 100 chars), `aspect_ratio`, `width`, `height`, `seed`, `model_name`.
- **Raises:** `RuntimeError` if no images exist in the current session.
- **Implementation:** Calls `store.list_all()` and returns the first entry (newest first ordering).

#### Workflow
The two tools are designed to work together for a two-step save workflow:
1. `id = get_latest_image_id()["image_id"]`
2. `b64 = get_image_base64(id)`
3. Decode and write to `.png` on client filesystem

---

### 2. `ai-workspace/plans/add-tools.md` — Untracked (New File)

**Purpose:** Specification/plan document that describes the tool addition task.  
**Content:** Contains:
- Context about the MCP server architecture (mcp 1.27.0 FastMCP, singleton pattern)
- Complete code for both tools
- Verification commands
- Docker rebuild/restart instructions
- Constraints (what not to modify)

This file served as the implementation plan for the changes in `server.py`.

---

## Git Status

```
On branch bugfix/meta-tensor-and-lifespan-fixes
up to date with origin/bugfix/meta-tensor-and-lifespan-fixes

Changes not staged for commit:
  modified:   server.py

Untracked files:
  ai-workspace/plans/add-tools.md
```

---

## Recommendations

1. **Review** the diff in `server.py` to ensure the tools match the specification in `add-tools.md`.
2. **Decide** whether `add-tools.md` should be committed (as documentation of the change plan) or excluded (if plans are not tracked in git).
3. **Stage and commit** using the standard sequence:
   ```bash
   git add server.py
   git commit -m "feat: add get_image_base64 and get_latest_image_id tools"