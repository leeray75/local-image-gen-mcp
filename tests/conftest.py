"""Shared test setup.

IMAGE_OUTPUT_DIR must point at a writable directory BEFORE `storage` or
`server` is imported, because the store singleton creates it at import time.
"""

import os
import tempfile

os.environ.setdefault("IMAGE_OUTPUT_DIR", tempfile.mkdtemp(prefix="imggen-test-"))
# Keep the developer's/CI environment from skewing default resolution
for _var in ("MAX_INFERENCE_STEPS", "DEFAULT_CFG_SCALE", "IMAGE_GPU_MEMORY_LIMIT",
             "EDIT_OUTPUT_RESOLUTION", "GENERATE_OUTPUT_RESOLUTION", "MAX_OUTPUT_RESOLUTION"):
    os.environ.pop(_var, None)
