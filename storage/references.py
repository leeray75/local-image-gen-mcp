"""Reference-image decoding and resolution for edit_image.

References are either store image IDs (canonical hyphenated UUIDs) or
base64-encoded PNG/JPEG/WebP images. Validation happens here, before any
model work starts.
"""

import base64
import binascii
import io
import logging
import re

from PIL import Image as PILModule
from PIL.Image import Image as PILImage

log = logging.getLogger(__name__)

# Only the canonical 36-char hyphenated form is an ID. A bare 32-hex string is
# valid base64 and must not be mistaken for an ID.
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)
_DATA_URI_RE = re.compile(r"^data:image/[A-Za-z0-9.+-]+;base64,", re.I)
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}


def decode_base64_image(data: str, max_bytes: int) -> PILImage:
    """Decode a base64 image string into a fully loaded PIL image.

    Raises ValueError with a human-readable reason (no position; the caller
    adds it).
    """
    data = _DATA_URI_RE.sub("", data.strip())
    data = "".join(data.split())
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("invalid base64") from None
    if not raw:
        raise ValueError("invalid base64 (empty)")
    if len(raw) > max_bytes:
        raise ValueError(
            f"decoded size {len(raw) / (1024 * 1024):.1f} MB exceeds "
            f"MAX_REFERENCE_IMAGE_MB={max_bytes // (1024 * 1024)}"
        )
    try:
        img = PILModule.open(io.BytesIO(raw))
        fmt = img.format
        max_px = PILModule.MAX_IMAGE_PIXELS
        # Pillow only raises above 2x the limit (warns between 1x and 2x)
        if max_px is not None and img.width * img.height > max_px:
            raise ValueError("image too large (decompression limit)")
        if fmt not in ALLOWED_FORMATS:
            raise ValueError(
                f"unsupported format {fmt} (PNG, JPEG, WebP only)"
            )
        img.load()  # fail here on truncated/corrupt data, not inside the pipeline
    except PILModule.DecompressionBombError:
        raise ValueError("image too large (decompression limit)") from None
    except ValueError:
        raise
    except Exception as exc:  # UnidentifiedImageError, OSError, SyntaxError...
        raise ValueError(f"not a valid image ({type(exc).__name__})") from None
    return img


def resolve_references(
    entries: list[str],
    store,
    max_images: int,
    max_bytes: int,
) -> tuple[list[PILImage], list[str]]:
    """Resolve reference entries to PIL images, in caller order.

    Returns (images, store_ids_used_in_order). Every failure is a ValueError
    naming the 1-based position of the offending reference.
    """
    if not entries or len(entries) > max_images:
        raise ValueError(
            f"images must contain 1–{max_images} references (got {len(entries)})"
        )

    images: list[PILImage] = []
    ids: list[str] = []
    for i, entry in enumerate(entries, start=1):
        try:
            candidate = entry.strip()
            if UUID_RE.match(candidate):
                try:
                    candidate = candidate.lower()  # store keys are lowercase
                    img = store.load_image(candidate)
                except KeyError:
                    raise ValueError(
                        f"image '{candidate}' not found (images are kept only "
                        "for the server's lifetime)"
                    ) from None
                except FileNotFoundError:
                    raise ValueError(
                        f"image file for '{candidate}' is missing from disk"
                    ) from None
                ids.append(candidate)
            else:
                img = decode_base64_image(entry, max_bytes)
        except ValueError as exc:
            raise ValueError(f"reference {i}: {exc}") from None
        images.append(img)
    return images, ids
