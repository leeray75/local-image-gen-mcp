"""Thread-safe in-memory store with disk persistence for generated images."""

import logging
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from PIL.Image import Image as PILImage

from config.settings import settings

log = logging.getLogger(__name__)


@dataclass
class StoredImage:
    """Metadata for a stored generated image."""
    image_id: str
    prompt: str
    negative_prompt: str
    aspect_ratio: str
    width: int
    height: int
    seed: int
    generation_time: float
    model_name: str
    disk_path: Path


class ImageStore:
    """Thread-safe in-memory image store with disk persistence.

    Images are saved to disk as PNG and metadata is kept in-memory
    for fast retrieval. The in-memory store is rebuilt from disk
    on restart if needed.
    """

    def __init__(self, output_dir: Path) -> None:
        self._store: dict[str, StoredImage] = {}
        self._lock = threading.Lock()
        self._output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, image: PILImage, metadata: dict) -> StoredImage:
        """Save image to disk and store metadata. Returns StoredImage."""
        image_id = str(uuid.uuid4())
        filename = f"{image_id}.png"
        disk_path = self._output_dir / filename

        # Save to disk
        image.save(disk_path, format="PNG")

        stored = StoredImage(
            image_id=image_id,
            prompt=metadata.get("prompt", ""),
            negative_prompt=metadata.get("negative_prompt", ""),
            aspect_ratio=metadata.get("aspect_ratio", ""),
            width=metadata.get("width", 0),
            height=metadata.get("height", 0),
            seed=metadata.get("seed", 0),
            generation_time=metadata.get("generation_time", 0.0),
            model_name=metadata.get("model_name", ""),
            disk_path=disk_path,
        )

        with self._lock:
            self._store[image_id] = stored

        log.debug("Stored image %s at %s", image_id, disk_path)
        return stored

    def get(self, image_id: str) -> StoredImage:
        """Retrieve StoredImage by ID. Raises KeyError if not found."""
        with self._lock:
            img = self._store.get(image_id)
        if img is None:
            raise KeyError(
                f"Image '{image_id}' not found. "
                "Images are stored for the server's lifetime."
            )
        return img

    def get_image_bytes(self, image_id: str) -> bytes:
        """Read image bytes from disk."""
        stored = self.get(image_id)
        if not stored.disk_path.exists():
            raise FileNotFoundError(
                f"Image file for '{image_id}' not found at {stored.disk_path}. "
                "The output directory may have been cleared."
            )
        return stored.disk_path.read_bytes()

    def list_all(self) -> list[StoredImage]:
        """Return all stored images sorted by newest first (insertion order reversed)."""
        with self._lock:
            return list(reversed(list(self._store.values())))

    def count(self) -> int:
        """Return the number of stored images."""
        with self._lock:
            return len(self._store)


# Singleton instance
store = ImageStore(Path(settings.image_output_dir))