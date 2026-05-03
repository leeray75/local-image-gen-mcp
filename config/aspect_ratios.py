# Aspect ratio definitions for Qwen-Image-2512
# Exact pixel dimensions from official Qwen-Image-2512 model card

ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1":  (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928,  1664),
    "4:3":  (1472, 1104),
    "3:4":  (1104, 1472),
    "3:2":  (1584, 1056),
    "2:3":  (1056, 1584),
}


def get_dimensions(ratio: str) -> tuple[int, int]:
    """Return (width, height) for a ratio string. Raises ValueError if invalid."""
    if ratio not in ASPECT_RATIOS:
        raise ValueError(
            f"Invalid aspect_ratio '{ratio}'. "
            f"Must be one of: {sorted(ASPECT_RATIOS.keys())}"
        )
    return ASPECT_RATIOS[ratio]


def valid_ratios() -> list[str]:
    """Return sorted list of valid ratio strings."""
    return sorted(ASPECT_RATIOS.keys())