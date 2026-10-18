"""Configuration management using Pydantic BaseSettings.

All configuration is read from environment variables.
No values are hardcoded — defaults are provided for local development.
"""

from pydantic import field_validator
from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    """Server configuration loaded entirely from environment variables."""

    # MCP server
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8500
    mcp_path: str = "/mcp"

    # Model selection — controls which backend in models/ is loaded
    image_model: Literal["qwen_image_2_1"] = "qwen_image_2_1"

    # Storage
    image_output_dir: str = "/outputs"

    # Diffusion defaults
    # None = use the backend's native default (these are optional overrides)
    max_inference_steps: int | None = None
    default_cfg_scale: float | None = None
    # Device-memory cap in GB (accelerate max_memory); None = uncapped
    image_gpu_memory_limit: int | None = None
    max_reference_image_mb: int = 20
    torch_dtype: Literal["bfloat16", "float32"] = "bfloat16"

    # HuggingFace — from environment only, never hardcoded
    hf_token: str | None = None

    @field_validator(
        "max_inference_steps", "default_cfg_scale", "image_gpu_memory_limit",
        mode="before",
    )
    @classmethod
    def _empty_to_none(cls, v):
        """Treat an empty env string (e.g. from compose) as unset."""
        return None if v == "" else v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton instance — imported throughout the application
settings = Settings()