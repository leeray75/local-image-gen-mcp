"""Configuration management using Pydantic BaseSettings.

All configuration is read from environment variables.
No values are hardcoded — defaults are provided for local development.
"""

import math

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
    # Total HTTP request-body limit in MB (the SDK's own default is 4 MiB).
    # None = computed: room for 10 base64 references (see effective_max_request_body_mb()).
    max_request_body_mb: int | None = None
    # Pixel-area target for edit_image output (the pipeline's output_resolution).
    # Default 1024: the 006 A/B test found 2048 edits artifacted (research R-1).
    edit_output_resolution: int = 1024
    torch_dtype: Literal["bfloat16", "float32"] = "bfloat16"

    # HuggingFace — from environment only, never hardcoded
    hf_token: str | None = None

    @field_validator(
        "max_inference_steps", "default_cfg_scale", "image_gpu_memory_limit",
        "max_request_body_mb",
        mode="before",
    )
    @classmethod
    def _empty_to_none(cls, v):
        """Treat an empty env string (e.g. from compose) as unset."""
        return None if v == "" else v

    @field_validator("edit_output_resolution", mode="before")
    @classmethod
    def _empty_to_default(cls, v):
        """Treat an empty env string as 'use the default'."""
        return cls.model_fields["edit_output_resolution"].default if v == "" else v

    @field_validator("edit_output_resolution")
    @classmethod
    def _check_edit_resolution(cls, v: int) -> int:
        if v % 32 != 0 or not 512 <= v <= 2048:
            raise ValueError(
                "EDIT_OUTPUT_RESOLUTION must be a multiple of 32 between 512 and 2048"
            )
        return v

    @field_validator("max_request_body_mb")
    @classmethod
    def _check_request_body(cls, v: int | None) -> int | None:
        if v is not None and v < 4:
            raise ValueError("MAX_REQUEST_BODY_MB must be at least 4 (the SDK default)")
        return v

    def effective_max_request_body_mb(self, references: int = 10) -> int:
        """Explicit MAX_REQUEST_BODY_MB, else ceil(references x reference MB x 4/3) + 1."""
        if self.max_request_body_mb is not None:
            return self.max_request_body_mb
        return math.ceil(references * self.max_reference_image_mb * 4 / 3) + 1

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton instance — imported throughout the application
settings = Settings()