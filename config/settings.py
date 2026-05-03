"""Configuration management using Pydantic BaseSettings.

All configuration is read from environment variables.
No values are hardcoded — defaults are provided for local development.
"""

from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    """Server configuration loaded entirely from environment variables."""

    # MCP server
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8500
    mcp_path: str = "/mcp"

    # Model selection — controls which backend in models/ is loaded
    image_model: Literal["qwen_image_2512"] = "qwen_image_2512"

    # Storage
    image_output_dir: str = "/outputs"

    # Diffusion defaults
    max_inference_steps: int = 50
    default_cfg_scale: float = 4.0
    torch_dtype: Literal["bfloat16", "float32"] = "bfloat16"

    # HuggingFace — from environment only, never hardcoded
    hf_token: str | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton instance — imported throughout the application
settings = Settings()