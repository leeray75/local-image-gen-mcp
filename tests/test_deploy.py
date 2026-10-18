"""Deployment config checks: docker-compose.yml (006 AS-6)."""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML not installed in this image")

COMPOSE = Path(__file__).resolve().parent.parent / "docker" / "docker-compose.yml"


@pytest.fixture(scope="module")
def compose():
    return yaml.safe_load(COMPOSE.read_text())


def _mounts(compose) -> dict[str, str]:
    """target path -> source, for short-syntax volume entries."""
    svc = compose["services"]["local-image-gen-mcp"]
    out = {}
    for v in svc["volumes"]:
        src, target = v.split(":")[:2]
        out[target] = src
    return out


def test_compile_caches_mounted_and_declared(compose):
    mounts = _mounts(compose)
    declared = compose["volumes"]
    for target in ("/root/.nv", "/root/.cache/torch"):
        assert target in mounts, f"{target} is not mounted"
        assert mounts[target] in declared, f"{mounts[target]} not declared"
    assert mounts["/root/.nv"] != "/root/.cache/huggingface"


def test_hf_bind_mount_not_shadowed(compose):
    mounts = _mounts(compose)
    assert mounts["/root/.cache/huggingface"] == "~/.cache/huggingface"
    assert "/root/.cache" not in mounts


def test_override_network_is_external_with_default_name():  # 006 F-3
    override = yaml.safe_load(COMPOSE.with_name("docker-compose.override.yml").read_text())
    net = override["networks"]["ai-bridge"]
    assert net["external"] is True
    assert net["name"] == "${AI_BRIDGE_NETWORK:-ai-litellm-proxy_ai-bridge}"


def test_restart_policy_is_no(compose):
    # YAML 1.1 parses an unquoted `no` as False
    assert compose["services"]["local-image-gen-mcp"]["restart"] in ("no", False)


def test_compose_passes_resolution_settings_through(compose):  # 007 FR-007-08
    env = compose["services"]["local-image-gen-mcp"]["environment"]
    for var in ("GENERATE_OUTPUT_RESOLUTION", "MAX_OUTPUT_RESOLUTION", "EDIT_OUTPUT_RESOLUTION"):
        assert env[var] == "${%s:-}" % var
