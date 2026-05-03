#!/usr/bin/env bash
# =============================================================================
# scaffold.sh — Generate project scaffolding for local-image-gen-mcp
#
# Run from inside the local-image-gen-mcp directory:
#   chmod +x scaffold.sh
#   ./scaffold.sh
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log()  { echo -e "${GREEN}[+]${NC} $1"; }
info() { echo -e "${BLUE}[.]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }

# ---------------------------------------------------------------------------
# Safety check — confirm we are inside local-image-gen-mcp
# ---------------------------------------------------------------------------
CURRENT_DIR=$(basename "$PWD")
if [ "$CURRENT_DIR" != "local-image-gen-mcp" ]; then
    warn "Current directory is '$CURRENT_DIR', not 'local-image-gen-mcp'."
    read -r -p "Continue anyway? [y/N] " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

echo ""
echo "  Scaffolding local-image-gen-mcp"
echo "  ================================"
echo ""

# ---------------------------------------------------------------------------
# Create directories
# ---------------------------------------------------------------------------
info "Creating directories..."

mkdir -p models
mkdir -p config
mkdir -p storage
mkdir -p docker
mkdir -p tests

log "Directories created"

# ---------------------------------------------------------------------------
# Root files
# ---------------------------------------------------------------------------
info "Creating root files..."

touch README.md
touch pyproject.toml
touch server.py

# .env.example
cat > .env.example << 'EOF'
# =============================================================================
# .env.example — copy to .env and fill in your values
# =============================================================================

# HuggingFace token (required to download models)
HF_TOKEN=hf_your_token_here

# MCP server bind settings
MCP_HOST=0.0.0.0
MCP_PORT=8500

# Model to use — must match a backend in models/
# Options: qwen_image_2512
IMAGE_MODEL=qwen_image_2512

# Image output directory (inside container)
IMAGE_OUTPUT_DIR=/outputs

# Diffusion settings
MAX_INFERENCE_STEPS=50
DEFAULT_CFG_SCALE=4.0
TORCH_DTYPE=bfloat16
EOF

# .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
.Python
*.egg-info/
dist/
build/
.eggs/
*.egg

# Virtual environments
.venv/
venv/
env/

# uv
.uv/

# Environment
.env

# Generated images
outputs/
*.png
*.jpg
*.jpeg
*.webp

# Model cache (large — don't commit)
.cache/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Docker
docker/data/
EOF

log "Root files created"

# ---------------------------------------------------------------------------
# models/
# ---------------------------------------------------------------------------
info "Creating models/ package..."

touch models/__init__.py
touch models/base.py
touch models/qwen_image_2512.py

log "models/ package created"

# ---------------------------------------------------------------------------
# config/
# ---------------------------------------------------------------------------
info "Creating config/ package..."

touch config/__init__.py
touch config/settings.py
touch config/aspect_ratios.py

log "config/ package created"

# ---------------------------------------------------------------------------
# storage/
# ---------------------------------------------------------------------------
info "Creating storage/ package..."

touch storage/__init__.py
touch storage/image_store.py

log "storage/ package created"

# ---------------------------------------------------------------------------
# docker/
# ---------------------------------------------------------------------------
info "Creating docker/ files..."

touch docker/Dockerfile
touch docker/docker-compose.yml
touch docker/docker-compose.override.yml

log "docker/ files created"

# ---------------------------------------------------------------------------
# tests/
# ---------------------------------------------------------------------------
info "Creating tests/ package..."

touch tests/__init__.py
touch tests/test_tools.py
touch tests/test_models.py

log "tests/ package created"

# ---------------------------------------------------------------------------
# Print final tree
# ---------------------------------------------------------------------------
echo ""
echo "  Done. Project structure:"
echo ""

if command -v tree &> /dev/null; then
    tree -a --dirsfirst -I ".git"
else
    # Fallback if tree is not installed
    find . -not -path "./.git/*" -not -name ".git" | sort | \
    sed -e 's/[^-][^\/]*\//  │  /g' -e 's/─\//─ /' -e 's/[^│]*─ /├── /'
fi

echo ""
echo -e "${GREEN}Scaffold complete.${NC}"
echo ""
echo "  Next steps:"
echo "    1. cp .env.example .env"
echo "    2. Fill in HF_TOKEN in .env"
echo "    3. Implement models/base.py (abstract base class)"
echo "    4. Implement models/qwen_image_2512.py (first backend)"
echo "    5. Implement config/settings.py (Pydantic settings)"
echo "    6. Implement server.py (FastMCP entrypoint)"
echo ""