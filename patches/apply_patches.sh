#!/bin/bash
# Re-apply site-package patches after reinstall or environment reset.
# These patches are required for inference to work with PyTorch 2.3 (nv24.03).
#
# Patches:
#   1. diffusers/utils/torch_utils.py    — handle missing torch.xpu.*
#   2. diffusers/models/attention_dispatch.py — enable_gqa TypeError fallback
#   3. transformers/utils/import_utils.py    — accept nv24.xx PyTorch builds

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SITE_PKG="$HOME/.local/lib/python3.10/site-packages"

echo "Applying site-package patches..."

cp "$SCRIPT_DIR/site-packages/diffusers/utils/torch_utils.py" \
   "$SITE_PKG/diffusers/utils/torch_utils.py"
echo "  patched: diffusers/utils/torch_utils.py"

cp "$SCRIPT_DIR/site-packages/diffusers/models/attention_dispatch.py" \
   "$SITE_PKG/diffusers/models/attention_dispatch.py"
echo "  patched: diffusers/models/attention_dispatch.py"

cp "$SCRIPT_DIR/site-packages/transformers/utils/import_utils.py" \
   "$SITE_PKG/transformers/utils/import_utils.py"
echo "  patched: transformers/utils/import_utils.py"

echo "Done."
