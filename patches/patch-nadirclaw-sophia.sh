#!/bin/bash
# Re-apply Sophia's NadirClaw customizations after a git pull/update.
# Idempotent — safe to run multiple times.
#
# What this patches:
#   1. routing.py: Disables agentic override (OpenClaw always sends tools)
#   2. server.py: Tool forwarding, metadata stripping, high-water-mark,
#                 parameter sanitization, thought_signature workaround,
#                 LiteLLM tool_call_id preservation, session cache disabled
#
# Usage:
#   ~/.nadirclaw/patches/patch-nadirclaw-sophia.sh
#
# After running, restart the service:
#   systemctl --user restart nadirclaw

set -euo pipefail

NADIRCLAW_DIR="$HOME/.nadirclaw"
PATCH_DIR="$NADIRCLAW_DIR/patches"
PATCH_FILE="$PATCH_DIR/sophia-customizations.patch"

cd "$NADIRCLAW_DIR"

if [ ! -f "$PATCH_FILE" ]; then
    echo "ERROR: Patch file not found: $PATCH_FILE"
    echo "Run from Claude Code to regenerate the patch."
    exit 1
fi

# Check if patch is already applied (look for a unique marker)
if grep -q "_strip_openclaw_metadata" nadirclaw/server.py 2>/dev/null; then
    echo "Customizations appear to already be applied (found _strip_openclaw_metadata)."
    echo "If you want to force re-apply, run: git checkout -- nadirclaw/ && $0"
    exit 0
fi

# Backup originals
echo "Backing up originals..."
cp nadirclaw/routing.py "$PATCH_DIR/routing.py.upstream"
cp nadirclaw/server.py "$PATCH_DIR/server.py.upstream"

# Try to apply the patch
echo "Applying Sophia customizations..."
if git apply --check "$PATCH_FILE" 2>/dev/null; then
    git apply "$PATCH_FILE"
    echo "Patch applied cleanly."
else
    echo "WARNING: Patch does not apply cleanly (upstream code changed)."
    echo ""
    echo "Options:"
    echo "  1. Manual merge: diff the patch with current code"
    echo "  2. Force overwrite: copy patched files from backup"
    echo "  3. Ask Claude Code to re-create the customizations"
    echo ""
    echo "Patched file backups:"
    echo "  $PATCH_DIR/server.py.patched"
    echo "  $PATCH_DIR/routing.py.patched"
    echo ""

    # Save current patched versions if they exist
    if [ -f "$PATCH_DIR/server.py.patched" ]; then
        echo "To force overwrite with last known good patched files:"
        echo "  cp $PATCH_DIR/server.py.patched nadirclaw/server.py"
        echo "  cp $PATCH_DIR/routing.py.patched nadirclaw/routing.py"
    fi
    exit 1
fi

# Save patched copies for emergency restore
cp nadirclaw/routing.py "$PATCH_DIR/routing.py.patched"
cp nadirclaw/server.py "$PATCH_DIR/server.py.patched"

echo ""
echo "Done. Restart NadirClaw:"
echo "  systemctl --user restart nadirclaw"
echo "  curl localhost:8856/health"
