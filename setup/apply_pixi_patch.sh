#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMBINE_DIR="$PROJECT_ROOT/HiggsAnalysis/CombinedLimit"
PATCH_FILE="$SCRIPT_DIR/makefile_pixi.patch"

echo "🔧 Applying pixi patch to HiggsAnalysis/CombinedLimit Makefile..."

# Check if combine directory exists
if [[ ! -d "$COMBINE_DIR" ]]; then
    echo "❌ Error: $COMBINE_DIR not found"
    echo "💡 Please run this script after checking out combine:"
    echo "   git -c advice.detachedHead=false clone --depth 1 --branch v9.2.1 \\"
    echo "     https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit"
    exit 1
fi

# Check if patch file exists
if [[ ! -f "$PATCH_FILE" ]]; then
    echo "❌ Error: $PATCH_FILE not found"
    exit 1
fi

# Check if the patch has already been applied
if grep -q "PIXI" "$COMBINE_DIR/Makefile"; then
    echo "⚠️  Patch appears to already be applied"
    read -p "   Revert and reapply? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🔄 Reverting previous patch..."
        cd "$COMBINE_DIR"
        patch -R -p1 < "$PATCH_FILE" || echo "   (Revert may have failed, continuing...)"
        cd - > /dev/null
    else
        echo "✅ Patch already applied, nothing to do"
        exit 0
    fi
fi

# Apply the patch
echo "📝 Applying patch..."
cd "$COMBINE_DIR"
patch -p1 < "$PATCH_FILE"
cd - > /dev/null

echo "✅ Patch applied successfully!"
echo ""
echo "🚀 Now you can build with pixi using:"
echo "   pixi run build-combine"
echo ""
echo "💡 Or manually:"
echo "   cd $COMBINE_DIR"
echo "   pixi shell -c 'make PIXI=1 -j 8'"
