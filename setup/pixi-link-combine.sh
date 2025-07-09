#!/usr/bin/env bash

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Check input
if [[ -z "$1" ]]; then
  echo "❌ Error: Missing required argument"
  echo "📖 Usage: $0 /path/to/HiggsAnalysis/CombinedLimit"
  echo "💡 Example: $0 HiggsAnalysis/CombinedLimit"
  exit 1
fi

# Resolve absolute path
echo "🔍 Resolving path: $1"
# If path is relative, resolve from project root
if [[ "$1" = /* ]]; then
  TARGET_DIR="$(cd "$1" && pwd -P)"
else
  TARGET_DIR="$(cd "$PROJECT_ROOT/$1" && pwd -P)"
fi
echo "📍 Resolved to: $TARGET_DIR"

# Sanity check
if [[ ! -d "$TARGET_DIR" ]]; then
  echo "❌ Error: Directory '$TARGET_DIR' does not exist."
  echo "💡 Please check the path and ensure HiggsAnalysis/CombinedLimit is properly installed."
  exit 1
fi

echo "✅ Directory exists and is accessible"

# Check if template exists
TEMPLATE_FILE="$SCRIPT_DIR/pixi.toml.template"
if [[ ! -f "$TEMPLATE_FILE" ]]; then
  echo "❌ Error: pixi.toml.template not found at $TEMPLATE_FILE"
  echo "💡 Please ensure the setup directory contains pixi.toml.template"
  exit 1
fi

# Check if pixi.toml already exists in project root
PIXI_TOML="$PROJECT_ROOT/pixi.toml"
if [[ -f "$PIXI_TOML" ]]; then
  echo "⚠️  pixi.toml already exists. This will overwrite it with the template + Combine paths."
  read -p "   Continue? (y/N): " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Aborted by user"
    exit 1
  fi
fi

# Create pixi.toml from template
echo "⚙️  Creating pixi.toml from template..."
cp "$TEMPLATE_FILE" "$PIXI_TOML"

# Add combine paths to pixi.toml
echo "🔧 Adding Combine paths to pixi.toml..."

# Update the existing [activation] env section to include combine paths
# Use @ as delimiter to avoid conflicts with forward slashes in paths
sed -i 's@env = { PIXI_PREFIX = "\$CONDA_PREFIX" }@env = { PIXI_PREFIX = "\$CONDA_PREFIX", PATH = "'"$TARGET_DIR"'/build/bin:\$PATH", LD_LIBRARY_PATH = "'"$TARGET_DIR"'/build/lib:\$LD_LIBRARY_PATH", PYTHONPATH = "'"$TARGET_DIR"'/build/lib/python:\$PYTHONPATH" }@' "$PIXI_TOML"

echo "✅ Successfully linked HiggsAnalysis/CombinedLimit to pixi environment"
echo "📂 Configuration saved to: $PIXI_TOML (local copy, gitignored)"
echo ""
echo "🚀 Next steps:"
echo "   1. Run: pixi shell"
echo "   2. Test combine tools are available: combine --help"
echo "   3. Your analysis scripts can now use combine tools directly"
echo "   4. Build combine: pixi run build-combine"
echo ""
echo "ℹ️  The following paths have been added to your pixi environment:"
echo "   • PIXI_PREFIX: \$CONDA_PREFIX (points to pixi environment)"
echo "   • PATH: $TARGET_DIR/build/bin"
echo "   • LD_LIBRARY_PATH: $TARGET_DIR/build/lib"
echo "   • PYTHONPATH: $TARGET_DIR/build/lib/python"
echo ""
echo "💡 Note: pixi.toml is now customized for your system."
echo "   Make sure 'pixi.toml' is in your .gitignore file!"
