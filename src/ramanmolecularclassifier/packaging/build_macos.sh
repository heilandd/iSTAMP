#!/usr/bin/env bash
# Build a macOS .app bundle with PyInstaller.
#
# Usage:
#   cd 10_cross_platform_app
#   bash packaging/build_macos.sh
#
# The model checkpoint (best_model_full.pt) is large and license-restricted; it
# is NOT auto-bundled. Either copy it next to the .app and point RMC_CHECKPOINT
# at it, or uncomment the --add-data line below to embed a local copy.

set -euo pipefail
cd "$(dirname "$0")/.."

APP_NAME="RamanMolecularClassifier"
ENTRY="src/ramanmolecularclassifier/app.py"

python3 -m pip install --upgrade pyinstaller >/dev/null

pyinstaller \
  --noconfirm --clean --windowed \
  --name "$APP_NAME" \
  --osx-bundle-identifier "org.istamp.ramanmolecularclassifier" \
  --paths src \
  --collect-all torch_geometric \
  --collect-submodules ramanmolecularclassifier \
  --hidden-import scipy.interpolate \
  --hidden-import openpyxl \
  --hidden-import pydicom \
  --hidden-import tifffile \
  `# --add-data "/path/to/best_model_full.pt:."` \
  "$ENTRY"

echo "Built dist/$APP_NAME.app"
echo "Reminder: provide best_model_full.pt via RMC_CHECKPOINT or place beside the app."
