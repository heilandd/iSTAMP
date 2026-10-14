#!/usr/bin/env bash
# Build a Linux executable (PyInstaller one-folder) and, optionally, an AppImage.
#
# Usage:
#   cd 10_cross_platform_app
#   bash packaging/build_linux.sh           # PyInstaller one-folder
#   APPIMAGE=1 bash packaging/build_linux.sh # also try to wrap as AppImage
#
# The model checkpoint (best_model_full.pt) is NOT auto-bundled. Provide it via
# RMC_CHECKPOINT or place it next to the executable.

set -euo pipefail
cd "$(dirname "$0")/.."

APP_NAME="RamanMolecularClassifier"
ENTRY="src/ramanmolecularclassifier/app.py"

python3 -m pip install --upgrade pyinstaller >/dev/null

pyinstaller \
  --noconfirm --clean --windowed \
  --name "$APP_NAME" \
  --paths src \
  --collect-all torch_geometric \
  --collect-submodules ramanmolecularclassifier \
  --hidden-import scipy.interpolate \
  --hidden-import openpyxl \
  --hidden-import pydicom \
  --hidden-import tifffile \
  "$ENTRY"

echo "Built dist/$APP_NAME/$APP_NAME"

if [ "${APPIMAGE:-0}" = "1" ]; then
  echo "Attempting AppImage packaging ..."
  # Requires linuxdeploy + appimagetool on PATH.
  if ! command -v appimagetool >/dev/null; then
    echo "appimagetool not found; skipping AppImage. Install it from"
    echo "  https://github.com/AppImage/AppImageKit/releases"
    exit 0
  fi
  APPDIR="dist/${APP_NAME}.AppDir"
  rm -rf "$APPDIR"
  mkdir -p "$APPDIR/usr/bin"
  cp -r "dist/$APP_NAME/." "$APPDIR/usr/bin/"
  cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/bin/RamanMolecularClassifier" "$@"
EOF
  chmod +x "$APPDIR/AppRun"
  cat > "$APPDIR/RamanMolecularClassifier.desktop" <<'EOF'
[Desktop Entry]
Name=RamanMolecularClassifier
Exec=RamanMolecularClassifier
Icon=ramanmolecularclassifier
Type=Application
Categories=Science;
EOF
  # A placeholder icon is required by appimagetool.
  touch "$APPDIR/ramanmolecularclassifier.png"
  appimagetool "$APPDIR" "dist/${APP_NAME}-x86_64.AppImage"
  echo "Built dist/${APP_NAME}-x86_64.AppImage"
fi

echo "Reminder: provide best_model_full.pt via RMC_CHECKPOINT or place beside the executable."
