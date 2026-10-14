# Packaging RamanMolecularClassifier (iSTAMP Desktop)

This folder contains scripts to build standalone desktop bundles with
[PyInstaller](https://pyinstaller.org) for macOS, Windows and Linux.

| Platform | Script | Output |
|----------|--------|--------|
| macOS    | `build_macos.sh`   | `dist/RamanMolecularClassifier.app` |
| Windows  | `build_windows.bat`| `dist/RamanMolecularClassifier/RamanMolecularClassifier.exe` |
| Linux    | `build_linux.sh`   | `dist/RamanMolecularClassifier/…` (+ optional AppImage) |

## Prerequisites

* Python 3.10+ with the project dependencies installed (`pip install -r ../requirements.txt`).
* PyInstaller (the scripts `pip install` it for you).
* For the Linux AppImage path: `appimagetool` on `PATH` and `APPIMAGE=1`.

## Building

```bash
cd ..                       # the 10_cross_platform_app folder
bash packaging/build_macos.sh        # macOS
packaging\build_windows.bat          # Windows (cmd)
bash packaging/build_linux.sh        # Linux
APPIMAGE=1 bash packaging/build_linux.sh   # Linux + AppImage
```

## Model checkpoint

`best_model_full.pt` (~15 MB) is **not auto-bundled** by these scripts because
it may be license-restricted. Two options:

1. Place a copy next to the built executable and set the environment variable
   `RMC_CHECKPOINT=/path/to/best_model_full.pt`.
2. Embed it at build time by uncommenting the `--add-data` line in the relevant
   script (use `;` as the separator on Windows, `:` elsewhere). The app then
   discovers it via `RMC_CHECKPOINT` or the default path.

Patient SRH images and graph files are **never** bundled.

## Notes

* `--collect-all torch_geometric` and the `--hidden-import` lines ensure the
  scientific stack (scipy interpolation, openpyxl, pydicom, tifffile) is picked
  up by PyInstaller's static analysis.
* `app_metadata.json` records the canonical app identity / model configuration.
* See `icon_placeholder.txt` for icon guidance.
