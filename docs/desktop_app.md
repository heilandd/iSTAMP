# Desktop application
RamanMolecularClassifier (iSTAMP Desktop), PySide6. Panels: image viewer (zoom/pan), metadata,
prediction (class-probability bar chart + confidence), heatmap viewer (spatial subtype/confidence
overlays, low-confidence masking), report viewer, log panel. Packaging scripts for macOS `.app`,
Windows `.exe`, and Linux AppImage are in `packaging/`. Run headless tests with
`QT_QPA_PLATFORM=offscreen python -m pytest -q`.
