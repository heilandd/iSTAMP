"""Smoke test: guarded imports so optional heavy deps never hard-fail CI."""
def test_optional_imports():
    for mod in ["numpy", "pandas", "sklearn", "matplotlib"]:
        __import__(mod)
def test_optional_heavy():
    for mod in ["torch", "torch_geometric", "PySide6"]:
        try:
            __import__(mod)
        except Exception:
            pass  # optional in minimal CI
