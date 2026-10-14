# Contributing to iSTAMP

Thanks for your interest in improving iSTAMP.

## Workflow
1. Fork and create a feature branch (`feat/<short-name>`).
2. Set up the environment: `conda env create -f environment.yml` (or `pip install -r requirements.txt`).
3. Make focused changes with clear commit messages.
4. Add/adjust tests under `tests/` and run `python -m pytest -q` (use `QT_QPA_PLATFORM=offscreen` for headless GUI tests).
5. Run `python -m py_compile` on changed modules and keep style consistent with surrounding code.
6. Open a pull request describing the change and its scientific/clinical implications.

## Scientific integrity
- Never commit fabricated results. Evaluation must use patient-disjoint splits.
- Document assumptions and data provenance.
- This software is **research use only** — do not add clinical/diagnostic claims.

## Reporting issues
Open a GitHub issue with a minimal reproducible example and environment details.
