# Smartphone Battery Modeling – Open Data Pack (MCM/ICM 2026)

This repository contains processed, small-size datasets and scripts used/created for a continuous-time smartphone battery model.

## Contents
- `data/`
  - `power_profile_redfin.csv`: Android power profile parameters (Pixel 5 / redfin), extracted from open-source device tree.
  - `ocv_soc_curve.csv`: OCV–SOC curve exported from PyBaMM parameter set.
  - `aging_capacity_vs_cycle.csv`: Capacity fade vs cycle curve extracted from public cycling dataset (see references).
  - `scenario*_soc.csv`: SOC time series extracted from Android/HarmonyOS battery history logs (net-discharge segments).
  - `tte_summary_with_conditions.csv`: TTE summary (linear extrapolation) with scenario conditions.
  - `tte_summary_with_conditions.png`, `soc_curves.png`: plots generated from the summary script.

- `scripts/`: scripts to reproduce processed datasets and plots.

## License
- Our original processed datasets (scenario SOC curves, TTE summary/plots, etc.) are released under CC BY 4.0 unless noted.
- Third-party sources are cited in `docs/references.md` and retain their original licenses.

## Reproducibility
Run:
- `python scripts/make_tte_report.py`
to regenerate the summary CSV and plots from scenario SOC files.

## Notes on Privacy
We removed raw event strings from public SOC CSV files to avoid leaking app/package names. Only SOC/voltage/temp/plug/status and time are kept.
