# References & Data Sources

## Android Power Profile
- Source: LineageOS device tree for Google Pixel 5 (redfin), file `power_profile.xml`.
- We extracted parameters into `power_profile_redfin.csv`.
- Please include the repository URL and commit hash used in the report.

## PyBaMM OCV Curve
- PyBaMM project (BSD-3-Clause): https://github.com/pybamm-team/PyBaMM
- Parameter set used is recorded in `ocv_soc_curve.csv` (column `source_param_set`).

## Battery Aging Dataset
- Capacity vs cycle extracted from a public cycling dataset (MATR.io / Severson et al., 2019).
- We provide only processed curve data and scripts; raw large files should be downloaded from the official source.
