# Data Dictionary

This document describes the column meanings of the CSV files in this repository.

## data/scenario*_soc.csv (processed SOC time series)
- t_s: relative time in seconds (parsed from batterystats history; used for curve shape)
- soc_percent: state of charge (%) reported by the system history
- voltage_mV: battery voltage in mV (if present; forward-filled)
- temp_dC: battery temperature in 0.1°C (if present; forward-filled)
- plug: plug status (e.g., usb). Net-discharge segment is identified by plug being empty/NaN.
- status: charging status string (if present; forward-filled)

## data/tte_summary_with_conditions.csv
- scenario: scenario index (1..5)
- duration_min: experiment duration (minutes) computed from start/end epoch timestamps
- soc0_percent: initial SOC (%) of the net-discharge segment
- soc_min_percent: minimum SOC (%) observed in the net-discharge segment
- soc_drop_percent: SOC drop (%) = soc0 - soc_min
- rate_percent_per_h: discharge rate (%/h) computed from SOC drop and duration
- TTE_from_100_h: extrapolated time-to-empty from 100% SOC (hours), 100/rate
- TTE_from_SOC0_h: extrapolated time-to-empty from SOC0 (hours), SOC0/rate
- rate_upper_bound_percent_per_h, TTE_lower_bound_from_100_h: bounds used when SOC did not change during the measurement window (idle case)
- activity/screen_on/brightness_percent/network/gps/cpu_load/notes: manually recorded scenario conditions

## data_iphone_task2/raw/task2_real.csv
Scenario-level reference table used in Task 2 validation.
- scene: Standby/Video/Gaming/Navigation/Social
- tte_meas_h: reference/measured TTE in hours
- L: normalized brightness (0..1)
- U: average CPU load (0..1)
- net: OFF/WiFi/4G/5G
- gps: 0/1
- N_app: number of background apps
- T_env_c: ambient temperature (°C)

## data_raw_validation/
Sanitized raw logs and snapshots exported from batterystats/dumpsys, provided for reproducibility.
Quoted strings, package-like tokens, and uid-like tokens are redacted.
