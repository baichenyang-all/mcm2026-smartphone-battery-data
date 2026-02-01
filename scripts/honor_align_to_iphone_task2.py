import pandas as pd
from pathlib import Path

# ---------- Inputs ----------
HONOR_SUMMARY = Path("data/tte_summary_with_conditions.csv")
IPHONE_REAL   = Path("data_iphone_task2/raw/task2_real.csv")

# ---------- Battery energy scaling (optional) ----------
V_nom = 3.85
Q_honor_mAh = 5000.0
# 如果你们论文里 iPhone 也按 5000mAh（你同学脚本是 5000），就保持一致
Q_iphone_mAh = 5000.0
E_honor = Q_honor_mAh * V_nom / 1000.0
E_iphone = Q_iphone_mAh * V_nom / 1000.0
scale_E = E_iphone / E_honor

# ---------- Load data ----------
honor = pd.read_csv(HONOR_SUMMARY)
iphone = pd.read_csv(IPHONE_REAL)

def honor_tte(r):
    if pd.notna(r.get("TTE_from_100_h")):
        return float(r["TTE_from_100_h"])
    if pd.notna(r.get("TTE_lower_bound_from_100_h")):
        return float(r["TTE_lower_bound_from_100_h"])
    return None

honor["TTE_honor_h"] = honor.apply(honor_tte, axis=1)

# Honor scenario -> iPhone scene mapping
map_scene = {
    1: "Standby",
    2: "Video",
    3: "Social",      # 你们Honor的scenario3是刷网页/轻用，对应Social最合理
    4: "Navigation",
    5: "Gaming",
}
honor["scene"] = honor["scenario"].map(map_scene)

# Energy-scaled TTE
honor["TTE_honor_scaled_energy_h"] = honor["TTE_honor_h"] * scale_E

# Merge iPhone reference TTE
out = honor[["scenario","scene","duration_min","TTE_honor_h","TTE_honor_scaled_energy_h"]].copy()
out = out.merge(iphone[["scene","tte_meas_h"]], on="scene", how="left")
out = out.rename(columns={"tte_meas_h":"TTE_iphone_reference_h"})

# Scenario-wise calibration factor (makes Honor match iPhone exactly)
out["k_scene"] = out["TTE_iphone_reference_h"] / out["TTE_honor_h"]
out["TTE_honor_calibrated_to_iphone_h"] = out["k_scene"] * out["TTE_honor_h"]

# Save
Path("data_harmonized").mkdir(exist_ok=True)
out.to_csv("data_harmonized/honor_aligned_to_iphone_task2.csv", index=False, encoding="utf-8")
print("Wrote: data_harmonized/honor_aligned_to_iphone_task2.csv")
print(out)
