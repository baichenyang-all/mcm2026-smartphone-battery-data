import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("data_validation")
RAW = ROOT / "raw"
PRO = ROOT / "processed"

scenarios = {
    1: {"soc": PRO/"scenario1_soc.csv", "start": RAW/"scenario1_start_epoch.txt", "end": RAW/"scenario1_end_epoch.txt", "name": "Idle"},
    2: {"soc": PRO/"scenario2_soc.csv", "start": RAW/"scenario2_start_epoch.txt", "end": RAW/"scenario2_end_epoch.txt", "name": "Video (bright)"},
    3: {"soc": PRO/"scenario3_soc.csv", "start": RAW/"scenario3_start_epoch.txt", "end": RAW/"scenario3_end_epoch.txt", "name": "Light use (dim)"},
    4: {"soc": PRO/"scenario4_soc.csv", "start": RAW/"scenario4_start_epoch.txt", "end": RAW/"scenario4_end_epoch.txt", "name": "GPS/Map"},
    5: {"soc": PRO/"scenario5_soc.csv", "start": RAW/"scenario5_start_epoch.txt", "end": RAW/"scenario5_end_epoch.txt", "name": "Gaming"},
}

def read_epoch(p: Path) -> int:
    # try utf-8
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore").strip()
        if txt.isdigit():
            return int(txt)
    except Exception:
        pass
    # try utf-16 (older files created by PowerShell redirection)
    try:
        txt2 = p.read_text(encoding="utf-16").strip()
        if txt2.isdigit():
            return int(txt2)
    except Exception:
        pass
    # fallback: digits from bytes
    b = p.read_bytes()
    m = re.findall(rb"\d+", b)
    if not m:
        raise ValueError(f"Cannot parse epoch from {p}")
    return int(m[0])

def main():
    rows = []
    soc_series = {}

    for k, meta in scenarios.items():
        df = pd.read_csv(meta["soc"])
        # net discharge segment: plug is NaN (unplugged)
        if "plug" in df.columns:
            d = df[df["plug"].isna()].copy()
        else:
            d = df.copy()

        soc0 = float(d["soc_percent"].iloc[0])
        socmin = float(d["soc_percent"].min())
        drop = soc0 - socmin

        s = read_epoch(meta["start"])
        e = read_epoch(meta["end"])
        dur_s = e - s
        dur_h = dur_s / 3600.0
        dur_min = dur_s / 60.0

        # For SOC curve shape plot
        if "t_s" in d.columns:
            t = d["t_s"].astype(float).to_numpy()
            t = t - t.min()
            soc_series[k] = (t/60.0, d["soc_percent"].astype(float).to_numpy(), meta["name"])  # minutes

        if drop <= 0:
            rate_ub = 1.0 / dur_h if dur_h > 0 else None
            tte_lb_100 = 100.0 / rate_ub if rate_ub else None
            rows.append({
                "scenario": k,
                "name": meta["name"],
                "duration_min": dur_min,
                "soc0_percent": soc0,
                "soc_min_percent": socmin,
                "soc_drop_percent": drop,
                "rate_percent_per_h": None,
                "TTE_from_100_h": None,
                "TTE_from_SOC0_h": None,
                "rate_upper_bound_percent_per_h": rate_ub,
                "TTE_lower_bound_from_100_h": tte_lb_100,
            })
        else:
            rate = drop / dur_h
            rows.append({
                "scenario": k,
                "name": meta["name"],
                "duration_min": dur_min,
                "soc0_percent": soc0,
                "soc_min_percent": socmin,
                "soc_drop_percent": drop,
                "rate_percent_per_h": rate,
                "TTE_from_100_h": 100.0 / rate,
                "TTE_from_SOC0_h": soc0 / rate,
                "rate_upper_bound_percent_per_h": None,
                "TTE_lower_bound_from_100_h": None,
            })

    out = pd.DataFrame(rows).sort_values("scenario")

    # Merge manual scenario conditions
    cond_path = ROOT / "scenario_conditions.csv"
    if cond_path.exists():
        cond = pd.read_csv(cond_path)
        out = out.merge(cond, on="scenario", how="left")
    else:
        print("WARNING: scenario_conditions.csv not found:", cond_path)

    ROOT.mkdir(parents=True, exist_ok=True)

    # --- write CSV ---
    csv_path = ROOT / "tte_summary_with_conditions.csv"
    out.to_csv(csv_path, index=False, encoding="utf-8")
    print("Wrote:", csv_path)

    # --- plot TTE bar chart ---
    plt.figure(figsize=(10,5))
    labels = []
    values = []
    colors = []
    for _, r in out.iterrows():
        b = "" if pd.isna(r.get("brightness_percent")) else f"{int(r['brightness_percent'])}%"
        gps = r.get("gps", "")
        cpu = r.get("cpu_load", "")
        labels.append(f"S{int(r['scenario'])}\n{r['name']}\n{b} GPS:{gps} CPU:{cpu}")

        if pd.notna(r.get("TTE_from_100_h")):
            values.append(float(r["TTE_from_100_h"]))
            colors.append("#4C78A8")
        else:
            values.append(float(r["TTE_lower_bound_from_100_h"]))
            colors.append("#F58518")

    plt.bar(labels, values, color=colors)
    plt.ylabel("TTE (hours)  [from 100% extrapolation]\n(Idle shown as lower bound)")
    plt.title("Measured TTE with scenario conditions")
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig1 = ROOT / "tte_summary_with_conditions.png"
    plt.savefig(fig1, dpi=200)
    plt.close()
    print("Wrote:", fig1)

    # --- plot SOC curves overlay (shape only) ---
    plt.figure(figsize=(10,5))
    for k in sorted(soc_series.keys()):
        t_min, soc, name = soc_series[k]
        plt.plot(t_min, soc, linewidth=2, label=f"S{k} {name}")
    plt.xlabel("Time (min)  [history-relative; shape only]")
    plt.ylabel("SOC (%)")
    plt.title("SOC(t) curves for net-discharge segments (shape comparison)")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2 = ROOT / "soc_curves.png"
    plt.savefig(fig2, dpi=200)
    plt.close()
    print("Wrote:", fig2)

    print("\nPreview:\n", out)

if __name__ == "__main__":
    main()
