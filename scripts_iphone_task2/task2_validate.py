# -*- coding: utf-8 -*-
"""
问题二专用代码：模型验证与误差分析
功能：
1. 读取实测数据
2. 固定模型参数计算预测值（不做参数扰动）
3. 输出误差表格和整体统计
4. 生成论文图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ===================== 图表配置 =====================
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 10
rcParams['axes.linewidth'] = 1.2
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['figure.dpi'] = 300


class TTECalculator:
    """核心电池TTE计算引擎（问题二：固定参数版本）"""
    
    def __init__(self, cycle_count=0, dt=1.0):
        self.dt = dt
        
        # ===== 电池物理参数（固定，不扰动） =====
        self.Q_design_mAh = 5000
        self.V_cutoff = 3.0
        self.SOH_Q = 1.0 - 0.20 * np.sqrt(cycle_count / 1600.0)
        self.Q_bat = self.Q_design_mAh * self.SOH_Q * 3.6  # Coulomb
        
        self.R_ref = 0.12 * (1.0 + 0.0005 * cycle_count)
        self.T_ref = 298.15
        self.E_a_k = 3500.0
        
        self.m_Cp = 0.245 * 900.0
        self.hA = 1.5
        
        # ===== 功耗模型参数（固定，不扰动） =====
        self.P_leak = 0.02
        self.P_wake = 0.01
        self.alpha_scr = 2.8
        self.gamma = 2.2
        self.P_cpu_idle = 0.05
        self.beta_cpu = 4.8
        self.P_gps = 0.3
        self.P_net_map = {"OFF": 0.0, "WiFi": 0.15, "4G": 0.4, "5G": 0.8}
    
    def _get_ocv(self, s):
        s = np.clip(s, 0.001, 1.0)
        return (3.3 + 0.8*s - 0.2*s**2 + 0.3*s**3) - (0.6*np.exp(-15*s))
    
    def _get_rint(self, s, T):
        temp_factor = np.exp(self.E_a_k * (1.0/T - 1.0/self.T_ref))
        diff_factor = 1.0 + 1.5*np.exp(-15.0*s)
        return self.R_ref * temp_factor * diff_factor
    
    def calculate_power(self, L, U, net, gps, N_app):
        """计算各组件功耗"""
        p_base = self.P_leak + min(int(N_app), 20) * self.P_wake
        p_scr = self.alpha_scr * (float(L) ** self.gamma)
        p_cpu = self.P_cpu_idle + self.beta_cpu * float(U)
        p_net = self.P_net_map.get(str(net), 0.0)
        p_gps = self.P_gps if int(gps) == 1 else 0.0
        p_total = p_base + p_scr + p_cpu + p_net + p_gps
        return {
            "Base": p_base, "Screen": p_scr, "CPU": p_cpu,
            "Network": p_net, "GPS": p_gps, "Total": p_total
        }
    
    def run_tte_minutes(self, initial_soc, T_env_c, L, U, net, gps, N_app):
        """运行仿真，返回TTE（分钟）"""
        soc = float(initial_soc)
        if soc <= 0:
            return 0.0
        
        t = 0.0
        T = float(T_env_c) + 273.15
        T_env_K = float(T_env_c) + 273.15
        
        while soc > 0:
            P_req = self.calculate_power(L, U, net, gps, N_app)["Total"]
            R_int = self._get_rint(soc, T)
            V_ocv = self._get_ocv(soc)
            
            delta = V_ocv**2 - 4.0 * P_req * R_int
            if delta < 0:
                break
            
            V_term = (V_ocv + np.sqrt(delta)) / 2.0
            if V_term < self.V_cutoff:
                break
            
            I = P_req / V_term
            soc += (-I / self.Q_bat) * self.dt
            
            Q_gen = (I**2 * R_int) + (0.95 * P_req)
            Q_diss = self.hA * (T - T_env_K)
            T += (Q_gen - Q_diss) / self.m_Cp * self.dt
            
            t += self.dt
        
        return t / 60.0


def calculate_rmse(pred, meas):
    """计算RMSE"""
    pred = np.array(pred, dtype=float)
    meas = np.array(meas, dtype=float)
    return float(np.sqrt(np.mean((pred - meas) ** 2)))


def calculate_r2(pred, meas):
    """计算R²"""
    pred = np.array(pred, dtype=float)
    meas = np.array(meas, dtype=float)
    ss_res = np.sum((pred - meas) ** 2)
    ss_tot = np.sum((meas - np.mean(meas)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def run_validation():
    """
    核心功能：运行验证分析
    """
    # 读取实测数据
    try:
        df = pd.read_csv("task2_real.csv")
    except FileNotFoundError:
        print("❌ 错误：找不到 task2_real.csv 文件！")
        print("请在代码同目录创建该文件，内容如下：")
        print("-" * 50)
        print("scene,tte_meas_h,L,U,net,gps,N_app,T_env_c")
        print("Standby,48.0,0.10,0.05,OFF,0,2,25")
        print("Video,4.0,0.80,0.35,WiFi,0,3,25")
        print("Gaming,2.5,0.90,0.70,WiFi,0,2,25")
        print("Navigation,3.5,0.60,0.40,5G,1,3,17")
        print("Social,6.0,0.50,0.30,4G,0,3,25")
        return None, None
    
    calc = TTECalculator(cycle_count=0, dt=1.0)
    
    # 计算预测值和功耗
    preds_h = []
    powers = []
    power_breakdowns = []
    
    for _, row in df.iterrows():
        tte_min = calc.run_tte_minutes(
            initial_soc=1.0,
            T_env_c=row["T_env_c"],
            L=row["L"], U=row["U"], 
            net=row["net"], gps=row["gps"], N_app=row["N_app"]
        )
        tte_h = tte_min / 60.0
        preds_h.append(tte_h)
        
        power = calc.calculate_power(
            row["L"], row["U"], row["net"], row["gps"], row["N_app"]
        )
        powers.append(power["Total"])
        power_breakdowns.append(power)
    
    # 添加到DataFrame
    df["tte_pred_h"] = preds_h
    df["tte_pred_min"] = [h * 60 for h in preds_h]
    df["abs_err_h"] = df["tte_pred_h"] - df["tte_meas_h"]
    df["rel_err_pct"] = (df["abs_err_h"].abs() / df["tte_meas_h"]) * 100
    df["bias"] = np.where(df["abs_err_h"] >= 0, "高估", "低估")
    df["P_total_W"] = powers
    
    # 计算整体统计
    rmse = calculate_rmse(df["tte_pred_h"], df["tte_meas_h"])
    r2 = calculate_r2(df["tte_pred_h"], df["tte_meas_h"])
    
    # ===== 打印结果 =====
    print("\n" + "=" * 80)
    print("【问题二】模型验证：预测 vs 实测数据")
    print("=" * 80)
    
    print(f"\n{'场景':<12} {'实测(h)':<10} {'预测(h)':<10} {'误差(%)':<10} {'偏差':<8} {'功耗(W)':<10}")
    print("-" * 70)
    
    for _, row in df.iterrows():
        print(f"{row['scene']:<12} {row['tte_meas_h']:<10.2f} {row['tte_pred_h']:<10.2f} "
              f"{row['rel_err_pct']:<10.2f} {row['bias']:<8} {row['P_total_W']:<10.2f}")
    
    print("-" * 70)
    print(f"整体统计: RMSE = {rmse:.3f} h, R² = {r2:.4f}")
    print("=" * 80)
    
    # 打印功耗分解
    print("\n【功耗分解】")
    print("-" * 70)
    for i, row in df.iterrows():
        pb = power_breakdowns[i]
        print(f"{row['scene']}: 总功耗 {pb['Total']:.2f}W")
        print(f"  Base={pb['Base']:.3f}W, Screen={pb['Screen']:.2f}W, "
              f"CPU={pb['CPU']:.2f}W, Net={pb['Network']:.2f}W, GPS={pb['GPS']:.2f}W")
    
    # 保存结果
    df.to_csv("task2_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n✅ 结果已保存到 task2_results.csv")
    
    return df, {"RMSE": rmse, "R2": r2, "power_breakdowns": power_breakdowns}


def plot_validation_results(df):
    """生成验证结果图表"""
    if df is None:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ===== 图1：预测 vs 实测对比 =====
    names = df["scene"].tolist()
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, df["tte_pred_h"], width, 
                     label='Model Prediction', color='#4472C4', edgecolor='black')
    bars2 = ax1.bar(x + width/2, df["tte_meas_h"], width,
                     label='Measured Data', color='#ED7D31', edgecolor='black')
    
    ax1.set_ylabel('Time-to-Empty (hours)', fontsize=11)
    ax1.set_title('(a) Prediction vs Measured Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标注
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    
    # ===== 图2：误差柱状图 =====
    errors = df["rel_err_pct"].tolist()
    colors = ['#70AD47' if e < 10 else '#FFC000' if e < 15 else '#C00000' for e in errors]
    
    ax2.bar(names, errors, color=colors, edgecolor='black')
    ax2.axhline(y=10, color='#FFC000', linestyle='--', linewidth=2, label='10% Threshold')
    ax2.axhline(y=15, color='#C00000', linestyle='--', linewidth=2, label='15% Threshold')
    
    ax2.set_ylabel('Relative Error (%)', fontsize=11)
    ax2.set_title('(b) Prediction Error by Scenario', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 20)
    
    # 添加误差数值标注
    for i, (name, err) in enumerate(zip(names, errors)):
        ax2.annotate(f'{err:.1f}%', xy=(i, err), xytext=(0, 3),
                     textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('Fig_Task2_Validation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 图表已保存: Fig_Task2_Validation.png")


def plot_tte_vs_soc(df):
    """生成TTE随SOC变化图"""
    if df is None:
        return
    
    calc = TTECalculator(cycle_count=0, dt=1.0)
    initial_socs = [1.00, 0.75, 0.50, 0.25]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#4472C4', '#ED7D31', '#70AD47', '#FFC000', '#7030A0']
    
    for i, row in df.iterrows():
        tte_vals = []
        for soc in initial_socs:
            tte_min = calc.run_tte_minutes(
                initial_soc=soc,
                T_env_c=row["T_env_c"],
                L=row["L"], U=row["U"],
                net=row["net"], gps=row["gps"], N_app=row["N_app"]
            )
            tte_vals.append(tte_min)
        
        ax.plot(initial_socs, tte_vals, 'o-', color=colors[i],
                linewidth=2, markersize=8, label=row["scene"])
    
    ax.set_xlabel('Initial SOC', fontsize=12)
    ax.set_ylabel('Time-to-Empty (minutes)', fontsize=12)
    ax.set_title('TTE vs Initial SOC for Different Scenarios', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xticks(initial_socs)
    
    plt.tight_layout()
    plt.savefig('Fig_Task2_TTE_vs_SOC.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 图表已保存: Fig_Task2_TTE_vs_SOC.png")


def print_scenario_comparison(df, stats):
    """打印场景差异对比（用于7.4节）"""
    if df is None:
        return
    
    print("\n" + "=" * 80)
    print("【问题二】场景差异分析：续航衰减 & 功耗倍率")
    print("=" * 80)
    
    # 以待机为基准
    baseline_tte = df[df["scene"] == "Standby"]["tte_pred_min"].values[0]
    baseline_power = df[df["scene"] == "Standby"]["P_total_W"].values[0]
    
    print(f"\n{'场景':<12} {'TTE(min)':<12} {'功耗(W)':<10} {'衰减率':<12} {'功耗倍率':<10}")
    print("-" * 70)
    
    for _, row in df.iterrows():
        if row["scene"] == "Standby":
            decay = "--"
            ratio = "1.0x"
        else:
            decay = f"{(baseline_tte - row['tte_pred_min']) / baseline_tte * 100:.1f}%"
            ratio = f"{row['P_total_W'] / baseline_power:.1f}x"
        
        print(f"{row['scene']:<12} {row['tte_pred_min']:<12.1f} {row['P_total_W']:<10.2f} "
              f"{decay:<12} {ratio:<10}")
    
    print("=" * 80)


# ===================== 主程序 =====================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  问题二 (Task 2) 完整分析流程")
    print("  注意：本代码不进行模型参数扰动（那是问题三的内容）")
    print("=" * 80)
    
    # Step 1: 运行验证分析
    print("\n>>> Step 1: 运行模型验证...")
    df, stats = run_validation()
    
    if df is not None:
        # Step 2: 场景对比
        print("\n>>> Step 2: 场景差异分析...")
        print_scenario_comparison(df, stats)
        
        # Step 3: 生成图表
        print("\n>>> Step 3: 生成论文图表...")
        plot_validation_results(df)
        plot_tte_vs_soc(df)
        
        print("\n" + "=" * 80)
        print("  ✅ 问题二分析完成！")
        print("  生成文件：")
        print("    - task2_results.csv（数据表格）")
        print("    - Fig_Task2_Validation.png（验证对比图）")
        print("    - Fig_Task2_TTE_vs_SOC.png（TTE vs SOC图）")
        print("=" * 80)