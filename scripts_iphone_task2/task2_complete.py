# -*- coding: utf-8 -*-
"""
问题二完整分析代码
生成所有论文需要的图片和数据
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
rcParams['legend.frameon'] = False

# MCM浅色系配色
MCM_COLORS = ['#8FB9A8', '#C8B6E2', '#FFD966', '#F4BFBF', '#B4C7E7']


class TTECalculator:
    """核心电池TTE计算引擎"""
    
    def __init__(self, cycle_count=0, dt=1.0):
        self.dt = dt
        
        # 电池物理参数
        self.Q_design_mAh = 5000
        self.V_cutoff = 3.0
        self.SOH_Q = 1.0 - 0.20 * np.sqrt(cycle_count / 1600.0)
        self.Q_bat = self.Q_design_mAh * self.SOH_Q * 3.6
        
        self.R_ref = 0.12 * (1.0 + 0.0005 * cycle_count)
        self.T_ref = 298.15
        self.E_a_k = 3500.0
        
        self.m_Cp = 0.245 * 900.0
        self.hA = 1.5
        
        # 功耗模型参数
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


# ===================== 场景定义 =====================
SCENARIOS = [
    {'name': 'Standby', 'name_cn': '待机', 'tte_meas_h': 48.0,
     'L': 0.10, 'U': 0.05, 'net': 'OFF', 'gps': 0, 'N_app': 2, 'T_env_c': 25},
    {'name': 'Video', 'name_cn': '视频播放', 'tte_meas_h': 4.0,
     'L': 0.80, 'U': 0.35, 'net': 'WiFi', 'gps': 0, 'N_app': 3, 'T_env_c': 25},
    {'name': 'Gaming', 'name_cn': '游戏', 'tte_meas_h': 2.5,
     'L': 0.90, 'U': 0.70, 'net': 'WiFi', 'gps': 0, 'N_app': 2, 'T_env_c': 25},
    {'name': 'Navigation', 'name_cn': '导航', 'tte_meas_h': 3.5,
     'L': 0.60, 'U': 0.40, 'net': '5G', 'gps': 1, 'N_app': 3, 'T_env_c': 17},
    {'name': 'Social', 'name_cn': '社交浏览', 'tte_meas_h': 6.0,
     'L': 0.50, 'U': 0.30, 'net': '4G', 'gps': 0, 'N_app': 3, 'T_env_c': 25},
]


def run_complete_analysis():
    """运行完整分析"""
    calc = TTECalculator(cycle_count=0, dt=1.0)
    
    results = []
    for s in SCENARIOS:
        # 计算预测TTE
        tte_min = calc.run_tte_minutes(
            initial_soc=1.0, T_env_c=s['T_env_c'],
            L=s['L'], U=s['U'], net=s['net'], gps=s['gps'], N_app=s['N_app']
        )
        tte_h = tte_min / 60.0
        
        # 计算功耗
        power = calc.calculate_power(s['L'], s['U'], s['net'], s['gps'], s['N_app'])
        
        # 计算误差
        rel_err = abs(tte_h - s['tte_meas_h']) / s['tte_meas_h'] * 100
        
        results.append({
            'name': s['name'],
            'name_cn': s['name_cn'],
            'tte_meas_h': s['tte_meas_h'],
            'tte_pred_h': tte_h,
            'tte_pred_min': tte_min,
            'rel_err': rel_err,
            'power_total': power['Total'],
            'power_breakdown': power,
            'params': s
        })
    
    # 计算整体统计
    pred = np.array([r['tte_pred_h'] for r in results])
    meas = np.array([r['tte_meas_h'] for r in results])
    rmse = np.sqrt(np.mean((pred - meas)**2))
    ss_res = np.sum((pred - meas)**2)
    ss_tot = np.sum((meas - np.mean(meas))**2)
    r2 = 1 - ss_res / ss_tot
    
    # 打印结果
    print("\n" + "="*80)
    print("【问题二】完整分析结果")
    print("="*80)
    
    print(f"\n{'场景':<12} {'实测(h)':<10} {'预测(h)':<10} {'误差(%)':<10} {'功耗(W)':<10}")
    print("-"*60)
    for r in results:
        print(f"{r['name']:<12} {r['tte_meas_h']:<10.2f} {r['tte_pred_h']:<10.2f} "
              f"{r['rel_err']:<10.2f} {r['power_total']:<10.2f}")
    print("-"*60)
    print(f"整体统计: RMSE = {rmse:.2f} h, R² = {r2:.4f}")
    
    return results, {'rmse': rmse, 'r2': r2}


def plot_validation_comparison(results):
    """图1：预测vs实测对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    names = [r['name'] for r in results]
    pred = [r['tte_pred_h'] for r in results]
    meas = [r['tte_meas_h'] for r in results]
    errors = [r['rel_err'] for r in results]
    
    x = np.arange(len(names))
    width = 0.35
    
    # 图1a：预测vs实测
    bars1 = ax1.bar(x - width/2, pred, width, label='Model Prediction', 
                     color='#4472C4', edgecolor='black')
    bars2 = ax1.bar(x + width/2, meas, width, label='Measured Data',
                     color='#ED7D31', edgecolor='black')
    
    ax1.set_ylabel('Time-to-Empty (hours)', fontsize=11)
    ax1.set_title('(a) Prediction vs Measured Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    
    # 图1b：误差
    colors = ['#70AD47' if e < 10 else '#FFC000' if e < 15 else '#C00000' for e in errors]
    bars = ax2.bar(names, errors, color=colors, edgecolor='black')
    ax2.axhline(y=10, color='#FFC000', linestyle='--', linewidth=2, label='10% Threshold')
    ax2.axhline(y=15, color='#C00000', linestyle='--', linewidth=2, label='15% Threshold')
    
    ax2.set_ylabel('Relative Error (%)', fontsize=11)
    ax2.set_title('(b) Prediction Error by Scenario', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 20)
    
    for i, (bar, err) in enumerate(zip(bars, errors)):
        ax2.annotate(f'{err:.1f}%', xy=(bar.get_x() + bar.get_width()/2, err),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('Fig_Task2_Validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 已生成: Fig_Task2_Validation.png")


def plot_tte_vs_soc(results):
    """图2：TTE随SOC变化"""
    calc = TTECalculator(cycle_count=0, dt=1.0)
    initial_socs = [1.00, 0.75, 0.50, 0.25]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#4472C4', '#ED7D31', '#70AD47', '#FFC000', '#7030A0']
    
    for i, r in enumerate(results):
        tte_vals = []
        for soc in initial_socs:
            tte_min = calc.run_tte_minutes(
                initial_soc=soc, T_env_c=r['params']['T_env_c'],
                L=r['params']['L'], U=r['params']['U'],
                net=r['params']['net'], gps=r['params']['gps'], N_app=r['params']['N_app']
            )
            tte_vals.append(tte_min)
        
        ax.plot(initial_socs, tte_vals, 'o-', color=colors[i],
                linewidth=2, markersize=8, label=r['name'])
    
    ax.set_xlabel('Initial SOC', fontsize=12)
    ax.set_ylabel('Time-to-Empty (minutes)', fontsize=12)
    ax.set_title('TTE vs Initial SOC for Different Scenarios', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xticks(initial_socs)
    
    plt.tight_layout()
    plt.savefig('Fig_Task2_TTE_vs_SOC.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 已生成: Fig_Task2_TTE_vs_SOC.png")


def plot_power_by_soc(results):
    """图3：不同SOC下各场景的总功耗分布"""
    calc = TTECalculator(cycle_count=0, dt=1.0)
    initial_socs = [1.00, 0.75, 0.50, 0.25]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(results))
    width = 0.2
    colors = ['#4472C4', '#ED7D31', '#70AD47', '#FFC000']
    
    for soc_idx, soc in enumerate(initial_socs):
        powers = [r['power_total'] for r in results]
        bars = ax.bar(x + soc_idx * width, powers, width, 
                      label=f'Initial SOC = {soc:.2f}', color=colors[soc_idx], alpha=0.8)
        
        for bar, power in zip(bars, powers):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{power:.2f}', ha='center', va='bottom', fontsize=7)
    
    ax.set_xlabel('Usage Scenarios', fontsize=11)
    ax.set_ylabel('Total Power Consumption (W)', fontsize=11)
    ax.set_title('Total Power Consumption Across Scenarios (Different Initial SOC)', 
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([r['name'] for r in results], rotation=10, ha='right', fontsize=9)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('total_power_consumption_by_SOC.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 已生成: total_power_consumption_by_SOC.png")


def plot_power_ratio(results):
    """图4：各场景功耗占比饼图"""
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle('Power Consumption Ratio by Component (Initial SOC = 1.00)',
                 fontsize=12, fontweight='bold', y=0.98)
    
    for idx, r in enumerate(results):
        ax = axes[idx]
        power = r['power_breakdown']
        total = power['Total']
        
        # 准备数据
        labels = []
        values = []
        for comp in ['Base', 'Screen', 'CPU', 'Network', 'GPS']:
            pct = power[comp] / total * 100
            if pct > 0.5:  # 只显示>0.5%的
                labels.append(comp)
                values.append(pct)
        
        # 绘制饼图
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, colors=MCM_COLORS[:len(labels)],
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8}
        )
        
        ax.set_title(f'{r["name"]}\nTotal Power: {total:.2f} W',
                     fontsize=9, fontweight='bold', pad=10)
        
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(7)
    
    plt.subplots_adjust(wspace=0.3)
    plt.savefig('power_consumption_ratio_SOC_1.00.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print("✅ 已生成: power_consumption_ratio_SOC_1.00.png")


# ===================== 主程序 =====================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("  问题二 (Task 2) 完整分析 - 生成所有图片")
    print("="*80)
    
    # 运行分析
    results, stats = run_complete_analysis()
    
    # 生成所有图片
    print("\n>>> 生成论文图片...")
    plot_validation_comparison(results)
    plot_tte_vs_soc(results)
    plot_power_by_soc(results)
    plot_power_ratio(results)
    
    print("\n" + "="*80)
    print("  ✅ 完成！生成的图片：")
    print("    1. Fig_Task2_Validation.png")
    print("    2. Fig_Task2_TTE_vs_SOC.png")
    print("    3. total_power_consumption_by_SOC.png")
    print("    4. power_consumption_ratio_SOC_1.00.png")
    print("="*80)