import os
import numpy as np
import csv

def main():
    import pybamm

    # 1) 选择参数集（能加载哪个用哪个）
    candidates = ["Chen2020", "Marquis2019", "Xu2019"]
    param = None
    chosen = None
    last_err = None
    for name in candidates:
        try:
            param = pybamm.ParameterValues(name)
            chosen = name
            break
        except Exception as e:
            last_err = e
    if param is None:
        raise RuntimeError(f"Cannot load any known parameter set. Last error: {last_err}")

    # 2) 取正负极开路电势函数（OCP），拼成全电池 OCV
    # 这些键在 PyBaMM 中相对稳定
    Un = param["Negative electrode OCP [V]"]
    Up = param["Positive electrode OCP [V]"]

    # 3) 生成一条“归一化 SOC->化学计量比”的映射
    # 由于某些参数集不提供 min/max stoichiometry，我们用 [0,1] 作为归一化范围，
    # 得到 OCV 形状（对手机电池建模足够；容量/截止电压由你们手机参数再约束）。
    soc = np.linspace(0, 1, 201)
    x = soc                 # 负极计量比（归一化）
    y = 1 - soc             # 正极计量比（反向）

    ocv = np.array([float(Up(yi) - Un(xi)) for xi, yi in zip(x, y)])

    # 4) 输出 CSV
    os.makedirs("data_processed", exist_ok=True)
    out = "data_processed/ocv_soc_curve.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["soc", "ocv_V", "source_param_set", "pybamm_version"])
        for s, v in zip(soc, ocv):
            w.writerow([float(s), float(v), chosen, pybamm.__version__])

    print("Wrote:", out)
    print("Used parameter set:", chosen)
    print("PyBaMM version:", pybamm.__version__)

if __name__ == "__main__":
    main()