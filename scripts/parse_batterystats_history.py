import re, sys
import pandas as pd

# 行例：
# +1m06s356ms (3) 100 +running ...
# +2s438ms (2) 100 volt=4192 ...
# +29ms (2) 100 -usb_data ...

# 匹配相对时间字符串（在行首）
re_time = re.compile(r"^\s*\+([0-9a-zA-Z]+)\s+\(\d+\)\s+(\d+)\b")

# 电压/温度/插电状态等可选字段
re_volt = re.compile(r"\bvolt=(\d+)\b")
re_temp = re.compile(r"\btemp=(\-?\d+)\b")
re_plug = re.compile(r"\bplug=([a-zA-Z_]+)\b")
re_status = re.compile(r"\bstatus=([a-zA-Z_]+)\b")

def reltime_to_ms(s: str) -> int:
    ms_total = 0
    # minutes: 'Xm' but NOT 'ms'
    mm = re.search(r"(\d+)m(?!s)", s)
    if mm:
        ms_total += int(mm.group(1)) * 60 * 1000
    # seconds: 'Xs' but NOT 'sms' (其实主要防 'ms' 干扰)
    ss = re.search(r"(\d+)s(?!\d*ms)", s)
    if ss:
        ms_total += int(ss.group(1)) * 1000
    # milliseconds: 'Xms'
    mms = re.search(r"(\d+)ms", s)
    if mms:
        ms_total += int(mms.group(1))
    return ms_total

def main(in_txt, out_csv):
    rows = []
    t_ms = 0  # 累计时间（毫秒）

    with open(in_txt, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            # 我们只解析以 + 开头的相对时间行
            if not line.startswith("+"):
                continue
            mt = re_time.match(line)
            if not mt:
                continue

            rel = mt.group(1)         # 例如 1m06s356ms
            soc = int(mt.group(2))    # 例如 100
            t_ms += reltime_to_ms(rel)

            mv = None
            mtv = re_volt.search(line)
            if mtv:
                mv = int(mtv.group(1))

            tdC = None
            mtt = re_temp.search(line)
            if mtt:
                tdC = int(mtt.group(1))

            plug = None
            mpg = re_plug.search(line)
            if mpg:
                plug = mpg.group(1)

            status = None
            mst = re_status.search(line)
            if mst:
                status = mst.group(1)

            rows.append((t_ms/1000.0, soc, mv, tdC, plug, status, line))

    if not rows:
        raise RuntimeError("No parsable '+time ... soc ...' lines found.")

    df = pd.DataFrame(rows, columns=["t_s","soc_percent","voltage_mV","temp_dC","plug","status","raw_line"])

    # SOC/电压/温度不是每行都有，前向填充更像真实记录
    df["soc_percent"] = df["soc_percent"].ffill()
    df["voltage_mV"] = df["voltage_mV"].ffill()
    df["temp_dC"] = df["temp_dC"].ffill()
    df["plug"] = df["plug"].ffill()
    df["status"] = df["status"].ffill()

    # 同一时刻可能多条，保留最后一条
    df = df.drop_duplicates(subset=["t_s"], keep="last").reset_index(drop=True)

    df.to_csv(out_csv, index=False, encoding="utf-8")
    print("Wrote:", out_csv, "rows:", len(df))
    print(df.head())
    print(df.tail())

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/parse_batterystats_history.py <history.txt> <out.csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])