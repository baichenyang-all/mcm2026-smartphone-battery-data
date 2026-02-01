import glob, re

files = glob.glob(r"release_github/data_raw_validation/scenario*_history.txt")
for f in files:
    s = open(f, "r", encoding="utf-8", errors="ignore").read()

    # 1) Redact any quoted strings "...", to hide SSID/package names etc.
    s = re.sub(r"\".*?\"", "\"<redacted>\"", s)

    # 2) Redact package-like tokens (e.g., com.xxx.yyy)
    s = re.sub(r"\b[a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)+\b", "<pkg>", s)

    # 3) Redact uid-like tokens (e.g., u0a123)
    s = re.sub(r"\bu0a\d+\b", "<uid>", s)

    open(f, "w", encoding="utf-8").write(s)

print("sanitized history files:", len(files))
print("example file:", files[0] if files else "None")
