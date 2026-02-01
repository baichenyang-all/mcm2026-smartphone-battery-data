import os, sys, csv
import xml.etree.ElementTree as ET

def find_power_profile_xml(root):
    hits = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn == "power_profile.xml":
                hits.append(os.path.join(dirpath, fn))
    return hits

def parse_power_profile(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows = []
    for child in root:
        tag = child.tag.strip()
        name = child.attrib.get("name", "").strip()
        if tag == "item":
            val = (child.text or "").strip()
            rows.append([name, "item", val, ""])
        elif tag == "array":
            vals = []
            for v in child.findall("value"):
                vals.append((v.text or "").strip())
            rows.append([name, "array", "|".join(vals), f"len={len(vals)}"])
    return rows

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_power_profile.py <repo_root> [out_csv]")
        sys.exit(1)

    repo_root = sys.argv[1]
    out_csv = sys.argv[2] if len(sys.argv) >= 3 else "power_profile.csv"

    xmls = find_power_profile_xml(repo_root)
    if not xmls:
        raise FileNotFoundError("No power_profile.xml found under: " + repo_root)

    xml_path = xmls[0]
    print("Using:", xml_path)

    rows = parse_power_profile(xml_path)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "type", "value", "note"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_csv}")

if __name__ == "__main__":
    main()