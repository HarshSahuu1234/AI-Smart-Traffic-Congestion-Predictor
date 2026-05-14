"""
Convert the pasted tab-separated dataset to a proper CSV.
Run this once, then run import_dataset.py to retrain the model.

HOW TO USE:
1. Copy ALL the data from Google Sheets (Ctrl+A, Ctrl+C)
2. Paste it into a file called datasets/google_traffic_raw.txt
3. Run: python convert_dataset.py
"""
import csv
import os

BASE = os.path.dirname(os.path.abspath(__file__))
RAW_TXT = os.path.join(BASE, "datasets", "google_traffic_raw.txt")
OUT_CSV = os.path.join(BASE, "datasets", "google_traffic_data.csv")

if not os.path.exists(RAW_TXT):
    print(f"[ERROR] Please paste your data into: {RAW_TXT}")
    print("  1. Create the file")
    print("  2. Paste ALL rows from Google Sheets")
    print("  3. Run this script again")
    exit(1)

print(f"Reading from: {RAW_TXT}")
with open(RAW_TXT, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Parse tab-separated data
rows = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    parts = line.split('\t')
    rows.append(parts)

if len(rows) < 2:
    print(f"[ERROR] Only {len(rows)} rows found. Check the file.")
    exit(1)

header = rows[0]
data = rows[1:]

print(f"Header: {header}")
print(f"Data rows: {len(data)}")

# Write as proper CSV
with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)

print(f"\nSaved {len(data)} rows to: {OUT_CSV}")
print("Now run: python import_dataset.py")
