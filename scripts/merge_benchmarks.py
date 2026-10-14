"""Merge one or more benchmark_raw_*.csv files into a single combined CSV.

Usage:
    python3 scripts/merge_benchmarks.py --inputs file1.csv,file2.csv --output combined.csv
"""
import argparse
import csv
import os


def merge(inputs, output):
    rows = []
    fieldnames = None
    for path in inputs:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                rows.append({k: row.get(k, "") for k in fieldnames})
        print(f"Loaded {path}")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Merged {len(rows)} rows from {len(inputs)} file(s) into {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge benchmark raw CSVs")
    parser.add_argument("--inputs", required=True, help="Comma-separated list of raw CSV paths")
    parser.add_argument("--output", required=True, help="Output combined CSV path")
    args = parser.parse_args()
    merge(args.inputs.split(","), args.output)
