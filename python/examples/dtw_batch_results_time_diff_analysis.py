#!/usr/bin/env python3
"""
Analyze time difference distribution between column 8 (platform_start) and
column 10 (breakout_date) in DTW batch results CSV.

Usage:
    python examples/dtw_batch_results_time_diff_analysis.py
    python examples/dtw_batch_results_time_diff_analysis.py --csv path/to/file.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Project root for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_date(s):
    """Parse date string (YYYY-MM-DD or similar) to datetime. Returns None if invalid/missing."""
    if pd.isna(s) or s is None or (isinstance(s, str) and not s.strip()):
        return None
    try:
        return pd.to_datetime(s)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Analyze time diff between col8 and col10 in DTW CSV")
    parser.add_argument(
        "--csv",
        type=str,
        default="/Users/guonic/Workspace/OpenSource/atm/outputs/dtw_batch_results.csv",
        help="Path to DTW batch results CSV (10 columns; col8=platform_start, col10=breakout_date)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of bins for histogram (default: 30)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if df.shape[1] < 10:
        print(f"Error: expected at least 10 columns, got {df.shape[1]}", file=sys.stderr)
        sys.exit(1)

    # Column indices are 1-based in user description: col8 = index 7, col10 = index 9
    col8_name = df.columns[7]
    col10_name = df.columns[9]

    col8 = df.iloc[:, 7]
    col10 = df.iloc[:, 9]

    # Parse to datetime
    d8 = col8.map(parse_date)
    d10 = col10.map(parse_date)

    # Valid pairs: both present
    valid = d8.notna() & d10.notna()
    n_valid = valid.sum()
    n_total = len(df)
    n_missing = n_total - n_valid

    print(f"File: {csv_path}")
    print(f"Total rows: {n_total}")
    print(f"Rows with both col8 ({col8_name}) and col10 ({col10_name}): {n_valid}")
    print(f"Rows with missing date(s): {n_missing}")
    print()

    if n_valid == 0:
        print("No valid date pairs to analyze.")
        sys.exit(0)

    # Time difference in days (breakout_date - platform_start)
    diff_days = (d10 - d8).dt.days[valid]

    print("=" * 60)
    print("Time difference (days): breakout_date - platform_start")
    print("=" * 60)
    print(f"Count:     {len(diff_days)}")
    print(f"Min:       {diff_days.min()} days")
    print(f"Max:       {diff_days.max()} days")
    print(f"Mean:      {diff_days.mean():.2f} days")
    print(f"Median:    {diff_days.median():.0f} days")
    print(f"Std:       {diff_days.std():.2f} days")
    print()
    print("Percentiles (days):")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  P{p:2d}: {diff_days.quantile(p / 100.0):.0f}")
    print()

    # Histogram (counts per bin)
    print("Distribution (histogram, days):")
    hist, bin_edges = pd.cut(diff_days, bins=args.bins, retbins=True, include_lowest=True)
    counts = hist.value_counts().sort_index()
    max_count = counts.max()
    scale = 50.0 / max_count if max_count > 0 else 0
    for interval in counts.index:
        n = counts[interval]
        bar = "█" * int(n * scale) + "░" * (50 - int(n * scale))
        print(f"  {interval}: {n:5d} |{bar}")
    print()

    # Value counts for integer days (for small ranges)
    if diff_days.max() - diff_days.min() <= 100:
        print("Exact day counts (top 20):")
        vc = diff_days.astype(int).value_counts().sort_index()
        for day, cnt in vc.head(20).items():
            print(f"  {int(day):4d} days: {int(cnt):5d}")
    else:
        print("Day range summary (buckets):")
        buckets = [(0, 0, "0"), (1, 5, "1-5"), (6, 10, "6-10"), (11, 20, "11-20"),
                   (21, 40, "21-40"), (41, 60, "41-60"), (61, 100, "61-100"), (101, 9999, ">100")]
        for low, high, label in buckets:
            n = ((diff_days >= low) & (diff_days <= high)).sum()
            if n > 0:
                print(f"  {label:>6} days: {n:5d}")


if __name__ == "__main__":
    main()
