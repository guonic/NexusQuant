# -*- coding: utf-8 -*-
"""
CPU 5-channel DTW brute-force scanner.

Traverses multiple gold templates (.pt), loads one stock in a date range,
runs sliding-window match with local normalization per template, and outputs
aggregated hits for training or inspection.

Usage:
    # Scan with all .pt in a directory (multiple templates)
    PYTHONPATH=python python python/examples/teapot_dtw_cpu_scan.py \
        --templates outputs/dtw_templates \
        --symbol 600487.SH \
        --start-date 2024-01-01 --end-date 2024-12-31 \
        --output outputs/dtw_cpu_hits --use-cache

    # Single template (path to one .pt file)
    PYTHONPATH=python python python/examples/teapot_dtw_cpu_scan.py \
        --templates outputs/dtw_templates/002540.SZ_20250609_20250626.pt \
        --symbol 600487.SH --start-date 2024-01-01 --end-date 2024-12-31 --use-cache
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import polars as pl

from nq.config import DatabaseConfig, load_config
from nq.data.processor.teapot import TeapotDataLoader
from nq.trading.selector.teapot.dtw_cpu_scanner import (
    brute_force_cpu_scanner,
    load_template,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _merge_overlapping_hits(
    all_hits: List[Dict[str, Any]],
    gap_days: int = 5,
) -> List[Dict[str, Any]]:
    """
    Global dedup: group by ts_code only, merge overlapping/nearby intervals (any template).
    In each cluster pick representative: best score (min), then earliest end_date.
    Output one row per cluster with rep's template, start, end, score; hit_count = cluster size.
    """
    if not all_hits:
        return []

    merged: List[Dict[str, Any]] = []
    for ts_code, group in _group_by(all_hits, key=lambda h: h.get("ts_code", "")):
        # Sort by start_date then score
        sorted_group = sorted(group, key=lambda h: (h["start_date"], h["score"]))
        buckets: List[List[Dict[str, Any]]] = []
        for h in sorted_group:
            s, e = h["start_date"], h["end_date"]
            if not buckets:
                buckets.append([h])
                continue
            last_bucket = buckets[-1]
            last_max_end = max(x["end_date"] for x in last_bucket)
            if _gap_days(last_max_end, s) <= gap_days:
                last_bucket.append(h)
            else:
                buckets.append([h])

        for bucket in buckets:
            # Best = min score, then earliest end_date (分数最优，结束时间最早)
            rep = min(bucket, key=lambda x: (x["score"], x["end_date"]))
            merged.append({
                "template": rep["template"],
                "ts_code": rep["ts_code"],
                "start_date": rep["start_date"],
                "end_date": rep["end_date"],
                "score": rep["score"],
                "hit_count": len(bucket),
                "start_index": rep["start_index"],
                "end_index": rep["end_index"],
                "data_slice": rep.get("data_slice"),
            })

    merged.sort(key=lambda h: (h["score"], h["end_date"]))
    return merged


def _group_by(items: List[Any], key) -> List[tuple]:
    from itertools import groupby
    sorted_items = sorted(items, key=key)
    return [(k, list(g)) for k, g in groupby(sorted_items, key=key)]


def _gap_days(end_str: str, start_str: str) -> int:
    """Days from end_str to start_str (positive if start is after end)."""
    end_d = datetime.strptime(end_str, "%Y-%m-%d")
    start_d = datetime.strptime(start_str, "%Y-%m-%d")
    return (start_d - end_d).days


def _collect_template_paths(templates_arg: str) -> List[Path]:
    """Resolve --templates to a list of .pt paths (directory → glob *.pt, or single file)."""
    p = Path(templates_arg)
    if not p.exists():
        return []
    if p.is_file():
        return [p] if p.suffix.lower() == ".pt" else []
    return sorted(p.glob("*.pt"))


def run_scan(
    templates_arg: str,
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: str,
    use_cache: bool = False,
    conso_len: int = 20,
    threshold: float = 0.18,
    breakout_min: float = 1.0,
    require_trap_structure: bool = True,
    trap_depth_ratio: float = 0.97,
    reject_top_pattern: bool = True,
    start_flat_max_rise: float = 0.05,
    last_rising_min_ratio: float = 1.0,
    merge_gap_days: int = 5,
    save_slices: bool = False,
) -> None:
    """
    Load stock data once, then for each template run CPU DTW scanner;
    aggregate hits with template name, write one CSV and optionally save slices.
    """
    template_paths = _collect_template_paths(templates_arg)
    if not template_paths:
        logger.error("No .pt templates found: %s (use a .pt file or a directory of .pt)", templates_arg)
        return

    logger.info("Templates to scan: %d %s", len(template_paths), [t.name for t in template_paths])

    # Load stock data once (extra history for vol_ma20)
    from datetime import datetime, timedelta

    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=100)
        load_start = start_dt.strftime("%Y-%m-%d")
    except Exception:
        load_start = start_date

    try:
        config = load_config()
        db_config = config.database
    except Exception as e:
        logger.warning("Load config failed: %s", e)
        db_config = DatabaseConfig()

    loader = TeapotDataLoader(
        db_config=db_config, schema="quant", use_cache=use_cache
    )
    pl_df = loader.load_daily_data(
        start_date=load_start,
        end_date=end_date,
        symbols=[symbol],
    )

    if pl_df.is_empty():
        logger.error("No data for %s %s..%s", symbol, load_start, end_date)
        return

    pl_df = pl_df.filter(pl.col("ts_code") == symbol).sort("trade_date")
    pl_df = pl_df.with_columns(
        pl.col("volume").rolling_mean(window_size=20).alias("vol_ma20")
    )

    stock_df = pl_df.to_pandas()
    stock_df = stock_df.rename(columns={"trade_date": "date"})
    stock_df["date"] = pd.to_datetime(stock_df["date"])

    logger.info("Stock %s: %d rows", symbol, len(stock_df))

    all_hits: List[dict] = []
    for template_path in template_paths:
        template_np, meta = load_template(str(template_path))
        t_len = template_np.shape[1]
        logger.info("Matching template %s (length %d)", template_path.name, t_len)

        hits = brute_force_cpu_scanner(
            stock_df=stock_df,
            template_path=str(template_path),
            conso_len=conso_len,
            threshold=threshold,
            breakout_min=breakout_min,
            require_trap_structure=require_trap_structure,
            trap_depth_ratio=trap_depth_ratio,
            reject_top_pattern=reject_top_pattern,
            start_flat_max_rise=start_flat_max_rise,
            last_rising_min_ratio=last_rising_min_ratio,
            date_col="date",
            ts_code_col="ts_code",
        )

        for h in hits:
            h["template"] = template_path.name
        all_hits.extend(hits)

    all_hits.sort(key=lambda h: (h["score"], h["start_date"]))

    # Merge overlapping / nearby hits (same ts_code + template) into one per cluster
    if merge_gap_days > 0:
        merged_hits = _merge_overlapping_hits(all_hits, gap_days=merge_gap_days)
        logger.info(
            "Total hits %d -> merged %d (gap_days=%d)",
            len(all_hits),
            len(merged_hits),
            merge_gap_days,
        )
    else:
        merged_hits = [dict(h, hit_count=1) for h in all_hits]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not merged_hits:
        logger.info("No hits after merge")
        return

    for i, h in enumerate(merged_hits, 1):
        logger.info(
            "  [%d] template=%s | %s ~ %s | score=%.4f | hit_count=%d",
            i,
            h["template"],
            h["start_date"],
            h["end_date"],
            h["score"],
            h.get("hit_count", 1),
        )

    rows = [
        {
            "template": h["template"],
            "ts_code": h.get("ts_code", symbol),
            "start_date": h["start_date"],
            "end_date": h["end_date"],
            "start_index": h.get("start_index"),
            "end_index": h.get("end_index"),
            "score": h["score"],
            "hit_count": h.get("hit_count", 1),
        }
        for h in merged_hits
    ]
    summary_df = pd.DataFrame(rows)
    csv_path = out_dir / f"dtw_hits_{symbol}_{start_date}_{end_date}.csv"
    summary_df.to_csv(csv_path, index=False)
    logger.info("Wrote %s", csv_path)

    if save_slices:
        slices_dir = out_dir / "slices"
        slices_dir.mkdir(parents=True, exist_ok=True)
        for i, h in enumerate(merged_hits):
            if h.get("data_slice") is None:
                continue
            base = h["template"].replace(".pt", "")
            name = f"{symbol}_{base}_{h['start_date']}_{h['end_date']}_score{h['score']:.4f}.npy"
            np.save(slices_dir / name, h["data_slice"])
        logger.info("Saved %d slices to %s", len([h for h in merged_hits if h.get("data_slice") is not None]), slices_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU 5-channel DTW scanner: match one stock to one or many gold templates"
    )
    parser.add_argument(
        "--templates",
        type=str,
        required=True,
        help="Path to a single .pt file or a directory of .pt templates (all will be scanned)",
    )
    parser.add_argument("--symbol", type=str, required=True, help="Stock code (e.g. 600487.SH)")
    parser.add_argument("--start-date", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/dtw_cpu_hits",
        help="Output directory for CSV and optional slices",
    )
    parser.add_argument("--use-cache", action="store_true", help="Use teapot cache")
    parser.add_argument(
        "--conso-len",
        type=int,
        default=20,
        help="Days at start of each window used as consolidation anchor (default 20)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.18,
        help="Max distance to count as hit (default 0.18; try 0.15 for stricter)",
    )
    parser.add_argument(
        "--breakout-min",
        type=float,
        default=1.0,
        help="Only count hit if last-day close (norm) >= this (default 1.0; 0 to disable)",
    )
    parser.add_argument(
        "--no-trap-structure",
        action="store_true",
        help="Disable requirement for middle dip (conso→trap→breakout); match any shape",
    )
    parser.add_argument(
        "--trap-depth-ratio",
        type=float,
        default=0.97,
        help="Middle dip: mid_min <= ratio*first/last mean (default 0.97; smaller = deeper dip)",
    )
    parser.add_argument(
        "--no-reject-top",
        action="store_true",
        help="Do not reject top pattern (rise then fall); default is to reject it",
    )
    parser.add_argument(
        "--start-flat-max-rise",
        type=float,
        default=0.05,
        help="Max rise in first conso segment (default 0.05); reject if start ramps up more",
    )
    parser.add_argument(
        "--last-rising-min-ratio",
        type=float,
        default=1.0,
        help="Require mean(last 3)/mean(prev 3) >= this (default 1.0); reject if last part declining",
    )
    parser.add_argument(
        "--merge-gap-days",
        type=int,
        default=5,
        help="Merge hits with same template if interval gap <= this (default 5); 0 = no merge",
    )
    parser.add_argument(
        "--save-slices",
        action="store_true",
        help="Save each hit's data_slice as .npy for training",
    )

    args = parser.parse_args()

    run_scan(
        templates_arg=args.templates,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output,
        use_cache=args.use_cache,
        conso_len=args.conso_len,
        threshold=args.threshold,
        breakout_min=args.breakout_min,
        require_trap_structure=not args.no_trap_structure,
        trap_depth_ratio=args.trap_depth_ratio,
        reject_top_pattern=not args.no_reject_top,
        start_flat_max_rise=args.start_flat_max_rise,
        last_rising_min_ratio=args.last_rising_min_ratio,
        merge_gap_days=args.merge_gap_days,
        save_slices=args.save_slices,
    )


if __name__ == "__main__":
    main()
