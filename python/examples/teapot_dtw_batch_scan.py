# -*- coding: utf-8 -*-
"""
Batch DTW Scanner: Scan all stocks from qlib instruments list.

Reads stock list from qlib all.txt, scans each stock (2020-2023) with multiple DTW templates,
merges overlapping hits globally, and outputs to one CSV file.

Usage:
    PYTHONPATH=python python python/examples/teapot_dtw_batch_scan.py \
        --templates outputs/dtw_templates \
        --instruments /Users/guonic/.qlib/qlib_data/cn_data/instruments/all.txt \
        --output outputs/dtw_batch_results.csv \
        --use-cache

    # With custom date range and threshold
    PYTHONPATH=python python python/examples/teapot_dtw_batch_scan.py \
        --templates outputs/dtw_templates \
        --instruments /Users/guonic/.qlib/qlib_data/cn_data/instruments/all.txt \
        --start-date 2020-01-01 --end-date 2023-12-31 \
        --threshold 0.15 --merge-gap-days 5 \
        --output outputs/dtw_batch_results.csv --use-cache
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import polars as pl

from nq.config import DatabaseConfig, load_config
from nq.data.processor.teapot import TeapotDataLoader
from nq.trading.selector.teapot.dtw_cpu_scanner import (
    brute_force_cpu_scanner,
    load_template,
)
from nq.utils.data_normalize import normalize_stock_code

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _collect_template_paths(templates_arg: str) -> List[Path]:
    """Resolve --templates to a list of .pt paths (directory → glob *.pt, or single file)."""
    p = Path(templates_arg)
    if not p.exists():
        return []
    if p.is_file():
        return [p] if p.suffix.lower() == ".pt" else []
    return sorted(p.glob("*.pt"))


def _load_instruments(instruments_path: str) -> List[str]:
    """Load stock codes from qlib all.txt (format: code.market\tstart_date\tend_date)."""
    path = Path(instruments_path)
    if not path.exists():
        logger.error("Instruments file not found: %s", instruments_path)
        return []

    codes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if parts:
                code_raw = parts[0].strip()
                normalized = normalize_stock_code(code_raw)
                if normalized:
                    codes.append(normalized)

    logger.info("Loaded %d stock codes from %s", len(codes), instruments_path)
    return codes


def _merge_overlapping_hits(
    all_hits: List[Dict[str, Any]],
    gap_days: int = 5,
) -> List[Dict[str, Any]]:
    """
    Global dedup: group by ts_code only, merge overlapping/nearby intervals (any template).
    In each cluster pick representative: best score (min), then earliest end_date.
    """
    if not all_hits:
        return []

    from datetime import datetime

    def _gap_days(end_str: str, start_str: str) -> int:
        end_d = datetime.strptime(end_str, "%Y-%m-%d")
        start_d = datetime.strptime(start_str, "%Y-%m-%d")
        return (start_d - end_d).days

    def _group_by(items: List[Any], key) -> List[tuple]:
        from itertools import groupby

        sorted_items = sorted(items, key=key)
        return [(k, list(g)) for k, g in groupby(sorted_items, key=key)]

    merged: List[Dict[str, Any]] = []
    for ts_code, group in _group_by(all_hits, key=lambda h: h.get("ts_code", "")):
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
            rep = min(bucket, key=lambda x: (x["score"], x["end_date"]))
            merged.append({
                "template": rep["template"],
                "ts_code": rep["ts_code"],
                "start_date": rep["start_date"],
                "end_date": rep["end_date"],
                "score": rep["score"],
                "hit_count": len(bucket),
            })

    merged.sort(key=lambda h: (h["score"], h["end_date"]))
    return merged


def scan_one_stock(
    symbol: str,
    template_paths: List[Path],
    start_date: str,
    end_date: str,
    loader: TeapotDataLoader,
    conso_len: int,
    threshold: float,
    breakout_min: float,
    require_trap_structure: bool,
    trap_depth_ratio: float,
    reject_top_pattern: bool,
    start_flat_max_rise: float,
    last_rising_min_ratio: float,
) -> List[Dict[str, Any]]:
    """Scan one stock with all templates; return list of hits."""
    from datetime import datetime, timedelta

    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=100)
        load_start = start_dt.strftime("%Y-%m-%d")
    except Exception:
        load_start = start_date

    pl_df = loader.load_daily_data(
        start_date=load_start,
        end_date=end_date,
        symbols=[symbol],
    )

    if pl_df.is_empty():
        return []

    pl_df = pl_df.filter(pl.col("ts_code") == symbol).sort("trade_date")
    pl_df = pl_df.with_columns(
        pl.col("volume").rolling_mean(window_size=20).alias("vol_ma20")
    )

    stock_df = pl_df.to_pandas()
    stock_df = stock_df.rename(columns={"trade_date": "date"})
    stock_df["date"] = pd.to_datetime(stock_df["date"])

    all_hits: List[Dict[str, Any]] = []
    for template_path in template_paths:
        try:
            template_np, meta = load_template(str(template_path))
            t_len = template_np.shape[1]
            if len(stock_df) < t_len:
                continue

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
        except Exception as e:
            logger.debug("Skip template %s for %s: %s", template_path.name, symbol, e)
            continue

    return all_hits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch DTW scanner: scan all stocks from qlib instruments list"
    )
    parser.add_argument(
        "--templates",
        type=str,
        required=True,
        help="Path to a single .pt file or a directory of .pt templates",
    )
    parser.add_argument(
        "--instruments",
        type=str,
        required=True,
        help="Path to qlib all.txt (format: code.market\\tstart\\tend per line)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-01-01",
        help="Start date YYYY-MM-DD (default: 2020-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2023-12-31",
        help="End date YYYY-MM-DD (default: 2023-12-31)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/dtw_batch_results.csv",
        help="Output CSV file path",
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
        help="Max distance to count as hit (default 0.18)",
    )
    parser.add_argument(
        "--breakout-min",
        type=float,
        default=1.0,
        help="Only count hit if last-day close (norm) >= this (default 1.0)",
    )
    parser.add_argument(
        "--no-trap-structure",
        action="store_true",
        help="Disable requirement for middle dip",
    )
    parser.add_argument(
        "--trap-depth-ratio",
        type=float,
        default=0.97,
        help="Middle dip depth ratio (default 0.97)",
    )
    parser.add_argument(
        "--no-reject-top",
        action="store_true",
        help="Do not reject top pattern",
    )
    parser.add_argument(
        "--start-flat-max-rise",
        type=float,
        default=0.05,
        help="Max rise in first conso segment (default 0.05)",
    )
    parser.add_argument(
        "--last-rising-min-ratio",
        type=float,
        default=1.0,
        help="Require mean(last 3)/mean(prev 3) >= this (default 1.0)",
    )
    parser.add_argument(
        "--merge-gap-days",
        type=int,
        default=5,
        help="Merge hits if interval gap <= this (default 5)",
    )
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=None,
        help="Limit number of stocks to scan (for testing)",
    )

    args = parser.parse_args()

    template_paths = _collect_template_paths(args.templates)
    if not template_paths:
        logger.error("No templates found: %s", args.templates)
        return

    logger.info("Templates: %d %s", len(template_paths), [t.name for t in template_paths])

    instruments = _load_instruments(args.instruments)
    if not instruments:
        logger.error("No instruments loaded")
        return

    if args.max_stocks:
        instruments = instruments[: args.max_stocks]
        logger.info("Limited to first %d stocks", args.max_stocks)

    try:
        config = load_config()
        db_config = config.database
    except Exception as e:
        logger.warning("Load config failed: %s", e)
        db_config = DatabaseConfig()

    loader = TeapotDataLoader(
        db_config=db_config, schema="quant", use_cache=args.use_cache
    )

    all_hits: List[Dict[str, Any]] = []
    total = len(instruments)
    for i, symbol in enumerate(instruments, 1):
        try:
            hits = scan_one_stock(
                symbol=symbol,
                template_paths=template_paths,
                start_date=args.start_date,
                end_date=args.end_date,
                loader=loader,
                conso_len=args.conso_len,
                threshold=args.threshold,
                breakout_min=args.breakout_min,
                require_trap_structure=not args.no_trap_structure,
                trap_depth_ratio=args.trap_depth_ratio,
                reject_top_pattern=not args.no_reject_top,
                start_flat_max_rise=args.start_flat_max_rise,
                last_rising_min_ratio=args.last_rising_min_ratio,
            )
            all_hits.extend(hits)
            if hits:
                logger.info(
                    "[%d/%d] %s: %d hits (total so far: %d)",
                    i,
                    total,
                    symbol,
                    len(hits),
                    len(all_hits),
                )
            elif i % 100 == 0:
                logger.info("[%d/%d] %s: no hits", i, total, symbol)
        except Exception as e:
            logger.warning("[%d/%d] %s failed: %s", i, total, symbol, e)
            continue

    logger.info("Total raw hits: %d", len(all_hits))

    merged_hits = _merge_overlapping_hits(all_hits, gap_days=args.merge_gap_days)
    logger.info("After merge: %d hits", len(merged_hits))

    if not merged_hits:
        logger.info("No hits to write")
        return

    rows = [
        {
            "template": h["template"],
            "ts_code": h["ts_code"],
            "start_date": h["start_date"],
            "end_date": h["end_date"],
            "score": h["score"],
            "hit_count": h.get("hit_count", 1),
        }
        for h in merged_hits
    ]
    summary_df = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    logger.info("Wrote %d merged hits to %s", len(merged_hits), output_path)


if __name__ == "__main__":
    main()
