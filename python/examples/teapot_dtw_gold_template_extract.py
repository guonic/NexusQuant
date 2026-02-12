# -*- coding: utf-8 -*-
"""
DTW Gold Template Batch Extraction (实体收复归一化).

Reads samples (ts_code, platform_start, platform_end, breakout_date), loads daily data,
computes vol_ma20, and exports .pt templates with anchor = consolidation close.max.

Usage:
    # Use built-in samples (no CSV)
    PYTHONPATH=python python python/examples/teapot_dtw_gold_template_extract.py --use-cache

    # Use CSV: ts_code, platform_start, platform_end, breakout_date (YYYYMMDD)
    PYTHONPATH=python python python/examples/teapot_dtw_gold_template_extract.py --samples data/dtw_gold_samples.csv --output outputs/dtw_templates --use-cache
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import polars as pl

from nq.config import DatabaseConfig, load_config
from nq.data.processor.teapot import TeapotDataLoader
from nq.trading.selector.teapot.dtw_gold_template import generate_pytorch_template

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default samples: ts_code, platform_start (YYYYMMDD), platform_end (YYYYMMDD), breakout_date (YYYYMMDD)
DEFAULT_SAMPLES: List[Tuple[str, str, str, str]] = [
    ("002540.SZ", "20250609", "20250617", "20250626"),
    ("600499.SH", "20250616", "20250618", "20250701"),
    ("688067.SH", "20241225", "20241231", "20250121"),
    ("688067.SH", "20251124", "20251210", "20260109"),
    ("688067.SH", "20250915", "20250918", "20251023"),
    ("603429.SH", "20230510", "20230524", "20230711"),
    ("603100.SH", "20220825", "20220830", "20220909"),
    ("603100.SH", "20221108", "20221111", "20221121"),
]


def _yyyymmdd_to_iso(d: str) -> str:
    """Convert YYYYMMDD to YYYY-MM-DD."""
    s = str(d).strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s


def load_samples(samples_path: Optional[str]) -> List[Tuple[str, str, str, str]]:
    """Load samples from CSV or return default. CSV columns: ts_code, platform_start, platform_end, breakout_date (YYYYMMDD)."""
    if not samples_path:
        return DEFAULT_SAMPLES

    path = Path(samples_path)
    if not path.exists():
        logger.warning("Samples file not found %s, using default samples", path)
        return DEFAULT_SAMPLES

    df = pd.read_csv(path, dtype=str)
    for col in ("ts_code", "platform_start", "platform_end", "breakout_date"):
        if col not in df.columns:
            raise ValueError(f"CSV must have column: {col}")

    out = []
    for _, row in df.iterrows():
        out.append(
            (
                str(row["ts_code"]).strip(),
                str(row["platform_start"]).strip().replace("-", "")[:8],
                str(row["platform_end"]).strip().replace("-", "")[:8],
                str(row["breakout_date"]).strip().replace("-", "")[:8],
            )
        )
    return out


def extract_one(
    ts_code: str,
    platform_start: str,
    platform_end: str,
    breakout_date: str,
    loader: TeapotDataLoader,
    output_dir: Path,
    use_cache: bool,
    use_float16: bool = False,
) -> Optional[Path]:
    """
    Load data for one sample, add vol_ma20, generate .pt template.

    Returns path to saved .pt or None on error.
    """
    p_start_iso = _yyyymmdd_to_iso(platform_start)
    p_end_iso = _yyyymmdd_to_iso(platform_end)
    break_iso = _yyyymmdd_to_iso(breakout_date)

    # Load extra history for rolling(20) / vol_ma20: need ≥20 days before platform_start
    # so that vol_ma20 is valid over the full template range (total_range starts at platform_start).
    from datetime import datetime, timedelta

    try:
        start_dt = datetime.strptime(p_start_iso, "%Y-%m-%d") - timedelta(days=100)
        load_start = start_dt.strftime("%Y-%m-%d")
    except Exception:
        load_start = p_start_iso
    load_end = break_iso

    pl_df = loader.load_daily_data(
        start_date=load_start,
        end_date=load_end,
        symbols=[ts_code],
    )

    if pl_df.is_empty():
        logger.warning("No data for %s %s-%s", ts_code, load_start, load_end)
        return None

    pl_df = pl_df.filter(pl.col("ts_code") == ts_code).sort("trade_date")

    # Rolling vol_ma20 (over ts_code already single)
    pl_df = pl_df.with_columns(
        pl.col("volume").rolling_mean(window_size=20).alias("vol_ma20")
    )

    # To pandas for template
    df = pl_df.to_pandas()
    df = df.rename(columns={"trade_date": "date"})
    df["date"] = pd.to_datetime(df["date"])

    total_range = (p_start_iso, break_iso)
    consolidation_range = (p_start_iso, p_end_iso)

    out_name = f"{ts_code}_{platform_start}_{breakout_date}.pt"
    save_path = output_dir / out_name

    try:
        generate_pytorch_template(
            df=df,
            total_range=total_range,
            consolidation_range=consolidation_range,
            save_path=str(save_path),
            date_col="date",
            use_float16=use_float16,
        )
        return save_path
    except ValueError as e:
        logger.warning("Skip %s %s-%s: %s", ts_code, platform_start, breakout_date, e)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract DTW gold templates (entity-recovery normalization)"
    )
    parser.add_argument(
        "--samples",
        type=str,
        default=None,
        help="CSV path: ts_code, platform_start, platform_end, breakout_date (YYYYMMDD). Default: built-in list.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/dtw_templates",
        help="Output directory for .pt files",
    )
    parser.add_argument("--use-cache", action="store_true", help="Use teapot cache")
    parser.add_argument(
        "--float16",
        action="store_true",
        help="Save tensor as float16 (save VRAM for 4090 batch DTW)",
    )

    args = parser.parse_args()

    samples = load_samples(args.samples)
    logger.info("Loaded %d samples", len(samples))

    try:
        config = load_config()
        db_config = config.database
    except Exception as e:
        logger.warning("Load config failed: %s, using defaults", e)
        db_config = DatabaseConfig()

    loader = TeapotDataLoader(
        db_config=db_config, schema="quant", use_cache=args.use_cache
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for ts_code, platform_start, platform_end, breakout_date in samples:
        path = extract_one(
            ts_code=ts_code,
            platform_start=platform_start,
            platform_end=platform_end,
            breakout_date=breakout_date,
            loader=loader,
            output_dir=output_dir,
            use_cache=args.use_cache,
            use_float16=args.float16,
        )
        if path:
            saved.append(path)

    logger.info("Done. Saved %d templates to %s", len(saved), output_dir)
    for p in saved:
        logger.info("  %s", p)


if __name__ == "__main__":
    main()
