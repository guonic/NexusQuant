# -*- coding: utf-8 -*-
"""
Teapot Vertical Piercing Scanner (多周期阻力簇垂直贯穿 信号捕捉).

Runs capture_vertical_piercing: multi-period MA resistance cloud + vertical piercing signal.
仅做信号捕捉，输出 CSV：ts_code, signal_date, trade_date, cloud_top, cloud_bottom, close。

Usage:
  PYTHONPATH=python python python/examples/teapot_vertical_piercing_scan.py --start-date 2023-01-01 --end-date 2024-12-31
  PYTHONPATH=python python python/examples/teapot_vertical_piercing_scan.py --start-date 2023-01-01 --end-date 2024-12-31 --symbols 600487.SH --output outputs/teapot/piercing_signals.csv --use-cache
"""

import argparse
import logging
from pathlib import Path

import polars as pl

from nq.config import DatabaseConfig, load_config
from nq.data.processor.teapot import TeapotDataLoader
from nq.trading.selector.teapot import VerticalPiercingScanner
from nq.utils.data_normalize import normalize_stock_code

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_symbols(raw: list) -> list:
    """Normalize and expand comma-separated symbols to ts_code list."""
    parts = []
    for s in raw:
        parts.extend(p.strip() for p in s.split(",") if p.strip())
    codes = [normalize_stock_code(p) for p in parts]
    seen = set()
    return [c for c in codes if c and c not in seen and not seen.add(c)]


def run_scan(
    df: pl.DataFrame,
    scanner: VerticalPiercingScanner,
    output_path: str,
) -> None:
    """Run vertical piercing scanner, keep rows where signal_piercing, write CSV."""
    analyzed = scanner.analyze(df, ts_code_col="ts_code")

    signals = analyzed.filter(pl.col("signal_piercing"))

    if signals.is_empty():
        logger.warning("No signals to write")
        out_df = pl.DataFrame(schema={
            "ts_code": pl.Utf8,
            "signal_date": pl.Utf8,
            "trade_date": pl.Utf8,
            "cloud_top": pl.Float64,
            "cloud_bottom": pl.Float64,
            "close": pl.Float64,
        })
    else:
        out_df = signals.select([
            pl.col("ts_code"),
            pl.col("trade_date").alias("signal_date"),
            pl.col("trade_date"),
            pl.col("cloud_top"),
            pl.col("cloud_bottom"),
            pl.col("close"),
        ])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.write_csv(output_path)
    logger.info("Wrote %d signals to %s", len(out_df), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teapot vertical piercing scanner (多周期阻力簇垂直贯穿)"
    )
    parser.add_argument("--start-date", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="*",
        help="Stock codes (e.g. 600487 or 600487.SH)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/teapot/piercing_signals.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--ma-periods",
        type=int,
        nargs="+",
        default=[5, 10, 20, 30, 60],
        help="MA periods (default: 5 10 20 30 60)",
    )
    parser.add_argument(
        "--deep-pit-days",
        type=int,
        default=5,
        help="Lookback days for had been under cloud (default: 5)",
    )
    parser.add_argument(
        "--no-yang-line",
        action="store_true",
        help="Do not require close > open for signal",
    )
    parser.add_argument(
        "--require-volume",
        action="store_true",
        help="Require volume expansion",
    )
    parser.add_argument(
        "--volume-ratio",
        type=float,
        default=1.5,
        help="Volume expansion threshold (default: 1.5)",
    )
    parser.add_argument("--use-cache", action="store_true")
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols) if args.symbols else None
    if args.symbols and not symbols:
        logger.error("No valid symbols: %s", args.symbols)
        return

    try:
        config = load_config()
        db_config = config.database
    except Exception as e:
        logger.warning("Load config failed: %s", e)
        db_config = DatabaseConfig()

    loader = TeapotDataLoader(
        db_config=db_config,
        schema="quant",
        use_cache=args.use_cache,
    )
    logger.info("Loading data %s to %s", args.start_date, args.end_date)
    df = loader.load_daily_data(
        start_date=args.start_date,
        end_date=args.end_date,
        symbols=symbols,
    )
    if df.is_empty():
        logger.error("No data loaded")
        return

    scanner = VerticalPiercingScanner(
        ma_periods=args.ma_periods,
        deep_pit_days=args.deep_pit_days,
        require_yang_line=not args.no_yang_line,
        require_volume_expansion=args.require_volume,
        volume_ratio_threshold=args.volume_ratio,
    )
    run_scan(df, scanner, args.output)


if __name__ == "__main__":
    main()
