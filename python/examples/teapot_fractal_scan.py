# -*- coding: utf-8 -*-
"""
Teapot Fractal Box Scanner (分型转折破壳而出 信号捕捉).

Runs capture_topological_final: fractal-boundary blocks + "break out of hood" signal.
仅做信号捕捉，输出 CSV：ts_code, signal_date, trade_date, hood_ceiling, hood_floor, close。

Usage:
  PYTHONPATH=python python python/examples/teapot_fractal_scan.py --start-date 2023-01-01 --end-date 2024-12-31
  PYTHONPATH=python python python/examples/teapot_fractal_scan.py --start-date 2023-01-01 --end-date 2024-12-31 --symbols 600487.SH --output outputs/teapot/fractal_signals.csv --use-cache
"""

import argparse
import logging
from pathlib import Path

import polars as pl

from nq.config import DatabaseConfig, load_config
from nq.data.processor.teapot import TeapotDataLoader
from nq.trading.selector.teapot import FractalBoxScanner
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
    scanner: FractalBoxScanner,
    output_path: str,
) -> None:
    """Run fractal box scanner, keep rows where signal_pure_topology, write CSV."""
    analyzed = scanner.analyze(df, ts_code_col="ts_code")

    signals = analyzed.filter(pl.col("signal_pure_topology"))

    if signals.is_empty():
        logger.warning("No signals to write")
        out_df = pl.DataFrame(schema={
            "ts_code": pl.Utf8,
            "signal_date": pl.Utf8,
            "trade_date": pl.Utf8,
            "hood_ceiling": pl.Float64,
            "hood_floor": pl.Float64,
            "close": pl.Float64,
        })
    else:
        out_df = signals.select([
            pl.col("ts_code"),
            pl.col("trade_date").alias("signal_date"),
            pl.col("trade_date"),
            pl.col("the_hood").struct.field("c").alias("hood_ceiling"),
            pl.col("the_hood").struct.field("f").alias("hood_floor"),
            pl.col("close"),
        ])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.write_csv(output_path)
    logger.info("Wrote %d signals to %s", len(out_df), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teapot fractal box scanner (分型转折破壳而出)"
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
        default="outputs/teapot/fractal_signals.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--had-below-floor-window",
        type=int,
        default=10,
        help="Lookback days for had broken below hood floor (default: 10)",
    )
    parser.add_argument(
        "--no-yang-line",
        action="store_true",
        help="Do not require close > open for signal",
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

    scanner = FractalBoxScanner(
        had_below_floor_window=args.had_below_floor_window,
        require_yang_line=not args.no_yang_line,
    )
    run_scan(df, scanner, args.output)


if __name__ == "__main__":
    main()
