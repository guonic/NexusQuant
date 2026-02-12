# -*- coding: utf-8 -*-
"""
Teapot Topological Trend Scanner (状态拓扑选股器).

State evolution: 线段 (Trend Strings) + 中继节点 (Mediation Nodes).
Atomic states: 上升中 / 下跌中 / 纠缠中 -> streaks -> last_relay_high -> signal_4 / signal_6.
Outputs signals to file with stage time intervals; consecutive trigger days keep first only.

CSV columns:
  - signal_type=4 (buy1): d_start/end (下跌段), r_start/end (中继段); u_start/end, s4_start/end are null.
  - signal_type=6 (buy2): u_start/end (上升段), r_start/end (中继段), s4_start/end (signal_4发生日期); d_start/end are null.

Usage:
  python teapot_topological_scan.py --start-date 2023-01-01 --end-date 2024-12-31 --symbols 600487.SH
  python teapot_topological_scan.py --start-date 2023-01-01 --end-date 2024-12-31 --output outputs/teapot/topological_signals.csv --use-cache
"""

import argparse
import logging
from pathlib import Path

import polars as pl

from nq.config import DatabaseConfig, load_config
from nq.data.processor.teapot import TeapotDataLoader
from nq.trading.selector.teapot.topological_trend_scanner import TopologicalTrendScanner
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


def _first_of_run(df: pl.DataFrame, signal_col: str, ts_code_col: str = "ts_code") -> pl.DataFrame:
    """Keep only the first date of each consecutive run of True for signal_col per ts_code."""
    if df.is_empty():
        return df
    df = df.sort([ts_code_col, "trade_date"])
    df = df.with_columns(pl.col("trade_date").shift(1).over(ts_code_col).alias("_prev_date"))
    prev = df.select([ts_code_col, "trade_date", signal_col]).rename({
        "trade_date": "_prev_date",
        signal_col: "_prev_sig",
    })
    df = df.join(prev, on=[ts_code_col, "_prev_date"], how="left")
    df = df.filter(pl.col(signal_col))
    df = df.filter(pl.col("_prev_date").is_null() | pl.col("_prev_sig").is_null() | (pl.col("_prev_sig") == False))
    df = df.drop(["_prev_date", "_prev_sig"])
    return df


def run_scan(
    df: pl.DataFrame,
    scanner: TopologicalTrendScanner,
    output_path: str,
) -> None:
    """Run topological scanner, dedupe consecutive signals (first only), write to file with stage intervals."""
    analyzed = scanner.analyze(df, ts_code_col="ts_code")

    out_rows = []
    for ts_code in analyzed["ts_code"].unique().to_list():
        sub = analyzed.filter(pl.col("ts_code") == ts_code).sort("trade_date")
        s4 = _first_of_run(sub, "signal_4", "ts_code")
        for row in s4.iter_rows(named=True):
            out_rows.append({
                "ts_code": row["ts_code"],
                "trade_date": row["trade_date"],
                "signal_type": 4,
                "d_start": row.get("d_start"),
                "d_end": row.get("d_end"),
                "r_start": row.get("r_start"),
                "r_end": row.get("r_end"),
                "u_start": None,
                "u_end": None,
                "s4_start": None,
                "s4_end": None,
            })
        s6 = _first_of_run(sub, "signal_6", "ts_code")
        for row in s6.iter_rows(named=True):
            out_rows.append({
                "ts_code": row["ts_code"],
                "trade_date": row["trade_date"],
                "signal_type": 6,
                "d_start": None,
                "d_end": None,
                "r_start": row.get("r2_start"),  # s1 (R) for signal_6
                "r_end": row.get("r2_end"),
                "u_start": row.get("u_start"),  # s2 (U) for signal_6
                "u_end": row.get("u_end"),
                "s4_start": row.get("s4_start"),
                "s4_end": row.get("s4_end"),
            })

    if not out_rows:
        logger.warning("No signals to write")
        out_df = pl.DataFrame(schema={
            "ts_code": pl.Utf8,
            "trade_date": pl.Utf8,
            "signal_type": pl.Int32,
            "d_start": pl.Utf8,
            "d_end": pl.Utf8,
            "r_start": pl.Utf8,
            "r_end": pl.Utf8,
            "u_start": pl.Utf8,
            "u_end": pl.Utf8,
            "s4_start": pl.Utf8,
            "s4_end": pl.Utf8,
        })
    else:
        out_df = pl.DataFrame(out_rows)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.write_csv(output_path)
    n4 = len(out_df.filter(pl.col("signal_type") == 4)) if not out_df.is_empty() else 0
    n6 = len(out_df.filter(pl.col("signal_type") == 6)) if not out_df.is_empty() else 0
    logger.info(
        "Wrote %d signals to %s (signal_4=%d, signal_6=%d). stage_signal4_* only for signal_6.",
        len(out_df),
        output_path,
        n4,
        n6,
    )

    # Log current slice per stock
    for ts_code in analyzed["ts_code"].unique().to_list():
        sub = analyzed.filter(pl.col("ts_code") == ts_code)
        last = sub.tail(1)
        if last.is_empty():
            continue
        row = last.to_dicts()[0]
        logger.info(
            "Topological current slice %s: trade_date=%s s_up=%s s_down=%s s_relay=%s last_relay_high=%s current_stage=%s",
            ts_code,
            row.get("trade_date"),
            row.get("s_up"),
            row.get("s_down"),
            row.get("s_relay"),
            row.get("last_relay_high"),
            row.get("current_stage"),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Teapot topological trend scanner")
    parser.add_argument("--start-date", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--symbols", type=str, nargs="*", help="Stock codes (e.g. 600487 or 600487.SH)")
    parser.add_argument("--output", type=str, default="outputs/teapot/topological_signals.csv", help="Output CSV path")
    parser.add_argument("--relay-std-ratio", type=float, default=0.007, help="Max std(ma2,ma3,ma5)/ma5 for relay (default: 0.007)")
    parser.add_argument("--box-ceiling-window", type=int, default=15, help="Window size for box_ceiling (recent N days of non-up states, default: 15)")
    parser.add_argument("--ma-diff-threshold", type=float, default=0.003, help="Threshold for ma2-ma3 diff relative to ma5 for relay (default: 0.003)")
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

    loader = TeapotDataLoader(db_config=db_config, schema="quant", use_cache=args.use_cache)
    logger.info("Loading data %s to %s", args.start_date, args.end_date)
    df = loader.load_daily_data(
        start_date=args.start_date,
        end_date=args.end_date,
        symbols=symbols,
    )
    if df.is_empty():
        logger.error("No data loaded")
        return

    scanner = TopologicalTrendScanner(
        relay_std_ratio=args.relay_std_ratio,
        box_ceiling_window=args.box_ceiling_window,
        ma_diff_threshold=args.ma_diff_threshold,
    )
    run_scan(df, scanner, args.output)


if __name__ == "__main__":
    main()
