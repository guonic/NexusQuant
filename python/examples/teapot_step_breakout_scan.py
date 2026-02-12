# -*- coding: utf-8 -*-
"""
Teapot Step Breakout Scanner (阶梯式突破扫描).

Loads daily K-line, runs StepBreakoutScanner, determines whether current slice
is in stage 4 (first breakout / buy1) or stage 6 (second breakout / buy2),
and logs detailed stage time intervals.

Usage:
  python teapot_step_breakout_scan.py --start-date 2023-01-01 --end-date 2024-12-31 --symbols 600487.SH
  python teapot_step_breakout_scan.py --start-date 2023-01-01 --end-date 2024-12-31 --symbols 600487 688568 --use-cache
"""

import argparse
import logging

import polars as pl

from nq.config import DatabaseConfig, load_config
from nq.data.processor.teapot import TeapotDataLoader
from nq.trading.selector.teapot.step_breakout_scanner import (
    StepBreakoutResult,
    StepBreakoutScanner,
)
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
    df,
    scanner: StepBreakoutScanner,
) -> None:
    """
    Run scanner, determine current-slice stage 4/6, and log stage intervals.
    """
    results = scanner.get_current_slice_results(df, ts_code_col="ts_code")
    if not results:
        logger.warning("No current-slice results (empty data or insufficient bars)")
        return

    for r in results:
        logger.info("========== StepBreakout current slice ==========")
        _log_result(r, scanner)

    # Also list all signal_4 and signal_6 points in history
    analyzed = scanner.analyze(df, ts_code_col="ts_code")
    for ts_code in analyzed["ts_code"].unique().to_list():
        sub = analyzed.filter(pl.col("ts_code") == ts_code)
        s4 = sub.filter(pl.col("signal_4"))
        s6 = sub.filter(pl.col("signal_6"))
        if not s4.is_empty() or not s6.is_empty():
            logger.info("---------- %s: historical stage 4/6 points ----------", ts_code)
            for row in s4.iter_rows(named=True):
                logger.info("  signal_4 (buy1): trade_date=%s", row.get("trade_date"))
            for row in s6.iter_rows(named=True):
                logger.info("  signal_6 (buy2): trade_date=%s", row.get("trade_date"))


def _log_result(result: StepBreakoutResult, scanner: StepBreakoutScanner) -> None:
    """Log one StepBreakoutResult with stage intervals."""
    logger.info(
        "ts_code=%s trade_date=%s current_stage=%s in_stage_4=%s in_stage_6=%s",
        result.ts_code,
        result.trade_date,
        result.current_stage,
        result.in_stage_4,
        result.in_stage_6,
    )
    for iv in result.intervals:
        msg = "  Stage %d: %s ~ %s" % (iv.stage, iv.start_date or "", iv.end_date or "")
        if iv.high is not None and iv.low is not None:
            msg += " high=%.4f low=%.4f" % (iv.high, iv.low)
        if iv.extra:
            msg += " (%s)" % iv.extra
        logger.info(msg)
    if result.power_ratio is not None:
        above = ">= threshold" if result.power_ratio >= scanner.power_threshold else "< threshold"
        logger.info(
            "  power_ratio=%.4f (threshold=%.2f, %s) [|stage1_decline_slope|/up_slope]",
            result.power_ratio,
            scanner.power_threshold,
            above,
        )
    if result.volume_ratio_6_to_4 is not None:
        logger.info(
            "  volume_ratio_6_to_4=%.4f (stage6_vol/stage4_vol, >1 = 量能台阶)",
            result.volume_ratio_6_to_4,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Teapot step breakout scanner")
    parser.add_argument("--start-date", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--symbols", type=str, nargs="*", help="Stock codes (e.g. 600487 or 600487.SH)")
    parser.add_argument("--box-len", type=int, default=20, help="Stage-2 box length (default: 20)")
    parser.add_argument("--box-range", type=float, default=0.10, help="Stage-2 max width (default: 0.10)")
    parser.add_argument("--relay-box-range", type=float, default=0.05, help="Stage-5 max width (default: 0.05)")
    parser.add_argument("--power-threshold", type=float, default=1.5, help="Power ratio threshold, >1.5 = strong rebound (default: 1.5)")
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

    scanner = StepBreakoutScanner(
        box_len=args.box_len,
        box_range=args.box_range,
        relay_box_range=args.relay_box_range,
        power_threshold=args.power_threshold,
    )
    run_scan(df, scanner)


if __name__ == "__main__":
    main()
