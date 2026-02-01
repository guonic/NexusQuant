# -*- coding: utf-8 -*-
"""
Teapot DTW Golden Pattern Backtest (黄金形态模版库 相似度回测).

Sliding-window similarity vs golden templates (DTW). Pre-filters by drop and position
to reduce DTW calls. Outputs hit time points (lowest distance = most similar).

Usage:
    python teapot_dtw_backtest.py --templates-dir outputs/teapot/processed_samples --start-date 2020-01-01 --end-date 2024-06-30
    python teapot_dtw_backtest.py --templates-dir data/golden --symbols 600487 601688 --min-drop 0.20 --top-n 20 --output outputs/teapot/dtw_signals.csv

Arguments:
    --templates-dir  Directory of golden .pt (or .npy) files (5, 60) or (60,)
    --start-date    Backtest start (YYYY-MM-DD)
    --end-date      Backtest end (YYYY-MM-DD)
    --symbols       Optional stock codes (default: all from data)
    --min-drop      Pre-filter: skip window if (high_60 - low_60) / high_60 < this (default: 0.20)
    --position-min  Pre-filter: skip if close < high_60 * this (default: 0.75, i.e. near high)
    --top-n         Output only top N lowest-distance points per stock (default: 0 = all)
    --volume-penalty Alpha for score *= (1 + alpha * (1 - vol_ratio)) when vol_ratio < 1 (default: 0, disabled)
    --window-size   Lookback window (default: 60)
    --output        CSV path (ts_code, trade_date, similarity_score)
    --use-cache     Use cached data
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import polars as pl

from nq.ai.teapot_pattern.dtw_matcher import PatternMatcher
from nq.config import DatabaseConfig, load_config
from nq.data.processor.teapot import TeapotDataLoader
from nq.utils.data_normalize import normalize_stock_code

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _normalize_close(close: np.ndarray) -> np.ndarray:
    """MinMax normalize close to [0, 1] for DTW (same scale as seed .pt templates)."""
    close = np.asarray(close, dtype=np.float64).flatten()
    c_min, c_max = np.min(close), np.max(close)
    if c_max <= c_min:
        return np.zeros_like(close)
    return (close - c_min) / (c_max - c_min)


def backtest_similarity(
    df: pl.DataFrame,
    matcher: PatternMatcher,
    window_size: int = 60,
    min_drop: float = 0.20,
    position_min_ratio: float = 0.75,
    volume_penalty_alpha: float = 0.0,
) -> pl.DataFrame:
    """
    Sliding window over df; pre-filter by drop and position; compute DTW score.
    Returns DataFrame with ts_code, trade_date, similarity_score (lower = more similar).
    """
    rows: List[dict] = []
    for ts_code in df["ts_code"].unique().to_list():
        stock = (
            df.filter(pl.col("ts_code") == ts_code)
            .sort("trade_date")
            .select(["trade_date", "open", "high", "low", "close", "volume"])
        )
        n = len(stock)
        if n < window_size:
            continue
        high = stock["high"].to_numpy()
        low = stock["low"].to_numpy()
        close = stock["close"].to_numpy()
        volume = stock["volume"].to_numpy()
        dates = stock["trade_date"]

        for i in range(window_size - 1, n):
            w_high = high[i - window_size + 1 : i + 1]
            w_low = low[i - window_size + 1 : i + 1]
            w_close = close[i - window_size + 1 : i + 1]
            w_vol = volume[i - window_size + 1 : i + 1]
            h_max, l_min = float(np.max(w_high)), float(np.min(w_low))
            c_now = float(w_close[-1])

            # Pre-filter: drop
            if h_max <= 0 or (h_max - l_min) / h_max < min_drop:
                continue
            # Pre-filter: position (close near high)
            if c_now < h_max * position_min_ratio:
                continue

            norm_close = _normalize_close(w_close)
            score = matcher.get_score(norm_close)

            if volume_penalty_alpha > 0 and len(w_vol) > 0:
                vol_mean = np.mean(w_vol)
                if vol_mean > 0:
                    vol_ratio = float(w_vol[-1] / vol_mean)
                    if vol_ratio < 1.0:
                        score = score * (1.0 + volume_penalty_alpha * (1.0 - vol_ratio))

            rows.append({
                "ts_code": ts_code,
                "trade_date": dates[i],
                "similarity_score": round(float(score), 4),
            })

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="DTW golden pattern similarity backtest")
    parser.add_argument("--templates-dir", type=str, required=True, help="Directory of golden .pt/.npy files")
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--symbols", type=str, nargs="*", help="Optional stock codes (e.g. 600487 or 600487.SH)")
    parser.add_argument("--min-drop", type=float, default=0.20, help="Pre-filter: min (high-low)/high (default: 0.20)")
    parser.add_argument("--position-min", type=float, default=0.75, help="Pre-filter: close >= high_60 * this (default: 0.75)")
    parser.add_argument("--top-n", type=int, default=0, help="Output top N lowest-score per stock (0 = all)")
    parser.add_argument("--volume-penalty", type=float, default=0.0, help="Volume penalty alpha (0 = disabled)")
    parser.add_argument("--window-size", type=int, default=60, help="Lookback window (default: 60)")
    parser.add_argument("--output", type=str, default="outputs/teapot/dtw_signals.csv", help="Output CSV")
    parser.add_argument("--use-cache", action="store_true")

    args = parser.parse_args()

    templates_dir = Path(args.templates_dir)
    if not templates_dir.exists():
        logger.error("Templates dir not found: %s", templates_dir)
        return
    golden_files = sorted(templates_dir.glob("*.pt")) + sorted(templates_dir.glob("*.npy"))
    if not golden_files:
        logger.error("No .pt or .npy files in %s", templates_dir)
        return
    logger.info("Loading %d golden templates from %s", len(golden_files), templates_dir)
    matcher = PatternMatcher([str(p) for p in golden_files], top_k=3)

    try:
        config = load_config()
        db_config = config.database
    except Exception as e:
        logger.warning("Load config failed: %s", e)
        db_config = DatabaseConfig()

    # Normalize symbols to ts_code (e.g. 600487 -> 600487.SH); DB stores Tushare format
    symbols = None
    if args.symbols:
        symbols = [normalize_stock_code(s) for s in args.symbols]
        symbols = [s for s in symbols if s]
        if args.symbols and not symbols:
            logger.error("No valid symbols after normalization: %s", args.symbols)
            return
        logger.info("Resolved symbols: %s -> %s", args.symbols, symbols)

    loader = TeapotDataLoader(db_config=db_config, schema="quant", use_cache=args.use_cache)
    logger.info("Loading data %s to %s", args.start_date, args.end_date)
    df = loader.load_daily_data(
        start_date=args.start_date,
        end_date=args.end_date,
        symbols=symbols,
    )
    if df.is_empty():
        logger.error(
            "No data loaded. Check: 1) DB has daily kline for schema quant.stock_kline_day; "
            "2) ts_code format (e.g. 600487.SH); 3) date range %s to %s.",
            args.start_date,
            args.end_date,
        )
        return

    logger.info("Running DTW similarity (window=%d, min_drop=%.2f, position_min=%.2f)...",
                args.window_size, args.min_drop, args.position_min)
    result = backtest_similarity(
        df,
        matcher,
        window_size=args.window_size,
        min_drop=args.min_drop,
        position_min_ratio=args.position_min,
        volume_penalty_alpha=args.volume_penalty,
    )
    if result.is_empty():
        logger.warning("No rows passed pre-filters")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(schema={"ts_code": pl.Utf8, "trade_date": pl.Utf8, "similarity_score": pl.Float64}).write_csv(args.output)
        return

    if args.top_n > 0:
        result = result.sort("similarity_score").group_by("ts_code").head(args.top_n)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(args.output)
    logger.info("Wrote %d rows to %s (lower score = more similar)", len(result), args.output)


if __name__ == "__main__":
    main()
