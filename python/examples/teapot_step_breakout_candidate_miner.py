# -*- coding: utf-8 -*-
"""
Teapot Step-Breakout Candidate Miner (选矿机).

Scans a date range and stock list, uses fuzzy logic to find candidate points
with "阶梯下跌后尝试突破" (step-down then breakout attempt) pattern.
Saves visualization images and data slices for manual labeling.

Usage:
    python teapot_step_breakout_candidate_miner.py --start-date 2018-01-01 --end-date 2023-12-31
    python teapot_step_breakout_candidate_miner.py --start-date 2020-01-01 --end-date 2022-12-31 --output outputs/teapot/candidates_pool
    python teapot_step_breakout_candidate_miner.py --start-date 2022-01-01 --end-date 2023-06-30 --max-candidates 500

Arguments:
    --start-date       Start date (YYYY-MM-DD)
    --end-date         End date (YYYY-MM-DD)
    --symbols          Optional list of stock codes (default: all)
    --output           Output directory (default: outputs/teapot/candidates_pool)
    --window-size      Lookback window in days (default: 60)
    --future-days      Future days for labeling reference (default: 10)
    --min-drop         Min (high-low)/high in window (default: 0.20, 20%)
    --min-rebound      Min (close - low)/low from bottom (default: 0.10, 10%)
    --max-candidates   Max number of candidates to save (default: 2000)
    --use-cache        Use cached data if available
"""

import argparse
import logging
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import polars as pl

from nq.config import DatabaseConfig, load_config
from nq.data.processor.teapot import TeapotDataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_candidate_points(
    df: pl.DataFrame,
    ts_code: str,
    window_size: int = 60,
    future_days: int = 10,
    min_drop: float = 0.20,
    min_rebound: float = 0.10,
    require_high_in_first_half: bool = False,
) -> List[dict]:
    """
    Find candidate points with 'step-down + V-reversal breakout' pattern in one stock.

    Rules (all required):
    1. Deep drawdown: (window high - window low) / high >= min_drop.
    2. Step-down: mean(close) of 3 segments (0-20, 20-40, 40-60) strictly descending.
    3. Breakout: current close >= max(high) of past 15 days (excluding today).
    4. V-reversal: (current close - window low) / window low >= min_rebound.
    5. Optional: window high occurs in first half of window.

    Args:
        df: Single-stock DataFrame sorted by trade_date, columns: open, high, low, close, volume.
        ts_code: Stock code.
        window_size: Lookback window (default: 60).
        future_days: Days after T for future_ret (default: 10).
        min_drop: Min relative drop in window (default: 0.20).
        min_rebound: Min rebound from window low (default: 0.10).
        require_high_in_first_half: If True, require 60d high to occur in first 30 days.

    Returns:
        List of dicts: index (row in df), date, ts_code, future_ret.
    """
    candidates: List[dict] = []
    n = len(df)
    if n < window_size + future_days:
        return candidates

    dates = df["trade_date"]
    high = df["high"]
    low = df["low"]
    close = df["close"]

    for t in range(window_size, n - future_days):
        # Window [t - window_size, t] inclusive (window_size + 1 points)
        w_high = high.slice(t - window_size, window_size)
        w_low = low.slice(t - window_size, window_size)
        w_close = close.slice(t - window_size, window_size)

        p_max = float(w_high.max())
        p_min = float(w_low.min())
        current_close = float(close[t])

        # 1. Deep drawdown
        if p_max <= 0 or (p_max - p_min) / p_max < min_drop:
            continue

        # 2. Step-down: part1 > part2 > part3 (mean close of 3 segments)
        seg_len = window_size // 3  # 20, 20, 20
        part1 = float(w_close.slice(0, seg_len).mean())
        part2 = float(w_close.slice(seg_len, seg_len).mean())
        part3 = float(w_close.slice(seg_len * 2, seg_len).mean())
        if not (part1 > part2 > part3):
            continue

        # 3. Breakout: close >= max(high[-15:-1])
        recent_high = float(high.slice(max(0, t - 15), min(15, t)).max())
        if current_close < recent_high:
            continue

        # 4. V-reversal: rebound from window low
        if p_min <= 0 or (current_close - p_min) / p_min < min_rebound:
            continue

        # 5. Optional: high in first half
        if require_high_in_first_half:
            first_half_high = float(w_high.slice(0, window_size // 2).max())
            second_half_high = float(w_high.slice(window_size // 2, window_size // 2).max())
            if first_half_high < second_half_high:
                continue

        future_close = float(close[t + future_days])
        future_ret = (future_close / current_close) - 1.0

        candidates.append({
            "index": t,
            "date": dates[t],
            "ts_code": ts_code,
            "future_ret": future_ret,
        })

    return candidates


def save_candidate_assets(
    df: pl.DataFrame,
    candidate: dict,
    save_dir: Path,
    window_size: int,
    future_days: int,
    max_filename_len: int = 120,
) -> None:
    """
    Save one candidate: PNG (history + future) and .npy slice for labeling.

    Args:
        df: Single-stock DataFrame (same as passed to find_candidate_points).
        candidate: Dict with index, date, ts_code, future_ret.
        save_dir: Root output dir (images/ and data_slices/ under it).
        window_size: History days.
        future_days: Future days to plot.
        max_filename_len: Max length for filename (default: 120).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping image save")
        return

    t = candidate["index"]
    ts_code = candidate["ts_code"]
    date_val = candidate["date"]
    date_str = str(date_val).split(" ")[0].replace("-", "")
    future_ret = candidate["future_ret"]

    # Slice: history + future
    start = max(0, t - window_size)
    end = min(len(df), t + future_days + 1)
    plot_df = df.slice(start, end - start)

    close_vals = plot_df["close"].to_numpy()
    n_hist = min(window_size, t - start)
    n_fut = len(close_vals) - n_hist

    images_dir = save_dir / "images"
    data_dir = save_dir / "data_slices"
    images_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename: replace . with _
    sym_safe = ts_code.replace(".", "_")
    file_base = f"{sym_safe}_{date_str}_{future_ret:.2f}"
    if len(file_base) > max_filename_len:
        file_base = file_base[:max_filename_len]

    # Plot: blue history, orange future
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(n_hist), close_vals[:n_hist], color="blue", label="History")
    if n_fut > 0:
        ax.plot(
            range(n_hist, n_hist + n_fut),
            close_vals[n_hist:],
            color="orange",
            linestyle="--",
            label=f"Future_{future_days}d",
        )
    ax.axvline(x=n_hist, color="red", alpha=0.3)
    ax.set_title(f"{ts_code} | {date_str} | Future Ret: {future_ret:.2%}")
    ax.legend()
    ax.set_xlabel("Day")
    fig.tight_layout()
    fig.savefig(images_dir / f"{file_base}.png", dpi=100)
    plt.close(fig)

    # Data slice for training: window [t-60, t] inclusive
    slice_start = t - window_size
    slice_end = t + 1
    slice_df = df.slice(slice_start, slice_end - slice_start)
    snapshot = slice_df.select(["open", "high", "low", "close", "volume"]).to_numpy()
    np.save(data_dir / f"{file_base}.npy", snapshot)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teapot step-breakout candidate miner (选矿机) for manual labeling"
    )
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--symbols", type=str, nargs="*", help="Optional stock codes")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/teapot/candidates_pool",
        help="Output directory",
    )
    parser.add_argument("--window-size", type=int, default=60, help="Lookback window (default: 60)")
    parser.add_argument("--future-days", type=int, default=10, help="Future days for label ref (default: 10)")
    parser.add_argument(
        "--min-drop",
        type=float,
        default=0.20,
        help="Min (high-low)/high in window (default: 0.20)",
    )
    parser.add_argument(
        "--min-rebound",
        type=float,
        default=0.10,
        help="Min rebound from window low (default: 0.10)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=2000,
        help="Max candidates to save (default: 2000)",
    )
    parser.add_argument(
        "--high-in-first-half",
        action="store_true",
        help="Require 60d high to occur in first half of window",
    )
    parser.add_argument("--use-cache", action="store_true", help="Use cached data")

    args = parser.parse_args()

    try:
        config = load_config()
        db_config = config.database
    except Exception as e:
        logger.warning("Failed to load config: %s, using defaults", e)
        db_config = DatabaseConfig()

    loader = TeapotDataLoader(
        db_config=db_config,
        schema="quant",
        use_cache=args.use_cache,
    )

    logger.info("Loading data %s to %s", args.start_date, args.end_date)
    df_all = loader.load_daily_data(
        start_date=args.start_date,
        end_date=args.end_date,
        symbols=args.symbols if args.symbols else None,
    )

    if df_all.is_empty():
        logger.error("No data loaded")
        return

    logger.info("Loaded %d rows, %d stocks", len(df_all), df_all["ts_code"].n_unique())

    save_dir = Path(args.output)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_candidates: List[dict] = []
    for ts_code in df_all["ts_code"].unique().to_list():
        if len(all_candidates) >= args.max_candidates:
            break
        stock_df = (
            df_all.filter(pl.col("ts_code") == ts_code)
            .sort("trade_date")
            .select(["trade_date", "open", "high", "low", "close", "volume"])
        )
        res = find_candidate_points(
            stock_df,
            ts_code=ts_code,
            window_size=args.window_size,
            future_days=args.future_days,
            min_drop=args.min_drop,
            min_rebound=args.min_rebound,
            require_high_in_first_half=args.high_in_first_half,
        )
        all_candidates.extend(res)

    all_candidates = all_candidates[: args.max_candidates]
    logger.info("Found %d candidates (capped at %d)", len(all_candidates), args.max_candidates)

    for i, c in enumerate(all_candidates):
        ts_code = c["ts_code"]
        stock_df = (
            df_all.filter(pl.col("ts_code") == ts_code)
            .sort("trade_date")
            .select(["trade_date", "open", "high", "low", "close", "volume"])
        )
        save_candidate_assets(
            stock_df,
            c,
            save_dir=save_dir,
            window_size=args.window_size,
            future_days=args.future_days,
        )
        if (i + 1) % 100 == 0:
            logger.info("Saved %d / %d", i + 1, len(all_candidates))

    logger.info("Done. Images: %s/images  Data: %s/data_slices", save_dir, save_dir)


if __name__ == "__main__":
    main()
