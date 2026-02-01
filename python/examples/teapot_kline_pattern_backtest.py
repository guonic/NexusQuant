# -*- coding: utf-8 -*-
"""
Teapot K-line Pattern Backtest (多尺度 CNN 信号回测).

Loads trained MultiScaleCNN, runs sliding-window inference on daily K-line data,
filters by probability threshold (e.g. > 0.9), and outputs hit time points only (no holding-period returns).

Usage:
    python teapot_kline_pattern_backtest.py --model outputs/teapot/models/kline_v_pattern.pth --start-date 2020-01-01 --end-date 2024-06-30
    python teapot_kline_pattern_backtest.py --model kline_v_pattern.pth --threshold 0.9 --output outputs/teapot/backtest_signals.csv

Arguments:
    --model         Path to .pth (MultiScaleCNN state_dict)
    --start-date    Backtest start (YYYY-MM-DD)
    --end-date      Backtest end (YYYY-MM-DD)
    --symbols       Optional stock codes (default: all from data)
    --threshold     Min probability to emit signal (default: 0.9)
    --window-size   Lookback window (default: 60)
    --output        CSV path for hit signals (ts_code, trade_date, prob)
    --use-cache     Use cached data
"""

import argparse
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import polars as pl
import torch

from nq.ai.teapot_pattern.dataset import normalize_slice_for_model
from nq.ai.teapot_pattern.model import MultiScaleCNN
from nq.config import DatabaseConfig, load_config
from nq.data.processor.teapot import TeapotDataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_inference(
    df: pl.DataFrame,
    model: torch.nn.Module,
    device: torch.device,
    window_size: int = 60,
    batch_size: int = 64,
) -> pl.DataFrame:
    """
    Sliding 60-day window over each stock; run model in batches; return DataFrame with ts_code, trade_date, prob.
    """
    model.eval()
    rows: List[dict] = []
    batch_tensors: List[torch.Tensor] = []
    batch_meta: List[Tuple[str, Any]] = []

    def flush_batch() -> None:
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors, dim=0).to(device)
        with torch.no_grad():
            probs = model(x).cpu().numpy()
        for (ts_code, trade_date), p in zip(batch_meta, probs):
            rows.append({"ts_code": ts_code, "trade_date": trade_date, "prob": round(float(p), 4)})
        batch_tensors.clear()
        batch_meta.clear()

    for ts_code in df["ts_code"].unique().to_list():
        stock = (
            df.filter(pl.col("ts_code") == ts_code)
            .sort("trade_date")
            .select(["trade_date", "open", "high", "low", "close", "volume"])
        )
        n = len(stock)
        if n < window_size + 1:
            continue
        for t in range(window_size - 1, n):
            slice_df = stock.slice(t - window_size + 1, window_size)
            arr = slice_df.select(["open", "high", "low", "close", "volume"]).to_numpy()
            try:
                norm = normalize_slice_for_model(arr)
            except Exception:
                continue
            batch_tensors.append(torch.from_numpy(norm).float())
            batch_meta.append((ts_code, stock["trade_date"][t]))
            if len(batch_tensors) >= batch_size:
                flush_batch()
    flush_batch()

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Teapot K-line pattern signals")
    parser.add_argument("--model", type=str, required=True, help="Path to .pth (MultiScaleCNN state_dict)")
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--symbols", type=str, nargs="*", help="Optional stock codes")
    parser.add_argument("--threshold", type=float, default=0.9, help="Min prob for signal (default: 0.9)")
    parser.add_argument("--window-size", type=int, default=60, help="Lookback window (default: 60)")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/teapot/backtest_signals.csv",
        help="Output CSV for hit signals (ts_code, trade_date, prob)",
    )
    parser.add_argument("--use-cache", action="store_true", help="Use cached data")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiScaleCNN()
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)

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
        symbols=args.symbols if args.symbols else None,
    )
    if df.is_empty():
        logger.error("No data loaded")
        return

    logger.info("Running inference (window=%d)...", args.window_size)
    probs_df = run_inference(df, model, device, window_size=args.window_size)
    if probs_df.is_empty():
        logger.warning("No inference results")
        return

    signals = probs_df.filter(pl.col("prob") >= args.threshold)
    logger.info("Hit time points (prob >= %.2f): %d", args.threshold, len(signals))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    signals.write_csv(args.output)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
