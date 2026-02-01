# -*- coding: utf-8 -*-
"""
Teapot Negative Samples Generator (负样本生成).

Generates negative .pt samples by random 60-day slices from the same stocks
(as positive seed), excluding dates that are in the positive set.
Use with --pos-dir to avoid overlap, then feed --neg-dir to train script.

Usage:
    python teapot_neg_samples_generator.py --pos-dir outputs/teapot/processed_samples --neg-dir outputs/teapot/neg_samples --count 500
    python teapot_neg_samples_generator.py --pos-dir data/pos --neg-dir data/neg --count 300 --data-source db --start-date 2018-01-01 --end-date 2024-06-30
"""

import argparse
import logging
import random
from pathlib import Path
from typing import Set, Tuple

import numpy as np
import polars as pl
import torch

from nq.ai.teapot_pattern.dataset import normalize_slice_for_model
from nq.config import DatabaseConfig, load_config
from nq.data.processor.teapot import TeapotDataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_pos_dates(pos_dir: Path) -> Set[Tuple[str, str]]:
    """From pos_dir .pt filenames like 601688_20240730.pt, return set of (symbol, YYYYMMDD)."""
    out: Set[Tuple[str, str]] = set()
    for f in pos_dir.glob("*.pt"):
        name = f.stem
        if "_" in name:
            parts = name.split("_")
            if len(parts) >= 2 and len(parts[1]) == 8:
                out.add((parts[0], parts[1]))
    return out


def symbol_to_ts_code(symbol: str) -> str:
    s = str(symbol).strip()
    if "." in s:
        return s
    if s.startswith(("6", "5", "9")):
        return f"{s}.SH"
    return f"{s}.SZ"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate negative .pt samples (random 60-day slices)")
    parser.add_argument("--pos-dir", type=str, required=True, help="Positive .pt dir (to infer symbols and exclude dates)")
    parser.add_argument("--neg-dir", type=str, required=True, help="Output dir for negative .pt")
    parser.add_argument("--count", type=int, default=500, help="Number of negative samples")
    parser.add_argument("--data-source", type=str, choices=["db"], default="db")
    parser.add_argument("--start-date", type=str, default="2017-01-01")
    parser.add_argument("--end-date", type=str, default="2024-12-31")
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    pos_dir = Path(args.pos_dir)
    neg_dir = Path(args.neg_dir)
    neg_dir.mkdir(parents=True, exist_ok=True)

    exclude = parse_pos_dates(pos_dir) if pos_dir.exists() else set()
    symbols = list({s for s, _ in exclude}) if exclude else []
    if not symbols:
        logger.warning("No symbols from pos_dir; use --pos-dir with existing .pt files")
        return

    ts_codes = [symbol_to_ts_code(s) for s in symbols]
    try:
        config = load_config()
        db_config = config.database
    except Exception:
        db_config = DatabaseConfig()
    loader = TeapotDataLoader(db_config=db_config, schema="quant", use_cache=args.use_cache)
    df = loader.load_daily_data(start_date=args.start_date, end_date=args.end_date, symbols=ts_codes)
    if df.is_empty():
        logger.error("No data loaded")
        return

    candidates: list = []
    for ts_code in ts_codes:
        stock = df.filter(pl.col("ts_code") == ts_code).sort("trade_date")
        sym = ts_code.split(".")[0]
        n = len(stock)
        if n < args.window_size:
            continue
        for t in range(args.window_size - 1, n):
            d = str(stock["trade_date"][t])[:10].replace("-", "")
            if (sym, d) in exclude:
                continue
            candidates.append((ts_code, sym, t, stock))
    if len(candidates) < args.count:
        logger.warning("Only %d candidate slices (excl. pos); generating all", len(candidates))
        to_take = candidates
    else:
        to_take = random.sample(candidates, args.count)

    saved = 0
    for ts_code, sym, t, stock in to_take:
        slice_df = stock.slice(t - args.window_size + 1, args.window_size)
        arr = slice_df.select(["open", "high", "low", "close", "volume"]).to_numpy()
        try:
            norm = normalize_slice_for_model(arr)
        except Exception:
            continue
        date_str = str(stock["trade_date"][t])[:10].replace("-", "")
        name = f"{sym}_{date_str}_neg.pt"
        tensor = torch.from_numpy(norm).float()
        torch.save(tensor, neg_dir / name)
        saved += 1
    logger.info("Saved %d negative samples to %s", saved, neg_dir)


if __name__ == "__main__":
    main()
