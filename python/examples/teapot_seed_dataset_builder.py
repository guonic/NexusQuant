# -*- coding: utf-8 -*-
"""
Teapot Seed Dataset Builder (种子数据集构建).

Builds a PyTorch-ready seed dataset from (symbol, date) positive samples:
- Robust date parsing (Chinese comma, YYYYMMDD / YYYY-MM-DD).
- Data from AkShare or local DB (TeapotDataLoader).
- 60-day slice ending at T; price (O,H,L,C) and volume normalized separately.
- Saves (5, 60) tensors as .pt for training.

Usage:
    python teapot_seed_dataset_builder.py --samples samples.json --output processed_samples
    python teapot_seed_dataset_builder.py --samples samples.json --data-source db --output processed_samples
    python teapot_seed_dataset_builder.py --samples samples.json --data-source akshare --output processed_samples

Arguments:
    --samples        Path to JSON: {"601688": ["20240730", "20230404"], ...} (or use built-in default)
    --output         Output directory (default: outputs/teapot/processed_samples)
    --data-source    akshare | db (default: db)
    --window-size    Lookback days (default: 60)
    --use-cache      Use cache when data-source=db
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default positive samples (seed dataset)
DEFAULT_RAW_SAMPLES: Dict[str, List[str]] = {
    "601688": ["20240730", "20230404", "20210818", "20210528", "20210111", "20191213", "20181022"],
    "300792": ["20240930"],
    "600487": ["20220523", "20240227", "20210707", "20190909", "20190110", "20180712", "20180206"],
    "002155": ["20240115", "20240222", "20240927", "20180926"],
    "601208": ["20240830", "20240210", "20181101"],
    "300100": ["20240227", "20230511", "20210415"],
    "603799": ["20240710", "20240312", "20230720", "20220511", "20201029"],
}


def normalize_date_string(s: str) -> str:
    """
    Normalize date string: strip Chinese comma、space, keep digits and hyphen.

    Examples: "2024，07，30" -> "20240730", "2024-07-30" -> "20240730".
    """
    s = str(s).strip()
    s = re.sub(r"[\s，,、]", "", s)
    s = s.replace("-", "")
    if len(s) == 8 and s.isdigit():
        return s
    if len(s) >= 8:
        digits = "".join(c for c in s if c.isdigit())
        if len(digits) >= 8:
            return digits[:8]
    return s


def parse_date_to_yyyymmdd(s: str) -> Optional[str]:
    """Parse flexible date string to YYYYMMDD (8 chars)."""
    norm = normalize_date_string(s)
    if len(norm) == 8 and norm.isdigit():
        return norm
    return None


def normalize_slice(data_slice: np.ndarray) -> np.ndarray:
    """
    Normalize 60 x 5 slice: [open, high, low, close, volume].
    - Price (cols 0-3): single min-max over O,H,L,C.
    - Volume (col 4): independent min-max.
    Returns (5, 60) for channels-first.
    """
    data_slice = np.asarray(data_slice, dtype=np.float64)
    if data_slice.size == 0:
        raise ValueError("Empty slice")

    prices = data_slice[:, :4]
    p_min, p_max = np.min(prices), np.max(prices)
    norm_prices = (prices - p_min) / (p_max - p_min + 1e-9)

    v = data_slice[:, 4:5]
    v_min, v_max = np.min(v), np.max(v)
    norm_v = (v - v_min) / (v_max - v_min + 1e-9)

    combined = np.hstack((norm_prices, norm_v))
    return combined.T.astype(np.float32)


def symbol_to_ts_code(symbol: str) -> str:
    """Map 6-digit symbol to ts_code (e.g. 601688 -> 601688.SH, 002020 -> 002020.SZ)."""
    s = str(symbol).strip()
    if "." in s:
        return s
    if s.startswith(("6", "5", "9")):
        return f"{s}.SH"
    return f"{s}.SZ"


def get_stock_data_akshare(symbol: str, start_date: str = "20170101"):
    """Fetch A-share daily (qfq) via AkShare. Returns DataFrame index=date, columns open/high/low/close/volume."""
    try:
        import pandas as pd
        import akshare as ak
    except ImportError as e:
        logger.error("AkShare/pandas required for data-source=akshare: %s", e)
        return None

    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, adjust="qfq")
        if df is None or df.empty:
            return None
        df.columns = [c.strip() for c in df.columns]
        if "日期" in df.columns:
            df["日期"] = pd.to_datetime(df["日期"])
            df.set_index("日期", inplace=True)
        col_map = {"开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"}
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        for c in ["open", "high", "low", "close", "volume"]:
            if c not in df.columns:
                return None
        return df[["open", "high", "low", "close", "volume"]].sort_index()
    except Exception as e:
        logger.warning("AkShare fetch %s: %s", symbol, e)
        return None


def load_all_data_db(
    symbols: List[str],
    start_date: str,
    end_date: str,
    use_cache: bool = False,
):
    """Load daily data for given symbols from TeapotDataLoader. Returns Polars DataFrame."""
    from nq.config import DatabaseConfig, load_config
    from nq.data.processor.teapot import TeapotDataLoader

    try:
        config = load_config()
        db_config = config.database
    except Exception as e:
        logger.warning("Load config failed: %s, using defaults", e)
        db_config = DatabaseConfig()

    loader = TeapotDataLoader(db_config=db_config, schema="quant", use_cache=use_cache)
    ts_codes = [symbol_to_ts_code(s) for s in symbols]
    df = loader.load_daily_data(start_date=start_date, end_date=end_date, symbols=ts_codes)
    if df.is_empty():
        return None
    return df


def build_seed_dataset(
    raw_samples: Dict[str, List[str]],
    output_dir: Path,
    data_source: str = "db",
    window_size: int = 60,
    use_cache: bool = False,
) -> Tuple[int, List[str]]:
    """
    For each (symbol, date) in raw_samples: slice 60 days, normalize, save .pt.

    Returns (success_count, list of saved file paths).
    """
    try:
        import torch
    except ImportError:
        logger.error("PyTorch required: pip install torch")
        return 0, []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all (symbol, date_yyyymmdd) and infer date range
    flat: List[Tuple[str, str]] = []
    for symbol, dates in raw_samples.items():
        for d in dates:
            yyyymmdd = parse_date_to_yyyymmdd(d)
            if yyyymmdd:
                flat.append((symbol, yyyymmdd))

    if not flat:
        logger.warning("No valid (symbol, date) after parsing")
        return 0, []

    symbols = list({s for s, _ in flat})
    start_date = min(d for _, d in flat)
    end_date = max(d for _, d in flat)
    start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
    end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
    # Need extra window_size days before start_date for slicing
    from datetime import datetime, timedelta
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=window_size + 10)
    start_date = start_dt.strftime("%Y-%m-%d")

    if data_source == "akshare":
        stock_dfs: Dict[str, Any] = {}
        for sym in symbols:
            df = get_stock_data_akshare(sym, start_date=start_date.replace("-", "")[:8])
            if df is not None:
                stock_dfs[sym] = df
        if not stock_dfs:
            logger.error("No data from AkShare")
            return 0, []

        saved = []
        for symbol, date_yyyymmdd in flat:
            if symbol not in stock_dfs:
                continue
            df = stock_dfs[symbol]
            try:
                target_date = np.datetime64(f"{date_yyyymmdd[:4]}-{date_yyyymmdd[4:6]}-{date_yyyymmdd[6:8]}")
                idx = df.index.get_indexer([target_date], method="nearest")[0]
                if idx < window_size:
                    continue
                slice_60 = df.iloc[idx - window_size + 1 : idx + 1].values
                if slice_60.shape[0] != window_size:
                    continue
                norm = normalize_slice(slice_60)
                tensor = torch.from_numpy(norm).float()
                name = f"{symbol}_{date_yyyymmdd}.pt"
                path = output_dir / name
                torch.save(tensor, path)
                saved.append(str(path))
            except Exception as e:
                logger.debug("Skip %s %s: %s", symbol, date_yyyymmdd, e)
        return len(saved), saved

    # data_source == "db"
    df_all = load_all_data_db(symbols, start_date, end_date, use_cache=use_cache)
    if df_all is None or df_all.is_empty():
        logger.error("No data from DB")
        return 0, []

    saved = []
    for symbol, date_yyyymmdd in flat:
        ts_code = symbol_to_ts_code(symbol)
        stock = df_all.filter(pl.col("ts_code") == ts_code).sort("trade_date")
        if stock.is_empty():
            continue
        target_str = f"{date_yyyymmdd[:4]}-{date_yyyymmdd[4:6]}-{date_yyyymmdd[6:8]}"
        try:
            row_idx = None
            for i in range(len(stock)):
                ds = str(stock["trade_date"][i])[:10]
                if ds == target_str:
                    row_idx = i
                    break
            if row_idx is None:
                for i in range(len(stock)):
                    ds = str(stock["trade_date"][i])[:10]
                    if ds >= target_str:
                        row_idx = i
                        break
                if row_idx is None:
                    row_idx = len(stock) - 1
            if row_idx < window_size - 1:
                continue
            start_idx = row_idx - window_size + 1
            slice_df = stock.slice(start_idx, window_size)
            if len(slice_df) != window_size:
                continue
            arr = slice_df.select(["open", "high", "low", "close", "volume"]).to_numpy()
            norm = normalize_slice(arr)
            tensor = torch.from_numpy(norm).float()
            name = f"{symbol}_{date_yyyymmdd}.pt"
            path = output_dir / name
            torch.save(tensor, path)
            saved.append(str(path))
        except Exception as e:
            logger.debug("Skip %s %s: %s", symbol, date_yyyymmdd, e)

    return len(saved), saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teapot seed dataset builder: (symbol, date) -> normalized (5, 60) .pt tensors"
    )
    parser.add_argument(
        "--samples",
        type=str,
        default=None,
        help="JSON file path: {\"601688\": [\"20240730\", ...], ...}. If not set, use built-in default.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/teapot/processed_samples",
        help="Output directory for .pt files",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["akshare", "db"],
        default="db",
        help="Data source: akshare or db (TeapotDataLoader)",
    )
    parser.add_argument("--window-size", type=int, default=60, help="Lookback window (default: 60)")
    parser.add_argument("--use-cache", action="store_true", help="Use cache when data-source=db")

    args = parser.parse_args()

    if args.samples:
        path = Path(args.samples)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw_samples = json.load(f)
        else:
            logger.error("Samples file not found: %s", args.samples)
            return
    else:
        raw_samples = DEFAULT_RAW_SAMPLES
        logger.info("Using built-in default samples (%d symbols)", len(raw_samples))

    n, paths = build_seed_dataset(
        raw_samples=raw_samples,
        output_dir=Path(args.output),
        data_source=args.data_source,
        window_size=args.window_size,
        use_cache=args.use_cache,
    )

    logger.info("Done. Saved %d positive samples to %s", n, args.output)
    if paths:
        logger.info("Example: %s", paths[0])


if __name__ == "__main__":
    main()
