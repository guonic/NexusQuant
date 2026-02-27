# -*- coding: utf-8 -*-
"""
Bottom Divergence Scanner - Last Year Auto Scan.

Automatically scans the last 365 days for bottom divergence patterns.

Usage:
    python bottom_divergence_scan_last_year.py
    python bottom_divergence_scan_last_year.py --output outputs/divergence/bottom_divergence_last_year.csv --use-cache
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import polars as pl

from nq.config import DatabaseConfig, load_config
from nq.data.processor.teapot import TeapotDataLoader
from nq.trading.selector.divergence import BottomDivergenceDetector
from nq.utils.data_normalize import normalize_stock_code

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_symbols(raw: Optional[str]) -> Optional[List[str]]:
    """Normalize and expand comma-separated symbols to ts_code list."""
    if not raw:
        return None

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    codes = [normalize_stock_code(p) for p in parts]
    seen = set()
    return [c for c in codes if c and c not in seen and not seen.add(c)]


def scan_bottom_divergence(
    start_date: str,
    end_date: str,
    output_path: str,
    symbols: Optional[List[str]] = None,
    use_cache: bool = False,
    lookback_period: int = 30,
    min_divergence_bars: int = 5,
) -> None:
    """
    Scan market for bottom divergence patterns.

    Args:
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        output_path: Output CSV file path.
        symbols: Optional list of stock codes.
        use_cache: Use cache when loading data.
        lookback_period: Lookback period for finding previous lows.
        min_divergence_bars: Minimum bars between two lows.
    """
    # Load configuration
    try:
        config = load_config()
        db_config = config.database
    except Exception as e:
        logger.warning(f"Load config failed: {e}, using defaults")
        db_config = DatabaseConfig()

    # Load data
    logger.info(f"Loading data from {start_date} to {end_date}")
    data_loader = TeapotDataLoader(
        db_config=db_config, schema="quant", use_cache=use_cache
    )

    market_data = data_loader.load_daily_data(
        start_date=start_date, end_date=end_date, symbols=symbols
    )

    if market_data.is_empty():
        logger.error("No market data loaded")
        return

    logger.info(
        f"Loaded {len(market_data)} rows for {market_data['ts_code'].n_unique()} symbols"
    )

    # Ensure required columns exist
    required_cols = ["ts_code", "trade_date", "open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in market_data.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return

    # Sort by symbol and date
    market_data = market_data.sort(["ts_code", "trade_date"])

    # Detect bottom divergence
    logger.info("Detecting bottom divergence patterns...")
    detector = BottomDivergenceDetector(
        lookback_period=lookback_period,
        min_divergence_bars=min_divergence_bars,
    )

    result_df = detector.detect(market_data)

    # Filter to only divergence signals
    signals_df = result_df.filter(pl.col("is_bottom_divergence") == True)

    if signals_df.is_empty():
        logger.warning("No bottom divergence patterns found")
        # Create empty DataFrame with schema
        signals_df = pl.DataFrame(
            schema={
                "ts_code": pl.Utf8,
                "trade_date": pl.Utf8,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "is_bottom_divergence": pl.Boolean,
                "divergence_type": pl.Utf8,
                "prev_low_idx": pl.Int64,
                "prev_low_date": pl.Utf8,
            }
        )
    else:
        # Select relevant columns for output
        signals_df = signals_df.select([
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "is_bottom_divergence",
            "divergence_type",
            "prev_low_date",
        ])

        logger.info(f"Found {len(signals_df)} bottom divergence signals")

        # Log statistics
        divergence_types = signals_df["divergence_type"].value_counts().sort("count")
        logger.info("Divergence type distribution:")
        for row in divergence_types.iter_rows(named=True):
            logger.info(f"  {row['divergence_type']}: {row['count']}")

    # Save to CSV
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    signals_df.write_csv(output_path)

    logger.info(f"Saved {len(signals_df)} signals to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan last 365 days for bottom divergence patterns"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/divergence/bottom_divergence_last_year.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Optional comma-separated list of stock codes",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cache when loading data",
    )
    parser.add_argument(
        "--lookback-period",
        type=int,
        default=30,
        help="Lookback period for finding previous lows (default: 30)",
    )
    parser.add_argument(
        "--min-divergence-bars",
        type=int,
        default=5,
        help="Minimum bars between two lows (default: 5)",
    )

    args = parser.parse_args()

    # Calculate dates (last 365 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    logger.info(f"Scanning from {start_date_str} to {end_date_str} (last 365 days)")

    symbols_list = _parse_symbols(args.symbols)

    scan_bottom_divergence(
        start_date=start_date_str,
        end_date=end_date_str,
        output_path=args.output,
        symbols=symbols_list,
        use_cache=args.use_cache,
        lookback_period=args.lookback_period,
        min_divergence_bars=args.min_divergence_bars,
    )


if __name__ == "__main__":
    main()
