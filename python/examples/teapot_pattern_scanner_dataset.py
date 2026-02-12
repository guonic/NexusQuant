# -*- coding: utf-8 -*-
"""
Teapot Pattern Scanner Dataset Generator.

Generates PyTorch-ready tensor dataset from labeled market data:
1. Automatic labeling using PatternLabeler (runs 3 times with platform_lookback=20,40,60)
2. Extract 60-day sequences for positive and negative samples
3. Normalize features and save as .pt files

Usage:
    python teapot_pattern_scanner_dataset.py \
        --start-date 2015-01-01 \
        --end-date 2024-12-31 \
        --output outputs/teapot/pattern_scanner_dataset \
        --use-cache

Arguments:
    --start-date        Start date (YYYY-MM-DD)
    --end-date          End date (YYYY-MM-DD)
    --symbols           Optional comma-separated list of stock codes (e.g., "600487.SH,601688.SH")
    --output            Output directory for dataset files
    --use-cache         Use cache when loading data
    --seq-len           Sequence length (default: 60)
    --negative-ratio    Ratio of negative samples to positive samples (default: 1.0)
    --seed              Random seed for negative sample sampling (default: 42)
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import polars as pl
import torch

from nq.config import DatabaseConfig, load_config
from nq.data.processor.teapot import TeapotDataLoader
from nq.trading.selector.teapot.pattern_scanner import PatternLabeler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def normalize_slice(data_slice: np.ndarray) -> np.ndarray:
    """
    Normalize sequence slice: [open, high, low, close, volume].

    - Price (cols 0-3): single min-max over O,H,L,C.
    - Volume (col 4): independent min-max.

    Args:
        data_slice: Array of shape [seq_len, 5] with columns [open, high, low, close, volume].

    Returns:
        Normalized array of shape [5, seq_len] (channels-first).
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


def create_tensor_dataset(
    df: pl.DataFrame,
    seq_len: int = 60,
    negative_ratio: float = 1.0,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Convert labeled Polars DataFrame to PyTorch tensor dataset.

    Args:
        df: Labeled DataFrame with columns: ts_code, trade_date, open, high, low, close, volume, label.
        seq_len: Sequence length (default: 60).
        negative_ratio: Ratio of negative samples to positive samples.
        seed: Random seed for negative sample sampling.

    Returns:
        Tuple of (features_tensor, labels_tensor, metadata_dict).
        features_tensor: Shape [N, 5, seq_len] (N samples, 5 features, seq_len time steps).
        labels_tensor: Shape [N] (binary labels).
        metadata_dict: Dictionary with metadata (symbols, dates, etc.).
    """
    # Feature columns
    feature_cols = ["open", "high", "low", "close", "volume"]

    # Extract positive sample indices
    positive_mask = df["label"].to_numpy() == 1
    positive_indices = np.where(positive_mask)[0]

    logger.info(f"Found {len(positive_indices)} positive samples")

    # Sample negative samples
    negative_mask = df["label"].to_numpy() == 0
    negative_indices = np.where(negative_mask)[0]

    n_positive = len(positive_indices)
    n_negative = int(n_positive * negative_ratio)

    if len(negative_indices) < n_negative:
        logger.warning(
            f"Not enough negative samples: {len(negative_indices)} < {n_negative}, "
            f"using all {len(negative_indices)} negative samples"
        )
        sampled_negatives = negative_indices
    else:
        rng = np.random.RandomState(seed)
        sampled_negatives = rng.choice(negative_indices, size=n_negative, replace=False)

    logger.info(f"Sampled {len(sampled_negatives)} negative samples")

    # Combine indices
    target_indices = np.concatenate([positive_indices, sampled_negatives])
    target_indices = np.sort(target_indices)

    logger.info(f"Total samples: {len(target_indices)}")

    # Build feature matrix [N, seq_len, Features]
    features_list = []
    labels_list = []
    symbols_list = []
    dates_list = []

    for idx in target_indices:
        if idx < seq_len - 1:
            continue

        # Extract sequence
        start_idx = idx - seq_len + 1
        seq_df = df.slice(start_idx, seq_len)

        # Check if sequence is complete
        if len(seq_df) != seq_len:
            continue

        # Extract features
        feature_array = seq_df.select(feature_cols).to_numpy()

        # Normalize
        try:
            normalized = normalize_slice(feature_array)
            features_list.append(normalized)
            labels_list.append(df["label"][idx])
            symbols_list.append(df["ts_code"][idx])
            dates_list.append(df["trade_date"][idx])
        except Exception as e:
            logger.debug(f"Skip index {idx}: {e}")
            continue

    if not features_list:
        logger.warning("No valid samples extracted")
        return (
            torch.empty(0, 5, seq_len),
            torch.empty(0, dtype=torch.long),
            {},
        )

    # Convert to tensors
    features_tensor = torch.stack([torch.from_numpy(f) for f in features_list])
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    # Metadata
    metadata = {
        "symbols": symbols_list,
        "dates": [str(d) for d in dates_list],
        "n_positive": sum(labels_list),
        "n_negative": len(labels_list) - sum(labels_list),
        "seq_len": seq_len,
    }

    logger.info(
        f"Created dataset: {len(features_list)} samples "
        f"({metadata['n_positive']} positive, {metadata['n_negative']} negative)"
    )

    return features_tensor, labels_tensor, metadata


def generate_dataset(
    start_date: str,
    end_date: str,
    output_dir: Path,
    symbols: Optional[List[str]] = None,
    use_cache: bool = False,
    seq_len: int = 60,
    negative_ratio: float = 1.0,
    seed: int = 42,
    volatility_threshold: float = 0.03,
) -> None:
    """
    Generate pattern scanner dataset.

    Args:
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        output_dir: Output directory for dataset files.
        symbols: Optional list of stock codes.
        use_cache: Use cache when loading data.
        seq_len: Sequence length (default: 60).
        negative_ratio: Ratio of negative samples to positive samples.
        seed: Random seed for negative sample sampling.
        volatility_threshold: Volatility threshold for platform detection.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    logger.info(f"Loaded {len(market_data)} rows for {market_data['ts_code'].n_unique()} symbols")

    # Ensure required columns exist
    required_cols = ["ts_code", "trade_date", "open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in market_data.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return

    # Sort by symbol and date
    market_data = market_data.sort(["ts_code", "trade_date"])

    # Generate labels using multiple windows
    logger.info("Generating labels with multiple platform windows (20, 40, 60)...")
    labeler = PatternLabeler(volatility_threshold=volatility_threshold)
    labeled_data = labeler.label_multi_window(
        market_data, lookback_windows=[20, 40, 60]
    )

    # Count labels
    label_counts = labeled_data["label"].value_counts().sort("label")
    logger.info(f"Label distribution:\n{label_counts}")

    # Create tensor dataset
    logger.info("Creating tensor dataset...")
    features_tensor, labels_tensor, metadata = create_tensor_dataset(
        labeled_data,
        seq_len=seq_len,
        negative_ratio=negative_ratio,
        seed=seed,
    )

    if len(features_tensor) == 0:
        logger.error("No valid samples generated")
        return

    # Save dataset
    dataset_path = output_dir / "dataset.pt"
    torch.save(
        {
            "features": features_tensor,
            "labels": labels_tensor,
            "metadata": metadata,
        },
        dataset_path,
    )

    logger.info(f"Saved dataset to {dataset_path}")
    logger.info(f"Dataset shape: {features_tensor.shape}")
    logger.info(f"Labels shape: {labels_tensor.shape}")
    logger.info(f"Positive samples: {metadata['n_positive']}")
    logger.info(f"Negative samples: {metadata['n_negative']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pattern scanner dataset for training"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Optional comma-separated list of stock codes (e.g., '600487.SH,601688.SH')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/teapot/pattern_scanner_dataset",
        help="Output directory for dataset files",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cache when loading data",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=60,
        help="Sequence length (default: 60)",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=1.0,
        help="Ratio of negative samples to positive samples (default: 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for negative sample sampling (default: 42)",
    )
    parser.add_argument(
        "--volatility-threshold",
        type=float,
        default=0.03,
        help="Volatility threshold for platform detection (default: 0.03)",
    )

    args = parser.parse_args()

    symbols_list = None
    if args.symbols:
        symbols_list = [s.strip() for s in args.symbols.split(",")]

    generate_dataset(
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=Path(args.output),
        symbols=symbols_list,
        use_cache=args.use_cache,
        seq_len=args.seq_len,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
        volatility_threshold=args.volatility_threshold,
    )


if __name__ == "__main__":
    main()
