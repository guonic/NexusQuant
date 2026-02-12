"""
Automatic label generator for pattern scanner.

Generates binary labels (0/1) based on platform detection and breakout detection.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class PatternLabeler:
    """
    Automatic label generator for pattern scanner.

    Labels are generated based on mathematical definitions:
    - Label 1: Has platform AND has breakout
    - Label 0: Otherwise
    """

    def __init__(
        self,
        platform_window_start: int = 20,
        platform_window_end: int = 40,
        volatility_threshold: float = 0.03,
    ):
        """
        Initialize pattern labeler.

        Args:
            platform_window_start: Start of platform window (days before current).
            platform_window_end: End of platform window (days before current).
            volatility_threshold: Maximum volatility ratio for platform detection.
        """
        self.platform_window_start = platform_window_start
        self.platform_window_end = platform_window_end
        self.volatility_threshold = volatility_threshold

    def detect_platform(
        self, df: pl.DataFrame, current_idx: int
    ) -> Tuple[bool, Optional[float]]:
        """
        Detect if there is a platform in the specified window.

        Args:
            df: DataFrame with columns: close, high, low.
            current_idx: Current index in DataFrame.

        Returns:
            Tuple of (is_platform, platform_high).
            is_platform: True if platform detected.
            platform_high: Maximum high price in platform window, None if no platform.
        """
        if current_idx < self.platform_window_end:
            return False, None

        start_idx = current_idx - self.platform_window_end
        end_idx = current_idx - self.platform_window_start

        if start_idx < 0 or end_idx < 0:
            return False, None

        window_data = df.slice(start_idx, end_idx - start_idx + 1)

        if window_data.is_empty():
            return False, None

        # Calculate volatility ratio
        closes = window_data["close"].to_numpy()
        price_mean = np.mean(closes)
        price_std = np.std(closes)

        if price_mean == 0:
            return False, None

        volatility_ratio = price_std / price_mean

        # Check if platform exists
        is_platform = volatility_ratio < self.volatility_threshold

        if is_platform:
            platform_high = window_data["high"].max()
            return True, platform_high

        return False, None

    def detect_breakout(self, current_close: float, platform_high: float) -> bool:
        """
        Detect if current price breaks out above platform high.

        Args:
            current_close: Current closing price.
            platform_high: Platform high price.

        Returns:
            True if breakout detected.
        """
        if platform_high is None:
            return False

        return current_close > platform_high

    def label_single(
        self, df: pl.DataFrame, current_idx: int
    ) -> Tuple[int, Optional[float], bool]:
        """
        Generate label for a single point.

        Args:
            df: DataFrame with columns: close, high, low.
            current_idx: Current index in DataFrame.

        Returns:
            Tuple of (label, platform_high, is_breakout).
            label: 1 if positive sample, 0 otherwise.
            platform_high: Platform high price, None if no platform.
            is_breakout: True if breakout detected.
        """
        # Detect platform
        is_platform, platform_high = self.detect_platform(df, current_idx)

        if not is_platform:
            return 0, None, False

        # Detect breakout
        current_close = df["close"][current_idx]
        is_breakout = self.detect_breakout(current_close, platform_high)

        # Generate label
        label = 1 if is_breakout else 0

        return label, platform_high, is_breakout

    def label(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate labels for entire DataFrame.

        Runs labeling with current platform window settings.

        Args:
            df: DataFrame with columns: ts_code, trade_date, close, high, low.
                Must be sorted by ts_code and trade_date.

        Returns:
            DataFrame with added columns:
            - label: Binary label (0 or 1)
            - platform_high: Platform high price (None if no platform)
            - is_breakout: Boolean indicating breakout
        """
        if df.is_empty():
            return df.with_columns([
                pl.lit(0).alias("label"),
                pl.lit(None).cast(pl.Float64).alias("platform_high"),
                pl.lit(False).alias("is_breakout"),
            ])

        # Group by symbol
        results = []
        for symbol in df["ts_code"].unique():
            symbol_df = df.filter(pl.col("ts_code") == symbol).sort("trade_date")

            labels = []
            platform_highs = []
            is_breakouts = []

            for i in range(len(symbol_df)):
                label, platform_high, is_breakout = self.label_single(symbol_df, i)
                labels.append(label)
                platform_highs.append(platform_high)
                is_breakouts.append(is_breakout)

            symbol_df = symbol_df.with_columns([
                pl.Series("label", labels),
                pl.Series("platform_high", platform_highs),
                pl.Series("is_breakout", is_breakouts),
            ])

            results.append(symbol_df)

        return pl.concat(results)

    def label_multi_window(
        self, df: pl.DataFrame, lookback_windows: list[int] = [20, 40, 60]
    ) -> pl.DataFrame:
        """
        Generate labels using multiple platform lookback windows.

        If any window detects a positive sample (label=1), the final label is 1.

        Args:
            df: DataFrame with columns: ts_code, trade_date, close, high, low.
            lookback_windows: List of platform window end positions.

        Returns:
            DataFrame with added columns:
            - label: Binary label (0 or 1), 1 if any window detects positive sample
            - platform_high: Platform high price from the first positive detection
            - is_breakout: Boolean indicating breakout
        """
        if df.is_empty():
            return df.with_columns([
                pl.lit(0).alias("label"),
                pl.lit(None).cast(pl.Float64).alias("platform_high"),
                pl.lit(False).alias("is_breakout"),
            ])

        # Save original window settings
        original_end = self.platform_window_end
        original_start = self.platform_window_start

        # Group by symbol
        results = []
        for symbol in df["ts_code"].unique():
            symbol_df = df.filter(pl.col("ts_code") == symbol).sort("trade_date")

            labels = []
            platform_highs = []
            is_breakouts = []

            for i in range(len(symbol_df)):
                # Try each lookback window
                final_label = 0
                final_platform_high = None
                final_is_breakout = False

                for lookback in lookback_windows:
                    # Temporarily set window
                    self.platform_window_end = lookback
                    self.platform_window_start = max(1, lookback - 20)

                    label, platform_high, is_breakout = self.label_single(symbol_df, i)

                    # If any window detects positive sample, mark as positive
                    if label == 1:
                        final_label = 1
                        if final_platform_high is None:
                            final_platform_high = platform_high
                        final_is_breakout = True
                        break  # Found positive sample, no need to check other windows

                labels.append(final_label)
                platform_highs.append(final_platform_high)
                is_breakouts.append(final_is_breakout)

            symbol_df = symbol_df.with_columns([
                pl.Series("label", labels),
                pl.Series("platform_high", platform_highs),
                pl.Series("is_breakout", is_breakouts),
            ])

            results.append(symbol_df)

        # Restore original window settings
        self.platform_window_end = original_end
        self.platform_window_start = original_start

        return pl.concat(results)
