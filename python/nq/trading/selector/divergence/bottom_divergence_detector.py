"""
Bottom divergence detector for daily K-line data.

Detects bottom divergence patterns where price makes new lows but technical indicators don't.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl

from nq.trading.indicators.technical_indicators import calculate_macd, calculate_rsi

logger = logging.getLogger(__name__)


class BottomDivergenceDetector:
    """
    Bottom divergence detector.

    Detects when price makes new lows but technical indicators (MACD/RSI) don't,
    indicating potential reversal signals.
    """

    def __init__(
        self,
        lookback_period: int = 30,
        min_divergence_bars: int = 5,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        rsi_period: int = 14,
    ):
        """
        Initialize bottom divergence detector.

        Args:
            lookback_period: Period to look back for finding previous lows (default: 30).
            min_divergence_bars: Minimum bars between two lows for valid divergence (default: 5).
            macd_fast: MACD fast period (default: 12).
            macd_slow: MACD slow period (default: 26).
            macd_signal: MACD signal period (default: 9).
            rsi_period: RSI period (default: 14).
        """
        self.lookback_period = lookback_period
        self.min_divergence_bars = min_divergence_bars
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.rsi_period = rsi_period

    def _find_price_lows(self, lows: np.ndarray, current_idx: int) -> Optional[int]:
        """
        Find the previous low point within lookback period.

        Args:
            lows: Array of low prices.
            current_idx: Current index.

        Returns:
            Index of previous low point, or None if not found.
        """
        if current_idx < self.min_divergence_bars:
            return None

        start_idx = max(0, current_idx - self.lookback_period)
        window = lows[start_idx:current_idx]

        if len(window) < self.min_divergence_bars:
            return None

        # Find the minimum low in the window
        min_idx = np.argmin(window)
        prev_low_idx = start_idx + min_idx

        # Ensure there are enough bars between the two lows
        if current_idx - prev_low_idx < self.min_divergence_bars:
            return None

        return prev_low_idx

    def _detect_macd_divergence(
        self,
        macd_line: np.ndarray,
        histogram: np.ndarray,
        current_idx: int,
        prev_low_idx: int,
    ) -> tuple[bool, str]:
        """
        Detect MACD bottom divergence.

        Args:
            macd_line: MACD line values.
            histogram: MACD histogram values.
            current_idx: Current index.
            prev_low_idx: Previous low index.

        Returns:
            Tuple of (is_divergence, divergence_type).
            divergence_type: 'macd_line' or 'histogram'.
        """
        if (
            current_idx >= len(macd_line)
            or prev_low_idx >= len(macd_line)
            or current_idx < 0
            or prev_low_idx < 0
        ):
            return False, ""

        current_macd = macd_line[current_idx]
        prev_macd = macd_line[prev_low_idx]

        current_hist = histogram[current_idx]
        prev_hist = histogram[prev_low_idx]

        # Check for None values
        if (
            current_macd is None
            or prev_macd is None
            or current_hist is None
            or prev_hist is None
        ):
            return False, ""

        # MACD line divergence: current MACD > previous MACD
        macd_divergence = current_macd > prev_macd

        # Histogram divergence: current histogram > previous histogram
        hist_divergence = current_hist > prev_hist

        if macd_divergence:
            return True, "macd_line"
        elif hist_divergence:
            return True, "histogram"

        return False, ""

    def _detect_rsi_divergence(
        self, rsi: np.ndarray, current_idx: int, prev_low_idx: int
    ) -> bool:
        """
        Detect RSI bottom divergence.

        Args:
            rsi: RSI values.
            current_idx: Current index.
            prev_low_idx: Previous low index.

        Returns:
            True if RSI divergence detected.
        """
        if (
            current_idx >= len(rsi)
            or prev_low_idx >= len(rsi)
            or current_idx < 0
            or prev_low_idx < 0
        ):
            return False

        current_rsi = rsi[current_idx]
        prev_rsi = rsi[prev_low_idx]

        if current_rsi is None or prev_rsi is None:
            return False

        # RSI divergence: current RSI > previous RSI
        return current_rsi > prev_rsi

    def detect(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Detect bottom divergence patterns in DataFrame.

        Args:
            df: DataFrame with columns: ts_code, trade_date, open, high, low, close, volume.
                Must be sorted by ts_code and trade_date.

        Returns:
            DataFrame with added columns:
            - is_bottom_divergence: Boolean indicating if bottom divergence detected
            - divergence_type: Type of divergence ('macd_line', 'histogram', 'rsi', 'both')
            - prev_low_idx: Index of previous low point
            - prev_low_date: Date of previous low point
        """
        if df.is_empty():
            return df.with_columns([
                pl.lit(False).alias("is_bottom_divergence"),
                pl.lit("").alias("divergence_type"),
                pl.lit(None).cast(pl.Int64).alias("prev_low_idx"),
                pl.lit(None).cast(pl.Utf8).alias("prev_low_date"),
            ])

        results = []

        for symbol in df["ts_code"].unique():
            symbol_df = df.filter(pl.col("ts_code") == symbol).sort("trade_date")

            if len(symbol_df) < self.lookback_period + self.macd_slow:
                # Not enough data, add empty columns
                symbol_df = symbol_df.with_columns([
                    pl.lit(False).alias("is_bottom_divergence"),
                    pl.lit("").alias("divergence_type"),
                    pl.lit(None).cast(pl.Int64).alias("prev_low_idx"),
                    pl.lit(None).cast(pl.Utf8).alias("prev_low_date"),
                ])
                results.append(symbol_df)
                continue

            # Convert to pandas for indicator calculation
            closes = symbol_df["close"].to_numpy()
            lows = symbol_df["low"].to_numpy()

            # Calculate MACD
            closes_pd = pd.Series(closes)
            macd_result = calculate_macd(
                closes_pd,
                fast_period=self.macd_fast,
                slow_period=self.macd_slow,
                signal_period=self.macd_signal,
            )
            macd_line = np.array(macd_result["macd"])
            histogram = np.array(macd_result["histogram"])

            # Calculate RSI
            rsi_result = calculate_rsi(closes_pd, period=self.rsi_period)
            rsi = np.array(rsi_result)

            # Detect divergence for each point
            is_divergence_list = []
            divergence_type_list = []
            prev_low_idx_list = []
            prev_low_date_list = []

            for i in range(len(symbol_df)):
                # Find previous low
                prev_low_idx = self._find_price_lows(lows, i)

                if prev_low_idx is None:
                    is_divergence_list.append(False)
                    divergence_type_list.append("")
                    prev_low_idx_list.append(None)
                    prev_low_date_list.append(None)
                    continue

                # Check if current low is lower than previous low
                current_low = lows[i]
                prev_low = lows[prev_low_idx]

                if current_low >= prev_low:
                    # Not a new low, no divergence
                    is_divergence_list.append(False)
                    divergence_type_list.append("")
                    prev_low_idx_list.append(None)
                    prev_low_date_list.append(None)
                    continue

                # Detect MACD divergence
                macd_div, macd_type = self._detect_macd_divergence(
                    macd_line, histogram, i, prev_low_idx
                )

                # Detect RSI divergence
                rsi_div = self._detect_rsi_divergence(rsi, i, prev_low_idx)

                # Combine results (use int() for Polars indexing)
                prev_low_idx_int = int(prev_low_idx)
                prev_low_date_val = str(symbol_df["trade_date"][prev_low_idx_int])

                if macd_div and rsi_div:
                    is_divergence_list.append(True)
                    divergence_type_list.append("both")
                    prev_low_idx_list.append(prev_low_idx_int)
                    prev_low_date_list.append(prev_low_date_val)
                elif macd_div:
                    is_divergence_list.append(True)
                    divergence_type_list.append(macd_type)
                    prev_low_idx_list.append(prev_low_idx_int)
                    prev_low_date_list.append(prev_low_date_val)
                elif rsi_div:
                    is_divergence_list.append(True)
                    divergence_type_list.append("rsi")
                    prev_low_idx_list.append(prev_low_idx_int)
                    prev_low_date_list.append(prev_low_date_val)
                else:
                    is_divergence_list.append(False)
                    divergence_type_list.append("")
                    prev_low_idx_list.append(None)
                    prev_low_date_list.append(None)

            symbol_df = symbol_df.with_columns([
                pl.Series("is_bottom_divergence", is_divergence_list),
                pl.Series("divergence_type", divergence_type_list),
                pl.Series("prev_low_idx", prev_low_idx_list),
                pl.Series("prev_low_date", prev_low_date_list),
            ])

            results.append(symbol_df)

        return pl.concat(results)
