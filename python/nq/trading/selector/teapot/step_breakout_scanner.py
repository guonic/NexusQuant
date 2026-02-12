# -*- coding: utf-8 -*-
"""
Step Breakout Scanner for Teapot pattern recognition.

Identifies "ladder-style breakout" pattern: decline -> base box -> washout ->
first breakout (stage 4 / buy1) -> relay box -> second breakout (stage 6 / buy2).
Multi-stage relay structure; stages 4 and 6 are key entry points.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class StageInterval:
    """Time interval of a stage for logging."""

    stage: int
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    high: Optional[float] = None
    low: Optional[float] = None
    extra: str = ""


@dataclass
class StepBreakoutResult:
    """Result of step breakout analysis for current slice."""

    ts_code: str
    trade_date: str
    current_stage: Optional[int]  # 4 or 6 or None
    in_stage_4: bool
    in_stage_6: bool
    intervals: List[StageInterval] = field(default_factory=list)
    power_ratio: Optional[float] = None
    volume_ratio_6_to_4: Optional[float] = None


def _intervals_from_row(row: dict) -> List[StageInterval]:
    """Build stage intervals from a result row for logging."""
    intervals: List[StageInterval] = []
    if row.get("stage2_start_date") and row.get("stage2_end_date"):
        intervals.append(
            StageInterval(
                stage=2,
                start_date=str(row["stage2_start_date"]),
                end_date=str(row["stage2_end_date"]),
                high=float(row["stage2_high"]) if row.get("stage2_high") is not None else None,
                low=float(row["stage2_low"]) if row.get("stage2_low") is not None else None,
            )
        )
    if row.get("stage3_washout_start") and row.get("stage3_washout_end"):
        intervals.append(
            StageInterval(
                stage=3,
                start_date=str(row["stage3_washout_start"]),
                end_date=str(row["stage3_washout_end"]),
                extra="washout window",
            )
        )
    if row.get("stage5_start_date") and row.get("stage5_end_date"):
        intervals.append(
            StageInterval(
                stage=5,
                start_date=str(row["stage5_start_date"]),
                end_date=str(row["stage5_end_date"]),
                high=float(row["stage5_high"]) if row.get("stage5_high") is not None else None,
                low=float(row["stage5_low"]) if row.get("stage5_low") is not None else None,
            )
        )
    return intervals


class StepBreakoutScanner:
    """
    Step breakout scanner (阶梯式突破扫描).

    Stages:
    1. Prior decline (bearish context).
    2. Base box: horizontal range; High_Close = global resistance.
    3. Washout: price breaks below base box (fake breakdown).
    4. First breakout / Buy1: close above stage-2 high.
    5. Relay box: narrow consolidation above the step.
    6. Second breakout / Buy2: break above stage-5 high.
    """

    def __init__(
        self,
        box_len: int = 20,
        box_lookback_start: int = 40,
        box_lookback_end: int = 21,
        box_range: float = 0.10,
        washout_window: int = 6,
        washout_lookback_start: int = 10,
        washout_lookback_end: int = 5,
        relay_box_len: int = 10,
        relay_box_range: float = 0.05,
        power_threshold: float = 1.2,
    ):
        """
        Initialize scanner.

        Args:
            box_len: Length of stage-2 base box window (default: 20).
            box_lookback_start: Bars ago for start of stage-2 window (default: 40).
            box_lookback_end: Bars ago for end of stage-2 window (default: 21).
            box_range: Max (high-low)/low for stage-2 box (default: 0.10).
            washout_window: Length of window to detect break below box (default: 6).
            washout_lookback_start: Start of washout window bars ago (default: 10).
            washout_lookback_end: End of washout window bars ago (default: 5).
            relay_box_len: Length of stage-5 relay box (default: 10).
            relay_box_range: Max width for relay box (default: 0.05).
            power_threshold: Min power ratio for breakout strength (optional).
        """
        self.box_len = box_len
        self.box_lookback_start = box_lookback_start
        self.box_lookback_end = box_lookback_end
        self.box_range = box_range
        self.washout_window = washout_window
        self.washout_lookback_start = washout_lookback_start
        self.washout_lookback_end = washout_lookback_end
        self.relay_box_len = relay_box_len
        self.relay_box_range = relay_box_range
        self.power_threshold = power_threshold

    def _over_expr(self) -> List[str]:
        """Return grouping for per-stock rolling (empty if single stock)."""
        return []

    def analyze(
        self,
        df: pl.DataFrame,
        ts_code_col: str = "ts_code",
    ) -> pl.DataFrame:
        """
        Run step breakout logic on K-line DataFrame.

        Expects columns: trade_date, open, high, low, close, volume.
        If ts_code_col exists, rolling is done per stock.

        Returns:
            DataFrame with added columns: ma2, ma3, ma5, is_aligned (MA2>MA3>MA5),
            slope, stage1_decline_slope, power_ratio (slope/|decline_slope|),
            stage2_high/low, is_stage2_box, was_break_down, signal_4, stage5_high/low,
            is_stage5_box, signal_6, current_stage, and stage interval dates.
        """
        if len(df) < self.box_lookback_start + 10:
            return df

        over = [ts_code_col] if ts_code_col in df.columns else []

        def roll(expr: pl.Expr) -> pl.Expr:
            return expr.over(over) if over else expr

        # Short-term MA system: ma2, ma3, ma5 for alignment filter (阶梯感 / 短线起爆点)
        ma5 = pl.col("close").rolling_mean(5)
        df = df.with_columns([
            roll(pl.col("close").rolling_mean(2)).alias("ma2"),
            roll(pl.col("close").rolling_mean(3)).alias("ma3"),
            roll(ma5).alias("ma5"),
        ])
        # is_aligned: MA2 > MA3 > MA5 (多头排列), filters fake breakouts and downward drift
        df = df.with_columns(
            (
                (pl.col("ma2") > pl.col("ma3"))
                & (pl.col("ma3") > pl.col("ma5"))
            ).alias("is_aligned")
        )
        df = df.with_columns(roll(pl.col("ma5").diff(2)).alias("slope"))

        # Stage 1 decline slope (before base box): bars [box_lookback_start+20, box_lookback_start]
        stage1_len = 20
        decline_start = pl.col("close").shift(self.box_lookback_start + stage1_len)
        decline_end = pl.col("close").shift(self.box_lookback_start)
        decline_slope = (decline_end - decline_start) / stage1_len
        df = df.with_columns(roll(decline_slope).alias("stage1_decline_slope"))
        # Power ratio: current slope (ma5 diff) / |stage1_decline_slope|; > 1.5 = rebound stronger than prior decline
        df = df.with_columns(
            pl.when(pl.col("stage1_decline_slope").abs() > 1e-10)
            .then(pl.col("slope") / pl.col("stage1_decline_slope").abs())
            .otherwise(None)
            .alias("power_ratio")
        )

        # Stage 2: base box = [box_lookback_start .. box_lookback_end] bars ago
        # shift(box_lookback_end) then rolling_max(box_len) => max of [end - box_len .. end] = [start .. end]
        shift_s2 = self.box_lookback_end
        stage2_high = (
            pl.col("high").shift(shift_s2).rolling_max(self.box_len)
        )
        stage2_low = (
            pl.col("low").shift(shift_s2).rolling_min(self.box_len)
        )
        df = df.with_columns([
            roll(stage2_high).alias("stage2_high"),
            roll(stage2_low).alias("stage2_low"),
        ])
        df = df.with_columns(
            (
                (pl.col("stage2_high") - pl.col("stage2_low"))
                / (pl.col("stage2_low") + 1e-10)
            ).alias("stage2_width")
        )
        df = df.with_columns(
            (pl.col("stage2_width") < self.box_range).alias("is_stage2_box")
        )

        # Stage 2 interval dates for logging
        df = df.with_columns([
            roll(pl.col("trade_date").shift(self.box_lookback_start)).alias("stage2_start_date"),
            roll(pl.col("trade_date").shift(self.box_lookback_end)).alias("stage2_end_date"),
        ])

        # Stage 3: washout — min(close) in [washout_lookback_start .. washout_lookback_end] < stage2_low
        # shift(washout_lookback_end).rolling_min(washout_window) => min of close in that window
        washout_min = (
            pl.col("close")
            .shift(self.washout_lookback_end)
            .rolling_min(self.washout_window)
        )
        df = df.with_columns(roll(washout_min).alias("washout_min_close"))
        df = df.with_columns(
            (pl.col("washout_min_close") < pl.col("stage2_low")).alias("was_break_down")
        )
        df = df.with_columns([
            roll(pl.col("trade_date").shift(self.washout_lookback_start)).alias("stage3_washout_start"),
            roll(pl.col("trade_date").shift(self.washout_lookback_end)).alias("stage3_washout_end"),
        ])

        # Stage 4: first breakout — box + washout + close > stage2_high + is_aligned (均线多头)
        prev_close = pl.col("close").shift(1)
        df = df.with_columns(
            (
                pl.col("is_stage2_box")
                & pl.col("was_break_down")
                & (pl.col("close") > pl.col("stage2_high"))
                & (roll(prev_close) <= pl.col("stage2_high"))
                & pl.col("is_aligned")
            ).alias("signal_4")
        )

        # Stage 5: relay box — last relay_box_len bars, narrow range
        stage5_high = pl.col("close").shift(1).rolling_max(self.relay_box_len)
        stage5_low = pl.col("close").shift(1).rolling_min(self.relay_box_len)
        df = df.with_columns([
            roll(stage5_high).alias("stage5_high"),
            roll(stage5_low).alias("stage5_low"),
        ])
        df = df.with_columns(
            (
                (pl.col("stage5_high") - pl.col("stage5_low"))
                / (pl.col("stage5_low") + 1e-10)
            ).alias("stage5_width")
        )
        df = df.with_columns(
            (pl.col("stage5_width") < self.relay_box_range).alias("is_stage5_box")
        )
        df = df.with_columns([
            roll(pl.col("trade_date").shift(self.relay_box_len)).alias("stage5_start_date"),
            roll(pl.col("trade_date").shift(1)).alias("stage5_end_date"),
        ])

        # Stage 6: second breakout — above stage2 + relay box + close > stage5_high + is_aligned
        prev_close_1 = pl.col("close").shift(1)
        df = df.with_columns(
            (
                (pl.col("close") > pl.col("stage2_high"))
                & pl.col("is_stage5_box")
                & (pl.col("close") > pl.col("stage5_high"))
                & (roll(prev_close_1) <= pl.col("stage5_high"))
                & pl.col("is_aligned")
            ).alias("signal_6")
        )

        # Current stage: 4 or 6 or null
        df = df.with_columns(
            pl.when(pl.col("signal_6"))
            .then(pl.lit(6))
            .when(pl.col("signal_4"))
            .then(pl.lit(4))
            .otherwise(pl.lit(None))
            .cast(pl.Int32)
            .alias("current_stage")
        )

        return df

    def get_current_slice_results(
        self,
        df: pl.DataFrame,
        ts_code_col: str = "ts_code",
    ) -> List[StepBreakoutResult]:
        """
        Determine whether the current slice (last row per stock) is in stage 4 or 6,
        and collect stage intervals for logging.

        Args:
            df: DataFrame with ts_code, trade_date, close, high, low, volume.

        Returns:
            List of StepBreakoutResult, one per stock (or one element if single stock).
        """
        if df.is_empty():
            return []

        analyzed = self.analyze(df, ts_code_col=ts_code_col)
        has_ts = ts_code_col in analyzed.columns

        if has_ts:
            last = analyzed.group_by(ts_code_col).tail(1)
        else:
            last = analyzed.tail(1)

        if last.is_empty():
            return []

        results: List[StepBreakoutResult] = []
        for row in last.to_dicts():
            ts_code = str(row.get(ts_code_col, ""))
            trade_date = str(row.get("trade_date", ""))
            current_stage = row.get("current_stage")
            in_4 = bool(row.get("signal_4", False))
            in_6 = bool(row.get("signal_6", False))
            intervals = _intervals_from_row(row)

            power_ratio = None
            if row.get("power_ratio") is not None:
                try:
                    power_ratio = float(row["power_ratio"])
                except (TypeError, ValueError):
                    pass

            # Volume ratio: stage6 volume / stage4 volume (量能台阶)
            vol_ratio_6_to_4 = None
            if has_ts and in_6 and row.get("volume") is not None and row.get("volume") > 0:
                stock_df = analyzed.filter(pl.col(ts_code_col) == ts_code)
                signal_4_rows = (
                    stock_df.filter(pl.col("signal_4"))
                    .sort("trade_date", descending=True)
                )
                if not signal_4_rows.is_empty() and "volume" in signal_4_rows.columns:
                    vol_4 = signal_4_rows["volume"][0]
                    if vol_4 is not None and float(vol_4) > 0:
                        vol_ratio_6_to_4 = float(row["volume"]) / float(vol_4)

            results.append(
                StepBreakoutResult(
                    ts_code=ts_code,
                    trade_date=trade_date,
                    current_stage=current_stage,
                    in_stage_4=in_4,
                    in_stage_6=in_6,
                    intervals=intervals,
                    power_ratio=power_ratio,
                    volume_ratio_6_to_4=vol_ratio_6_to_4,
                )
            )
        return results

    def log_stage_intervals(self, result: StepBreakoutResult) -> None:
        """Log detailed stage time intervals for the current slice."""
        logger.info(
            "StepBreakout current slice: ts_code=%s trade_date=%s current_stage=%s in_stage_4=%s in_stage_6=%s",
            result.ts_code,
            result.trade_date,
            result.current_stage,
            result.in_stage_4,
            result.in_stage_6,
        )
        for iv in result.intervals:
            msg = (
                f"  Stage {iv.stage}: {iv.start_date} ~ {iv.end_date}"
            )
            if iv.high is not None and iv.low is not None:
                msg += f" high={iv.high:.4f} low={iv.low:.4f}"
            if iv.extra:
                msg += f" ({iv.extra})"
            logger.info(msg)
        if result.power_ratio is not None:
            logger.info("  power_ratio=%.4f (threshold=%.2f)", result.power_ratio, self.power_threshold)
        if result.volume_ratio_6_to_4 is not None:
            logger.info("  volume_ratio_6_to_4=%.4f", result.volume_ratio_6_to_4)
