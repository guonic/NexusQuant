# -*- coding: utf-8 -*-
"""
Tests for Teapot box detectors.

Ensures all detectors implement detect_box correctly and return required columns.
"""

import pytest

import polars as pl

from nq.trading.selector.teapot.box_detector import (
    BoxDetector,
    SimpleBoxDetector,
    MeanReversionBoxDetectorV2,
)
from nq.trading.selector.teapot.box_detector_dense_area import DenseAreaBoxDetector
from nq.trading.selector.teapot.box_detector_ribbon_coherence import (
    RibbonCoherenceDetector,
)
from nq.trading.selector.teapot.box_detector_composite_equilibrium import (
    CompositeEquilibriumDetector,
)
from nq.trading.selector.teapot.box_detector_anti_step import AntiStepBoxDetector


# Minimum rows for MA20 and 60-day volume
MIN_ROWS = 80

REQUIRED_COLUMNS = ["box_h", "box_l", "box_width", "is_box_candidate"]


@pytest.fixture
def kline_df():
    """Minimal K-line DataFrame for detector tests (single stock, enough rows for MA20/60)."""
    n = MIN_ROWS
    return pl.DataFrame({
        "ts_code": ["000001.SZ"] * n,
        "trade_date": [str(20230101 + i) for i in range(n)],
        "open": [10.0 + (i % 5) * 0.05 for i in range(n)],
        "high": [10.2 + (i % 5) * 0.05 for i in range(n)],
        "low": [9.8 + (i % 5) * 0.05 for i in range(n)],
        "close": [10.0 + (i % 5) * 0.05 for i in range(n)],
        "volume": [100_0000] * 30 + [80_0000] * (n - 30),
    })


@pytest.fixture
def kline_df_flat():
    """Nearly flat close (good for equilibrium-style detectors)."""
    n = MIN_ROWS
    return pl.DataFrame({
        "ts_code": ["000001.SZ"] * n,
        "trade_date": [str(20230101 + i) for i in range(n)],
        "open": [10.0] * n,
        "high": [10.05] * n,
        "low": [9.95] * n,
        "close": [10.0 + (i % 3) * 0.01 for i in range(n)],
        "volume": [80_0000] * n,
    })


def _assert_detector_output(df_out: pl.DataFrame) -> None:
    for col in REQUIRED_COLUMNS:
        assert col in df_out.columns, f"Missing column: {col}"
    assert len(df_out) > 0
    assert df_out["is_box_candidate"].dtype == pl.Boolean


def test_simple_box_detector(kline_df: pl.DataFrame) -> None:
    det = SimpleBoxDetector(box_window=20, box_width_threshold=0.15)
    out = det.detect_box(kline_df)
    _assert_detector_output(out)


def test_mean_reversion_v2_detector(kline_df: pl.DataFrame) -> None:
    det = MeanReversionBoxDetectorV2(
        box_window=20,
        max_relative_box_height=0.12,
    )
    out = det.detect_box(kline_df)
    _assert_detector_output(out)


def test_dense_area_detector(kline_df: pl.DataFrame) -> None:
    det = DenseAreaBoxDetector(
        box_window=30,
        dense_threshold=0.06,
        vol_stability_threshold=0.05,
        mid_slope_threshold=0.03,
    )
    out = det.detect_box(kline_df)
    _assert_detector_output(out)
    assert "dense_width" in out.columns
    assert "vol_stability" in out.columns


def test_ribbon_coherence_detector(kline_df: pl.DataFrame) -> None:
    det = RibbonCoherenceDetector(
        min_steady_days=15,
        convergence_threshold=0.02,
        ma_slope_threshold=0.015,
        price_to_ma_threshold=0.03,
    )
    out = det.detect_box(kline_df)
    _assert_detector_output(out)
    assert "ma_cv" in out.columns
    assert "ma_slope" in out.columns
    assert "price_dev" in out.columns


def test_composite_equilibrium_detector(kline_df: pl.DataFrame) -> None:
    det = CompositeEquilibriumDetector(
        box_window=20,
        ma_cohesion_threshold=0.02,
        quantile_width_threshold=0.06,
        cross_count_min=2,
        cross_window=15,
        volume_short=15,
        volume_long=60,
        volume_ratio=0.85,
    )
    out = det.detect_box(kline_df)
    _assert_detector_output(out)
    assert "ma_cohesion" in out.columns
    assert "quantile_width" in out.columns
    assert "ma_cross_count" in out.columns


def test_composite_equilibrium_detector_without_volume(kline_df: pl.DataFrame) -> None:
    """CompositeEquilibriumDetector should run when volume column is missing (D = true)."""
    df_no_vol = kline_df.drop("volume")
    det = CompositeEquilibriumDetector(
        box_window=20,
        ma_cohesion_threshold=0.03,
        quantile_width_threshold=0.08,
        cross_count_min=2,
    )
    out = det.detect_box(df_no_vol)
    _assert_detector_output(out)


def test_composite_equilibrium_detector_flat(kline_df_flat: pl.DataFrame) -> None:
    """On nearly flat price, composite detector may yield some candidates with relaxed params."""
    det = CompositeEquilibriumDetector(
        box_window=20,
        ma_cohesion_threshold=0.02,
        quantile_width_threshold=0.05,
        cross_count_min=2,
        volume_ratio=0.9,
    )
    out = det.detect_box(kline_df_flat)
    _assert_detector_output(out)


def test_anti_step_detector(kline_df: pl.DataFrame) -> None:
    det = AntiStepBoxDetector(
        box_window=20,
        r_threshold=0.5,
        center_dev_threshold=0.7,
        box_width_threshold=0.2,
    )
    out = det.detect_box(kline_df)
    _assert_detector_output(out)
    assert "r_value" in out.columns
    assert "center_dev" in out.columns


def test_anti_step_detector_rejects_step(kline_df: pl.DataFrame) -> None:
    """Strong downward trend in last half should reduce/eliminate box candidates."""
    n = len(kline_df)
    # Left half flat ~10, right half drop to ~9
    step_close = [10.0] * (n // 2) + [10.0 - (i - n // 2) * 0.05 for i in range(n // 2, n)]
    df_step = kline_df.with_columns(pl.Series("close", step_close))
    det = AntiStepBoxDetector(
        box_window=25,
        r_threshold=0.4,
        center_dev_threshold=0.6,
        box_width_threshold=0.15,
    )
    out = det.detect_box(df_step)
    _assert_detector_output(out)
    # In the cliff zone, r_value should be strongly negative; many rows should be excluded
    late = out.filter(pl.col("trade_date") >= out["trade_date"][n // 2])
    if len(late) > 0 and late["r_value"].null_count() < len(late):
        assert late["r_value"].min() < 0
