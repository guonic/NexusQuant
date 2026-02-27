"""
Market regime state machine: trend vs relay (consolidation).

Two parts:
1. State labeling (上行/下行/中继): pluggable via regime labeler interface.
2. Trading signal labeling: pattern match (下中下上等) + breakout validation.

Labeler implementations:
- ma: Mid + MA2/MA3/MA5 + bundle_tightness (default).
- overlap: K-line body overlap ratio + color switch (no MAs).
"""

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Regime labeler interface: (df, **kwargs) -> df with "regime" column
# regime: 1=上行, -1=下行, 0=中继
# ---------------------------------------------------------------------------

RegimeLabeler = Callable[..., pd.DataFrame]

REGIME_LABELERS: Dict[str, RegimeLabeler] = {}


def analyze_market_regime(
    df: pd.DataFrame,
    *,
    high_col: str = "high",
    low_col: str = "low",
    threshold_tightness: float = 0.1,
    slope_filter: float = 0.02,
    mid_col: Optional[str] = None,
    slope_overrides_relay: Optional[float] = None,
) -> pd.DataFrame:
    """
    Classify each bar as long trend (1), short trend (-1), or relay/consolidation (0)
    based on Mid price and MA2, MA3, MA5.

    Parameters
    ----------
    df : pd.DataFrame
        Must have high and low columns (or mid_col if provided).
    high_col : str
        Column name for high.
    low_col : str
        Column name for low.
    threshold_tightness : float
        Max bundle_tightness to consider as relay (default 0.1).
    slope_filter : float
        Min |MA5_slope| to confirm trend (filters flat drift).
    mid_col : str, optional
        If given, use this as mid; otherwise mid = (high + low) / 2.
    slope_overrides_relay : float, optional
        When |MA5_slope| > this value, do not force 中继 even if bundle is tight,
        so 连续上升/下降 is labeled 上行/下降 (default: same as slope_filter).

    Returns
    -------
    pd.DataFrame
        Copy of df with added columns: mid, ma2, ma3, ma5, ma2_slope, ma5_slope,
        ma2_slope_pct (slope percentage), bundle_tightness, regime, breakout_signal,
        exit_long, exit_short.
    """
    df = df.copy()
    if slope_overrides_relay is None:
        slope_overrides_relay = slope_filter

    if mid_col and mid_col in df.columns:
        df["mid"] = df[mid_col]
    else:
        df["mid"] = (df[high_col] + df[low_col]) / 2.0

    df["ma2"] = df["mid"].rolling(2).mean()
    df["ma3"] = df["mid"].rolling(3).mean()
    df["ma5"] = df["mid"].rolling(5).mean()

    df["ma2_slope"] = df["ma2"].diff()
    df["ma5_slope"] = df["ma5"].diff()
    # 斜率百分比：Slope = (MA2_t - MA2_{t-1}) / MA2_{t-1} * 100%
    # 二阶导数思想：在均线位置改变之前，速度（斜率）会先改变
    df["ma2_slope_pct"] = (df["ma2"] - df["ma2"].shift(1)) / df["ma2"].shift(1) * 100.0
    df["ma2_slope_pct"] = df["ma2_slope_pct"].fillna(0.0)

    bundle_tightness = df[["ma2", "ma3", "ma5"]].std(axis=1)
    df["bundle_tightness"] = bundle_tightness.fillna(0.0)

    is_long_trend = (
        (df["ma2"] > df["ma3"])
        & (df["ma3"] > df["ma5"])
        & (df["ma5_slope"] > slope_filter)
    )
    is_short_trend = (
        (df["ma2"] < df["ma3"])
        & (df["ma3"] < df["ma5"])
        & (df["ma5_slope"] < -slope_filter)
    )

    # 中继：bundle 收紧且斜率弱时才判中继；斜率足够大时（连续上升/下降）优先趋势，不判中继
    is_relay = (
        (df["bundle_tightness"] < threshold_tightness)
        & (df["ma5_slope"].abs() <= slope_overrides_relay)
    )

    df["regime"] = 0
    df.loc[is_long_trend & ~is_relay, "regime"] = 1
    df.loc[is_short_trend & ~is_relay, "regime"] = -1

    prev_regime = df["regime"].shift(1)
    df["breakout_signal"] = (df["regime"] != 0) & (prev_regime == 0)

    df["exit_long"] = (prev_regime == 1) & (df["ma2"] < df["ma3"])
    df["exit_short"] = (prev_regime == -1) & (df["ma2"] > df["ma3"])

    return df


REGIME_LABELERS["ma"] = analyze_market_regime


def label_regime_overlap_box(
    df: pd.DataFrame,
    *,
    open_col: str = "open",
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    overlap_threshold: float = 0.6,
    containment_threshold: float = 0.7,
    expand_left: bool = True,
    max_expand_left_bars: int = 1,
    min_trend_bars: int = 3,
    min_trend_bars_force: Optional[int] = None,
    max_range_expand_ratio: float = 0.2,
    max_center_drift_ratio: float = 0.5,
    max_half_center_drift_ratio: float = 0.35,
) -> pd.DataFrame:
    """
    Label regime by K-line body overlap and color switch (no MAs).

    Trend: consecutive same-color bars (low overlap, displacement).
    Relay/Box: color switch + high body overlap ratio.
    Long same-color runs are forced to trend (so 明显的 下降/上升 are not a flat line).
    New bar that would significantly expand box range (breakout) also stops expansion.

    Overlap ratio (two bars):
      HighBody = max(open, close), LowBody = min(open, close)
      Overlap = (min(H_t, H_{t-1}) - max(L_t, L_{t-1})) / (max(H_t, H_{t-1}) - min(L_t, L_{t-1}))
    If Overlap > overlap_threshold: strong relay. Box expands right until new bar
    no longer overlaps box; optionally expand left to include last same-color bar.

    Parameters
    ----------
    df : pd.DataFrame
        Must have open, close, high, low.
    open_col, close_col, high_col, low_col : str
        Column names.
    overlap_threshold : float
        Min overlap ratio to enter/continue box (default 0.6). >0.7 strong relay.
        If a dip (e.g. 绿-红-绿) is not 中继, the red bar(s) may have low body overlap
        with the green bar; try lowering to 0.4–0.5 to allow more 中继.
    containment_threshold : float
        If the smaller body is mostly inside the larger (intersect/small_body_size >= this),
        also start 中继 so 中间两根小实体 can be relay (default 0.7).
    expand_left : bool
        If True, when box is confirmed, expand start left to include last same-color bar(s).
    max_expand_left_bars : int
        Max bars to expand left from the switch bar (default 1). Prevents pulling the whole
        same-direction run into 中继 so 同向左侧 remains 上升/下降.
    min_trend_bars : int
        During box expansion: stop when run of same color >= this (default 3).
    min_trend_bars_force : int, optional
        After boxes: only force a run to trend (exclude from box) when run length >= this.
        If None, use min_trend_bars. Set to 4 so that 3-bar pullbacks can stay 中继 (default None).
    max_range_expand_ratio : float
        When adding a bar would expand box range (high-low) by more than this ratio (e.g. 0.2 = 20%),
        stop expanding (default 0.2). Avoids swallowing breakout bars into relay.
    max_center_drift_ratio : float
        When adding a bar would shift the box center (mid of high-low) by more than this ratio
        relative to the initial box range, stop expanding (default 0.5). Keeps 两个中继+一个上升
        from merging into one box with 重心严重偏离.
    max_half_center_drift_ratio : float
        When 前半段重心 and 后半段重心 (mean mid of first/second half of current box) differ by
        more than this ratio of box range, stop expanding (default 0.35). 中继前后半段重心应接近.

    Returns
    -------
    pd.DataFrame
        Copy of df with columns: color, body_high, body_low, regime.
        Bars inside a detected box get regime=0; others get regime=color (1 or -1).
    """
    df = df.copy()
    open_ = df[open_col]
    close = df[close_col]
    high = df[high_col]
    low = df[low_col]

    df["color"] = np.where(close > open_, 1, -1)
    df["body_high"] = df[[open_col, close_col]].max(axis=1)
    df["body_low"] = df[[open_col, close_col]].min(axis=1)

    if min_trend_bars_force is None:
        min_trend_bars_force = min_trend_bars
    n = len(df)
    in_box = np.zeros(n, dtype=bool)

    i = 1
    while i < n:
        if df["color"].iloc[i] != df["color"].iloc[i - 1]:
            bh_i = df["body_high"].iloc[i]
            bl_i = df["body_low"].iloc[i]
            bh_prev = df["body_high"].iloc[i - 1]
            bl_prev = df["body_low"].iloc[i - 1]

            intersect = min(bh_i, bh_prev) - max(bl_i, bl_prev)
            union = max(bh_i, bh_prev) - min(bl_i, bl_prev)
            overlap_ratio = intersect / union if union > 0 else 0.0
            body_i = bh_i - bl_i
            body_prev = bh_prev - bl_prev
            small_body = min(body_i, body_prev)
            containment = intersect / small_body if small_body > 1e-9 else 0.0

            if overlap_ratio > overlap_threshold or containment >= containment_threshold:
                start_idx = i - 1
                if expand_left and start_idx > 0 and max_expand_left_bars > 0:
                    same_color = df["color"].iloc[start_idx]
                    left_limit = max(0, i - max_expand_left_bars)
                    while start_idx > left_limit and df["color"].iloc[start_idx - 1] == same_color:
                        start_idx -= 1
                box_high = float(high.iloc[start_idx : i + 1].max())
                box_low = float(low.iloc[start_idx : i + 1].min())
                initial_center = (box_high + box_low) / 2.0
                initial_range = box_high - box_low

                j = i + 1
                run_same_color = 0
                while j < n:
                    bhi = df["body_high"].iloc[j]
                    bli = df["body_low"].iloc[j]
                    new_intersect = min(bhi, box_high) - max(bli, box_low)
                    if new_intersect <= 0:
                        break
                    current_range = box_high - box_low
                    new_high = max(box_high, float(high.iloc[j]))
                    new_low = min(box_low, float(low.iloc[j]))
                    new_range = new_high - new_low
                    # 新 K 线明显拉大 box 的 range（突破）则不再纳入中继
                    if current_range > 1e-9 and (new_range - current_range) / current_range > max_range_expand_ratio:
                        break
                    # 中继重心偏离：box 中点相对初始范围漂移过大则停止，避免两中继+一上升合并成一个大 box
                    if initial_range > 1e-9:
                        new_center = (new_high + new_low) / 2.0
                        if abs(new_center - initial_center) / initial_range > max_center_drift_ratio:
                            break
                    # 前后半段重心一致：中继段前半段与后半段重心差过大则停止，避免「中继偏了」
                    segment_len = j - start_idx + 1
                    if segment_len >= 4 and new_range > 1e-9:
                        half_len = segment_len // 2
                        mid_vals = (high.iloc[start_idx : j + 1].values + low.iloc[start_idx : j + 1].values) / 2.0
                        center_first = float(np.mean(mid_vals[:half_len]))
                        center_second = float(np.mean(mid_vals[half_len:]))
                        if abs(center_first - center_second) / new_range > max_half_center_drift_ratio:
                            break
                    # 明显上升/下降：连续同色达到 min_trend_bars 即停止扩展，不把趋势段吞进中继
                    if j > start_idx and df["color"].iloc[j] == df["color"].iloc[j - 1]:
                        run_same_color += 1
                    else:
                        run_same_color = 1
                    if run_same_color >= min_trend_bars:
                        break
                    box_high = new_high
                    box_low = new_low
                    j += 1

                end_idx = j - 1
                in_box[start_idx : end_idx + 1] = True
                i = j
            else:
                i += 1
        else:
            i += 1

    # 长同色连续段强制为趋势（仅当 run >= min_trend_bars_force；短回调如 3 根可保留为中继）
    if min_trend_bars_force >= 1:
        color_arr = df["color"].values
        i = 0
        while i < n:
            c = color_arr[i]
            run_start = i
            while i < n and color_arr[i] == c:
                i += 1
            run_len = i - run_start
            if run_len >= min_trend_bars_force:
                in_box[run_start:i] = False

    df["regime"] = np.where(in_box, 0, df["color"])
    df["bundle_tightness"] = 0.0
    return df


REGIME_LABELERS["overlap"] = label_regime_overlap_box


def get_regime_segment_table(
    df: pd.DataFrame,
    *,
    regime_col: str = "regime",
    datetime_col: Optional[str] = "datetime",
    high_col: str = "high",
) -> pd.DataFrame:
    """
    Build a segment table for debugging: one row per regime segment (group),
    with group_id, regime, start/end time, and high_max. Use to verify which
    segment is used as 2.中继 (e.g. 15:43 vs 15:50).

    Parameters
    ----------
    df : pd.DataFrame
        Must have regime_col. If datetime_col is provided, used for start/end.
    regime_col : str
        Column name for regime (1=上, -1=下, 0=中继).
    datetime_col : str, optional
        Column name for bar time. If None, uses index for start/end.
    high_col : str
        Column name for high (used for high_max per segment).

    Returns
    -------
    pd.DataFrame
        Columns: group_id, regime, regime_name (下/中继/上), start_time, end_time,
        bar_count, high_max. Sorted by group_id (chronological order).
    """
    regime = df[regime_col]
    relay_groups = (regime != regime.shift(1)).cumsum()
    gr = df.groupby(relay_groups)
    seg = pd.DataFrame({"regime": gr[regime_col].first(), "bar_count": gr[regime_col].count()})
    if datetime_col and datetime_col in df.columns:
        seg["start_time"] = gr[datetime_col].min()
        seg["end_time"] = gr[datetime_col].max()
    else:
        seg["start_time"] = gr.apply(lambda x: x.index.min())
        seg["end_time"] = gr.apply(lambda x: x.index.max())
    seg["high_max"] = gr[high_col].max() if high_col in df.columns else np.nan
    seg = seg.rename_axis("group_id").reset_index()
    seg["regime_name"] = seg["regime"].astype(int).map({-1: "下", 0: "中继", 1: "上"})
    return seg


def add_validated_breakout_signals(
    df: pd.DataFrame,
    *,
    open_col: str = "open",
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    regime_col: str = "regime",
    break_offset: float = 0.05,
    use_relay_high: bool = False,
    require_ma2_slope: bool = True,
    ma2_slope_col: str = "ma2_slope",
    debug_2_relay: bool = False,
) -> pd.DataFrame:
    """
    Add long-only "validated breakout" signal: trigger when in an up phase
    and close breaks above 2.中继 max, when sequence matches one of the search tree patterns.

    Logic:
    - 命中判定为搜索树，按以下序列匹配（当前为上行 g 时，看 g-1, g-2, ...）：
      1. 下 中 下 上   (Pattern A)
      2. 下 中 下 上 中 上   (Pattern B)
      3. 下 中 下 中 上   (Pattern C)
    - Pattern A: g-1=-1, g-2=0, g-3=-1. last_relay_max = 2.中继 = g-2（下中下里的中）.
    - Pattern B: g-1=0, g-2=1, g-3=-1, g-4=0, g-5=-1. last_relay_max = 2.中继 = g-4.
    - Pattern C: g-1=中, g-2=下, g-3=中, g-4=下. last_relay_max = 2.中继 = g-3（下中下里的中，与 A/B 一致）.
    - Trigger: in uptrend (regime==1), signal when close > last_relay_max + break_offset.
      门限取下跌期间那段中继的高点即可。
    - 形态顺序价格判定：
      1. 下跌中继下跌：第二个下跌的 low 须低于第一个下跌的 low。
      2. 上涨有序：当前上涨段的 high 须高于第一个下跌段的 high（突破性上涨，非反弹）。
      否则不命中。
      Only the first bar that breaks counts.

    Parameters
    ----------
    df : pd.DataFrame
        Must have regime from analyze_market_regime and open/close/high/low.
    open_col, close_col, high_col, regime_col : str
        Column names. low_col defaults to "low" if not provided.
    break_offset : float
        Extra buffer above relay high for confirmation (e.g. 0.05–0.1 for XAUUSD).
    use_relay_high : bool
        If True, use max(high) of 2.中继 as threshold (K-line high including wicks);
        if False, use max(open,close) of 2.中继 (body only).
    require_ma2_slope : bool
        If True, long entry also requires ma2_slope > 0.
    ma2_slope_col : str
        Column name for MA2 slope (from analyze_market_regime).
    debug_2_relay : bool
        If True, add column pattern_2_relay_gid: the group_id of the segment
        used as 2.中继 for threshold. Join with get_regime_segment_table() to
        see that segment's start_time/end_time (e.g. verify 15:43 vs 15:50).

    Returns
    -------
    pd.DataFrame
        Copy of df with added columns:
        - body_max, body_min: max/min of open/close per bar
        - last_relay_max: max of 2.中继 when pattern A/B/C matches
        - entry_signal_long: long entry signal flag
        - relay_solid_box_ceiling/floor: solid box (body) ceiling/floor for relay groups (regime==0)
        - relay_logical_box_ceiling/floor: logical box (with wicks) ceiling/floor for relay groups
        - relay_solid_box_center: absolute center of solid box (价格围绕中心对称变动)
        - relay_logical_box_center: absolute center of logical box
    """
    df = df.copy()
    df["body_max"] = df[[open_col, close_col]].max(axis=1)
    df["body_min"] = df[[open_col, close_col]].min(axis=1)

    regime = df[regime_col]
    relay_groups = (regime != regime.shift(1)).cumsum()
    group_regime = df.groupby(relay_groups)[regime_col].first().astype(int).sort_index()
    group_max = df.groupby(relay_groups)["body_max"].max().sort_index()
    if use_relay_high and high_col in df.columns:
        group_relay_max = df.groupby(relay_groups)[high_col].max().sort_index()
    else:
        group_relay_max = group_max

    # 中继段 box 计算：solid box (body) 和 logical box (包含影线)
    group_body_min = df.groupby(relay_groups)["body_min"].min().sort_index()
    group_high_max = df.groupby(relay_groups)[high_col].max().sort_index() if high_col in df.columns else group_max
    group_low_min = df.groupby(relay_groups)[low_col].min().sort_index() if low_col in df.columns else group_body_min

    # solid box: ceiling = max(close, open), floor = min(close, open)
    group_solid_ceiling = group_max
    group_solid_floor = group_body_min
    # logical box: ceiling = max(high), floor = min(low)
    group_logical_ceiling = group_high_max
    group_logical_floor = group_low_min

    # 映射回 DataFrame（仅中继段有值）
    is_relay_group = group_regime == 0
    df["relay_solid_box_ceiling"] = relay_groups.map(group_solid_ceiling.where(is_relay_group))
    df["relay_solid_box_floor"] = relay_groups.map(group_solid_floor.where(is_relay_group))
    df["relay_logical_box_ceiling"] = relay_groups.map(group_logical_ceiling.where(is_relay_group))
    df["relay_logical_box_floor"] = relay_groups.map(group_logical_floor.where(is_relay_group))
    
    # 绝对中心：价格在box中围绕中心对称变动
    df["relay_solid_box_center"] = (df["relay_solid_box_ceiling"] + df["relay_solid_box_floor"]) / 2.0
    df["relay_logical_box_center"] = (df["relay_logical_box_ceiling"] + df["relay_logical_box_floor"]) / 2.0

    # 搜索树：三种序列 1.下中下上 2.下中下上中上 3.下中下中上
    prev1 = group_regime.shift(1).astype("Int64")
    prev2 = group_regime.shift(2).astype("Int64")
    prev3 = group_regime.shift(3).astype("Int64")
    prev4 = group_regime.shift(4).astype("Int64")
    prev5 = group_regime.shift(5).astype("Int64")

    # 1. 下 中 下 上 => g-1=-1, g-2=0, g-3=-1; 2.中继 = g-2（下跌期间那段中继，高于其即可）
    pattern_a = (
        (group_regime == 1)
        & (prev1 == -1)
        & (prev2 == 0)
        & (prev3 == -1)
    ).fillna(False)
    # 2. 下 中 下 上 中 上 => g-1=0, g-2=1, g-3=-1, g-4=0, g-5=-1; 2.中继 = g-4
    pattern_b = (
        (group_regime == 1)
        & (prev1 == 0)
        & (prev2 == 1)
        & (prev3 == -1)
        & (prev4 == 0)
        & (prev5 == -1)
    ).fillna(False)
    # 3. 下 中 下 中 上 => g-1=中(0), g-2=下(-1), g-3=中(0), g-4=下(-1)；2.中继 = 下中下里的中 = g-3（与 A/B 一致，永远对比下跌期间中间那段中继）
    pattern_c = (
        (group_regime == 1)
        & (prev1 == 0)
        & (prev2 == -1)
        & (prev3 == 0)
        & (prev4 == -1)
    ).fillna(False)

    # 形态顺序价格判定：
    # 1. 下跌中继下跌：第二个下跌的 low 须低于第一个下跌的 low
    # A: 第一下=g-3, 第二下=g-1 → low(g-1) < low(g-3)
    price_down_ok_a = (group_low_min.shift(1) < group_low_min.shift(3)).fillna(False)
    # B: 第一下=g-5, 第二下=g-3 → low(g-3) < low(g-5)
    price_down_ok_b = (group_low_min.shift(3) < group_low_min.shift(5)).fillna(False)
    # C: 第一下=g-4, 第二下=g-2 → low(g-2) < low(g-4)
    price_down_ok_c = (group_low_min.shift(2) < group_low_min.shift(4)).fillna(False)
    # 2. 上涨有序：当前上涨段的 high 须高于第一个下跌段的 high（突破性上涨，非反弹）
    # A: 当前上=g, 第一下=g-3 → high(g) > high(g-3)
    price_up_ok_a = (group_high_max > group_high_max.shift(3)).fillna(False)
    # B: 当前上=g, 第一下=g-5 → high(g) > high(g-5)
    price_up_ok_b = (group_high_max > group_high_max.shift(5)).fillna(False)
    # C: 当前上=g, 第一下=g-4 → high(g) > high(g-4)
    price_up_ok_c = (group_high_max > group_high_max.shift(4)).fillna(False)

    pattern_a_ok = pattern_a & price_down_ok_a & price_up_ok_a
    pattern_b_ok = pattern_b & price_down_ok_b & price_up_ok_b
    pattern_c_ok = pattern_c & price_down_ok_c & price_up_ok_c

    # 调试：记录形态匹配和价格判定结果（用于 tooltip 显示"为啥没中"）
    if debug_2_relay:
        pattern_match = pattern_a | pattern_b | pattern_c
        price_down_ok = price_down_ok_a | price_down_ok_b | price_down_ok_c
        price_up_ok = price_up_ok_a | price_up_ok_b | price_up_ok_c
        pattern_ok = pattern_a_ok | pattern_b_ok | pattern_c_ok
        df["_pattern_match"] = relay_groups.map(
            pd.Series(pattern_match.values, index=group_regime.index)
        )
        df["_price_down_ok"] = relay_groups.map(
            pd.Series(price_down_ok.values, index=group_regime.index)
        )
        df["_price_up_ok"] = relay_groups.map(
            pd.Series(price_up_ok.values, index=group_regime.index)
        )
        df["_pattern_ok"] = relay_groups.map(
            pd.Series(pattern_ok.values, index=group_regime.index)
        )

    # 门限 = 下跌期间中间那段中继（2.中继）的高点；仅当形态+价格判定都通过时才给 last_relay_max
    relay_high_to_use = pd.Series(
        np.where(
            pattern_a_ok.values,
            group_relay_max.shift(2).values,
            np.where(
                pattern_b_ok.values,
                group_relay_max.shift(4).values,
                np.where(pattern_c_ok.values, group_relay_max.shift(3).values, np.nan),
            ),
        ),
        index=group_regime.index,
    )
    df["last_relay_max"] = relay_groups.map(relay_high_to_use)

    if debug_2_relay:
        # 用于查 2.中继 对应哪一段：pattern_2_relay_gid = 门限所用的中继段 group_id（仅形态+价格判定通过时）
        pattern_2_gid = np.where(
            pattern_a_ok.values,
            group_regime.index.values - 2,
            np.where(
                pattern_b_ok.values,
                group_regime.index.values - 4,
                np.where(pattern_c_ok.values, group_regime.index.values - 3, np.nan),
            ),
        )
        df["pattern_2_relay_gid"] = relay_groups.map(
            pd.Series(pattern_2_gid, index=group_regime.index)
        )

    close = df[close_col]
    threshold = df["last_relay_max"] + break_offset
    is_up = regime == 1
    break_long = close > threshold
    first_break_long = break_long & (~break_long.shift(1).fillna(False))

    df["entry_signal_long"] = (
        is_up
        & first_break_long
        & df["last_relay_max"].notna()
    )

    if require_ma2_slope and ma2_slope_col in df.columns:
        slope = df[ma2_slope_col]
        df["entry_signal_long"] = df["entry_signal_long"] & (slope > 0)

    return df


def adjust_relay_boundaries(
    df: pd.DataFrame,
    *,
    regime_col: str = "regime",
    ma2_slope_pct_col: str = "ma2_slope_pct",
    solid_box_center_col: str = "relay_solid_box_center",
    logical_box_center_col: str = "relay_logical_box_center",
    price_col: str = "close",
    slope_threshold: float = 0.1,
    lookback_bars: int = 5,
) -> pd.DataFrame:
    """
    修正中继段的起始和结束时间点。

    修正方法：
    1. 斜率（Slope）：在均线位置改变之前，速度（斜率）会先改变。
       使用 MA2 斜率百分比变化来更精确地确定中继段的边界。
    2. 绝对中心：价格在box中围绕中心对称变动。
       验证价格是否围绕 solid/logical box 中心对称变动。

    Parameters
    ----------
    df : pd.DataFrame
        必须包含 regime, ma2_slope_pct, relay_solid_box_center, relay_logical_box_center。
    regime_col : str
        regime 列名。
    ma2_slope_pct_col : str
        MA2 斜率百分比列名。
    solid_box_center_col, logical_box_center_col : str
        solid/logical box 中心列名。
    price_col : str
        价格列名（用于验证中心对称）。
    slope_threshold : float
        斜率变化阈值（百分比），用于判断斜率是否显著变化。
    lookback_bars : int
        向前/向后查找的K线数量。

    Returns
    -------
    pd.DataFrame
        添加了 relay_start_adjusted, relay_end_adjusted 列的 DataFrame。
        这些列标记修正后的中继段起始/结束点（True表示该点是修正后的边界）。
    """
    df = df.copy()
    
    if regime_col not in df.columns:
        return df
    
    regime = df[regime_col]
    relay_groups = (regime != regime.shift(1)).cumsum()
    
    # 初始化修正标记
    df["relay_start_adjusted"] = False
    df["relay_end_adjusted"] = False
    
    # 获取所有中继段
    group_regime = df.groupby(relay_groups)[regime_col].first().astype(int).sort_index()
    relay_group_ids = group_regime[group_regime == 0].index.tolist()
    
    if ma2_slope_pct_col not in df.columns:
        return df
    
    slope_pct = df[ma2_slope_pct_col].fillna(0.0)
    
    for group_id in relay_group_ids:
        relay_mask = relay_groups == group_id
        relay_indices = df.index[relay_mask].tolist()
        
        if len(relay_indices) == 0:
            continue
        
        start_idx = relay_indices[0]
        end_idx = relay_indices[-1]
        start_pos = df.index.get_loc(start_idx)
        end_pos = df.index.get_loc(end_idx)
        
        # 修正起始点：向前查找斜率开始平缓的点
        # 在均线位置改变之前，速度（斜率）会先改变
        adjusted_start_pos = start_pos
        if start_pos > 0:
            # 向前查找，找到斜率绝对值开始变小的点（从趋势转为平缓）
            for i in range(max(0, start_pos - lookback_bars), start_pos):
                if i >= len(df):
                    break
                current_slope_abs = abs(slope_pct.iloc[i])
                prev_slope_abs = abs(slope_pct.iloc[i - 1]) if i > 0 else current_slope_abs
                # 如果斜率从大变小，说明开始进入中继
                if prev_slope_abs > slope_threshold and current_slope_abs <= slope_threshold:
                    adjusted_start_pos = i
                    break
        
        # 修正结束点：向后查找斜率开始变化的点
        adjusted_end_pos = end_pos
        if end_pos < len(df) - 1:
            # 向后查找，找到斜率绝对值开始变大的点（从平缓转为趋势）
            for i in range(end_pos + 1, min(len(df), end_pos + 1 + lookback_bars)):
                if i >= len(df):
                    break
                current_slope_abs = abs(slope_pct.iloc[i])
                prev_slope_abs = abs(slope_pct.iloc[i - 1]) if i > 0 else current_slope_abs
                # 如果斜率从小变大，说明开始离开中继
                if prev_slope_abs <= slope_threshold and current_slope_abs > slope_threshold:
                    adjusted_end_pos = i - 1
                    break
        
        # 标记修正后的边界点
        if adjusted_start_pos < len(df):
            df.loc[df.index[adjusted_start_pos], "relay_start_adjusted"] = True
        if adjusted_end_pos < len(df):
            df.loc[df.index[adjusted_end_pos], "relay_end_adjusted"] = True
    
    return df
