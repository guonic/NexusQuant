# -*- coding: utf-8 -*-
"""
Topological Trend Scanner (状态拓扑选股器).

State evolution model: market as "Trend Strings" (线段) and "Mediation Nodes" (中继).
- Atomic states: 上升中 (up), 下跌中 (down), 纠缠中 (relay).
- Segment aggregation: consecutive same-state days form segments.
- Sequence template: 下跌 → 中继 → 反转 → 上升 (signal_4); then 上升 → 中继 → 二次上升 (signal_6).

Uses rolling bitmask / window logic in Polars for sequence matching without explicit loops.
"""

import logging

import polars as pl

logger = logging.getLogger(__name__)


def _relay_std_expr() -> pl.Expr:
    """Row-wise std of ma2, ma3, ma5 for relay (entanglement) detection."""
    mean_ma = (pl.col("ma2") + pl.col("ma3") + pl.col("ma5")) / 3.0
    var_ma = (
        (pl.col("ma2") - mean_ma).pow(2)
        + (pl.col("ma3") - mean_ma).pow(2)
        + (pl.col("ma5") - mean_ma).pow(2)
    ) / 3.0
    return var_ma.sqrt()


def _rolling_any(expr: pl.Expr, window: int) -> pl.Expr:
    """True if any value in the rolling window is truthy (Polars has no rolling_any)."""
    return expr.cast(pl.Int32).rolling_sum(window) >= 1


def _is_single_day_seg(r: dict, date_col_begin: str = "开始时间", date_col_end: str = "结束时间") -> bool:
    """Segment is single day iff 开始时间 == 结束时间."""
    b, e = r.get(date_col_begin), r.get(date_col_end)
    return b is not None and e is not None and str(b) == str(e)


def _merge_single_day_relay_segments(
    rows: list[dict],
    date_col_begin: str = "开始时间",
    date_col_end: str = "结束时间",
) -> list[dict]:
    """
    Merge single-day relay: DRD->D, URU->U, DRU->DU.
    Used so that signal is judged on merged segments (合并后的去判定信号).
    """
    if len(rows) < 3:
        return rows
    changed = False
    out: list[dict] = []
    i = 0
    while i < len(rows):
        r = rows[i]
        lab = r.get("状态")
        if lab != "中继" or not _is_single_day_seg(r, date_col_begin, date_col_end):
            out.append(r)
            i += 1
            continue
        prev = out[-1] if out else None
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        if prev is None or nxt is None:
            out.append(r)
            i += 1
            continue
        plab, nlab = prev.get("状态"), nxt.get("状态")
        if plab == "下跌" and nlab == "下跌":
            out[-1] = {"状态": "下跌", date_col_begin: prev[date_col_begin], date_col_end: nxt[date_col_end]}
            i += 2
            changed = True
            continue
        if plab == "上涨" and nlab == "上涨":
            out[-1] = {"状态": "上涨", date_col_begin: prev[date_col_begin], date_col_end: nxt[date_col_end]}
            i += 2
            changed = True
            continue
        if (plab == "下跌" and nlab == "上涨") or (plab == "上涨" and nlab == "下跌"):
            i += 1
            changed = True
            continue
        out.append(r)
        i += 1
    if changed:
        return _merge_single_day_relay_segments(out, date_col_begin, date_col_end)
    return out


class TopologicalTrendScanner:
    """
    Topological trend scanner (状态拓扑选股器) - State Chain Pattern Matching with Noise Filtering.

    Uses Run Length Encoding (RLE) to compress consecutive states into segments,
    then matches patterns on the state chain (s0=current, s1=prev, s2=prev-prev...).

    Key improvements:
    1. Inflection-first state switching: state changes start from MA2 inflection point (拐点先行), not when MA alignment completes.
    2. Dynamic recursive relay: relay extends recursively from entanglement start, new nodes included if they don't significantly shift the center mean (中继动态递归).
    3. Large-level relay: D+U segments with overlapping price ranges merge into large-level relay (大级别中继：D+U 区间重合).
    4. Cross-period nesting: Daily relay should be validated by 60min DRU combinations within the same price space (跨周期嵌套 - requires 60min data).
    5. Box ceiling: physical price ceiling from recent non-up states (filters small oscillations).

    Steps:
    1. Atomic state: raw_state = 1(Up), -1(Down), 0(Relay) with noise filtering.
    2. State compression: consecutive same states → state_id (RLE).
    3. State chain: s0..s5 = state values of current and previous 5 segments.
    4. Box ceiling: rolling_max(high) over recent N days when raw_state <= 0 (D or R).
    5. Pattern matching with physical breakout:
       - signal_4: [D, R, D, R, U] pattern + close > box_ceiling.
       - signal_6: [U, R, U] pattern + close > s4_barrier (box_ceiling at signal_4).
    """

    def __init__(
        self,
        relay_std_ratio: float = 0.007,
        box_ceiling_window: int = 15,
        ma_diff_threshold: float = 0.003,
        use_merged_segments_for_signal: bool = True,
    ):
        """
        Initialize scanner.

        Args:
            relay_std_ratio: Max std(ma2,ma3,ma5)/ma5 for relay state (default: 0.007).
            box_ceiling_window: Window size for box_ceiling (recent N days of non-up states, default: 15).
            ma_diff_threshold: Threshold for ma2-ma3 and ma3-ma5 diff relative to ma5 for relay detection (default: 0.003).
            use_merged_segments_for_signal: If True, signal_4/signal_6 are judged on merged segments (DRD->D, URU->U, DRU->DU), so pattern is D-U-D-U instead of D-R-D-U (default: True).
        """
        self.relay_std_ratio = relay_std_ratio
        self.box_ceiling_window = box_ceiling_window
        self.ma_diff_threshold = ma_diff_threshold
        self.use_merged_segments_for_signal = use_merged_segments_for_signal

    def analyze(
        self,
        df: pl.DataFrame,
        ts_code_col: str = "ts_code",
        keep_relay_debug: bool = False,
    ) -> pl.DataFrame:
        """
        Run state chain topology logic: RLE compression -> state chain -> pattern matching.

        Expects columns: trade_date, open, high, low, close, volume.
        If ts_code_col exists, all rolling/over is per stock.

        Args:
            keep_relay_debug: If True, add columns _debug_center_drift, _debug_price_to_center,
                _debug_is_relay_recursive, _debug_is_relay_traditional to inspect why a day is Relay.

        Returns:
            DataFrame with raw_state, state_id, s0-s5 (state chain), last_relay_high,
            s4_barrier, signal_4, signal_6, current_stage.
        """
        if len(df) < 10:
            return df

        over = [ts_code_col] if ts_code_col in df.columns else []

        def roll(expr: pl.Expr) -> pl.Expr:
            return expr.over(over) if over else expr

        # --- Step 1: State detection based on MA2 inflection point (拐点先行) ---
        df = df.with_columns([
            roll(pl.col("close").rolling_mean(2)).alias("ma2"),
            roll(pl.col("close").rolling_mean(3)).alias("ma3"),
            roll(pl.col("close").rolling_mean(5)).alias("ma5"),
        ])
        # 关键改进1：基于 MA2 拐点的状态切换（解决滞后性）
        df = df.with_columns([
            (pl.col("ma2").diff(1) > 0).alias("ma2_up"),    # MA2 拐头向上
            (pl.col("ma2").diff(1) < 0).alias("ma2_down"),  # MA2 拐头向下
        ])
        # 上升态起始于 MA2 拐头向上且 Close 站上 MA3
        is_up_start = pl.col("ma2_up") & (pl.col("close") > pl.col("ma3"))
        # 下跌态起始于 MA2 拐头向下且 Close 跌破 MA3
        is_down_start = pl.col("ma2_down") & (pl.col("close") < pl.col("ma3"))
        
        # --- Step 1.5: Dynamic recursive relay (中继动态递归) ---
        # 关键改进2：中继从开始纠缠时递归延伸，判断是否包含新节点要看是否大幅移动框内均值
        df = df.with_columns(
            ((pl.col("high") + pl.col("low")) / 2.0).alias("mid_p")
        )
        df = df.with_columns(
            roll(pl.col("mid_p").rolling_mean(5)).alias("relay_center")
        )
        # 中心线漂移：如果中心线漂移极小，且价格距离中心线很近，判定为"递归中继"
        center_drift = pl.col("relay_center").diff(1).abs() / (pl.col("relay_center") + 1e-10)
        price_to_center = (pl.col("close") - pl.col("relay_center")).abs() / (pl.col("relay_center") + 1e-10)
        is_relay_recursive = (center_drift < 0.003) & (price_to_center < 0.01)
        
        # 传统中继判定（均线纠缠）
        diff_23 = (pl.col("ma2") - pl.col("ma3")).abs()
        diff_35 = (pl.col("ma3") - pl.col("ma5")).abs()
        ma_std = _relay_std_expr()
        is_relay_traditional = (
            (ma_std / (pl.col("ma5") + 1e-10) < self.relay_std_ratio)
            | ((diff_23 / (pl.col("ma5") + 1e-10) < self.ma_diff_threshold) & (diff_35 / (pl.col("ma5") + 1e-10) < self.ma_diff_threshold))
        )
        # 中继 = 递归中继 OR 传统中继
        is_relay = is_relay_recursive | is_relay_traditional

        if keep_relay_debug:
            df = df.with_columns([
                center_drift.alias("_debug_center_drift"),
                price_to_center.alias("_debug_price_to_center"),
                is_relay_recursive.alias("_debug_is_relay_recursive"),
                is_relay_traditional.alias("_debug_is_relay_traditional"),
            ])

        # Up/Down 状态：基于拐点起始，延续到下一个拐点或中继
        # 状态优先级：中继 > 上升起始 > 下跌起始
        df = df.with_columns(
            pl.when(is_relay).then(pl.lit(0))  # 中继优先
            .when(is_up_start).then(pl.lit(1))  # 上升起始
            .when(is_down_start).then(pl.lit(-1))  # 下跌起始
            .otherwise(None)  # 未定义状态，需要 forward_fill
            .alias("raw_state_init")
        )
        # Forward fill 状态，使状态延续到下一个拐点
        # 这样，一旦在拐点标记了状态，该状态会一直延续直到下一个拐点或中继
        df = df.with_columns(
            roll(pl.col("raw_state_init").forward_fill()).alias("raw_state")
        )
        df = df.drop("raw_state_init")
        
        # 关键改进：引入"准上升态" (Early Up) - 基于拐点
        df = df.with_columns(
            (
                pl.col("ma2_up")
                & (pl.col("close") > pl.col("ma3"))
                & (~is_relay)
            ).alias("is_early_up")
        )

        # --- Step 2: State compression (Run Length Encoding) ---
        # state_changed: True when state differs from previous row
        df = df.with_columns(
            (pl.col("raw_state") != pl.col("raw_state").shift(1))
            .fill_null(True)
            .over(over)
            .alias("state_changed")
        )
        # state_id: cumulative sum of changes = unique ID for each consecutive state segment
        df = df.with_columns(
            roll(pl.col("state_changed").cast(pl.Int32).cum_sum()).alias("state_id")
        )

        # --- Step 3: State chain extraction (s0=current, s1=prev segment, s2=prev-prev...) ---
        # Build state segment map: state_id -> state_value (from first row of each segment)
        state_seg_map = (
            df.group_by(over + ["state_id"])
            .agg(pl.col("raw_state").first().alias("seg_state_val"))
            .sort(over + ["state_id"])
        )
        # Join to get current segment's state (s0)
        s0_cols = [ts_code_col, "state_id", "seg_state_val"] if over else ["state_id", "seg_state_val"]
        df = df.join(
            state_seg_map.select(s0_cols).rename({"seg_state_val": "s0"}),
            on=over + ["state_id"],
            how="left",
        )
        # For s1..s5: join state_seg_map shifted by n state_ids
        for n in range(1, 6):
            prev_seg_map = state_seg_map.with_columns(
                (pl.col("state_id") + n).alias("_target_sid")
            )
            prev_cols = [ts_code_col, "_target_sid", "seg_state_val"] if over else ["_target_sid", "seg_state_val"]
            prev_seg_map = prev_seg_map.select(prev_cols).rename({"seg_state_val": f"s{n}"})
            if over:
                df = df.join(
                    prev_seg_map,
                    left_on=over + ["state_id"],
                    right_on=[ts_code_col, "_target_sid"],
                    how="left",
                )
            else:
                df = df.join(
                    prev_seg_map,
                    left_on="state_id",
                    right_on="_target_sid",
                    how="left",
                )
            # Forward fill s{n} so all rows in a segment have the same value
            df = df.with_columns(roll(pl.col(f"s{n}").forward_fill()))
            # Drop _target_sid if it exists (join may not create it in some cases)
            try:
                df = df.drop("_target_sid")
            except Exception:
                pass  # Column doesn't exist or other error, skip

        # --- Step 3.5: Large-level relay detection (大级别中继：D+U 区间重合) ---
        # 关键改进3：如果一段下跌(D)的价格区间与随后的上升(U)区间高度重合，合体为大级别中继(R)
        # 计算每个状态段的波段覆盖（high/low）- 基于 state_id 分组
        seg_bounds = (
            df.group_by(over + ["state_id"])
            .agg([
                pl.col("high").max().alias("seg_high"),
                pl.col("low").min().alias("seg_low"),
                pl.col("raw_state").first().alias("seg_state"),
            ])
            .sort(over + ["state_id"])
        )
        # 为每个 state_id 添加前一个段的边界（用于重合度计算）
        prev_seg_bounds = seg_bounds.with_columns(
            (pl.col("state_id") + 1).alias("_target_sid")
        ).select([
            ts_code_col if over else pl.lit(None).alias("_dummy"),
            "_target_sid",
            pl.col("seg_high").alias("prev_seg_high"),
            pl.col("seg_low").alias("prev_seg_low"),
            pl.col("seg_state").alias("prev_seg_state"),
        ])
        if not over:
            prev_seg_bounds = prev_seg_bounds.drop("_dummy")
        # 合并当前段和前一段的边界信息
        seg_bounds = seg_bounds.join(
            prev_seg_bounds,
            left_on=over + ["state_id"],
            right_on=([ts_code_col] if over else []) + ["_target_sid"],
            how="left",
        )
        # 计算重合度
        overlap_high = pl.min_horizontal(pl.col("seg_high"), pl.col("prev_seg_high"))
        overlap_low = pl.max_horizontal(pl.col("seg_low"), pl.col("prev_seg_low"))
        overlap_length = (overlap_high - overlap_low).clip(lower_bound=0)
        seg_length = (pl.col("seg_high") - pl.col("seg_low")).clip(lower_bound=1e-10)
        seg_bounds = seg_bounds.with_columns(
            (overlap_length / seg_length).alias("overlap_ratio")
        )
        # 如果重合度超过阈值，且当前是U段，前一个是D段，则标记为大级别中继
        seg_bounds = seg_bounds.with_columns(
            (
                (pl.col("overlap_ratio") > 0.6)  # 重合度超过 60%
                & (pl.col("seg_state") == 1)  # 当前是上升段
                & (pl.col("prev_seg_state") == -1)  # 前一个是下跌段
            ).alias("is_large_relay")
        )
        # 将大级别中继标记合并回主 DataFrame
        seg_bounds = seg_bounds.select(
            over + ["state_id", "is_large_relay"]
        )
        df = df.join(seg_bounds, on=over + ["state_id"], how="left")
        # 将大级别中继的 raw_state 从 U 改为 R (0)
        df = df.with_columns(
            pl.when(pl.col("is_large_relay").fill_null(False))
            .then(pl.lit(0))
            .otherwise(pl.col("raw_state"))
            .alias("raw_state")
        )
        df = df.drop("is_large_relay")
        # 重新计算 state_changed 和 state_id（因为 raw_state 可能被修改）
        df = df.with_columns(
            (pl.col("raw_state") != pl.col("raw_state").shift(1))
            .fill_null(True)
            .over(over)
            .alias("state_changed")
        )
        df = df.with_columns(
            roll(pl.col("state_changed").cast(pl.Int32).cum_sum()).alias("state_id")
        )
        # 重新提取状态链（因为 state_id 可能变化）
        state_seg_map = (
            df.group_by(over + ["state_id"])
            .agg(pl.col("raw_state").first().alias("seg_state_val"))
            .sort(over + ["state_id"])
        )
        # 重新定义 s0_cols（因为 state_seg_map 可能变化）
        s0_cols = [ts_code_col, "state_id", "seg_state_val"] if over else ["state_id", "seg_state_val"]
        df = df.join(
            state_seg_map.select(s0_cols).rename({"seg_state_val": "s0"}),
            on=over + ["state_id"],
            how="left",
        )
        for n in range(1, 6):
            prev_seg_map = state_seg_map.with_columns(
                (pl.col("state_id") + n).alias("_target_sid")
            )
            prev_cols = [ts_code_col, "_target_sid", "seg_state_val"] if over else ["_target_sid", "seg_state_val"]
            prev_seg_map = prev_seg_map.select(prev_cols).rename({"seg_state_val": f"s{n}"})
            if over:
                df = df.join(
                    prev_seg_map,
                    left_on=over + ["state_id"],
                    right_on=[ts_code_col, "_target_sid"],
                    how="left",
                )
            else:
                df = df.join(
                    prev_seg_map,
                    left_on="state_id",
                    right_on="_target_sid",
                    how="left",
                )
            df = df.with_columns(roll(pl.col(f"s{n}").forward_fill()))
            try:
                df = df.drop("_target_sid")
            except Exception:
                pass

        # --- Step 4: Recent relay ceiling (最近中继天花板) - 修正锚定逻辑 ---
        # 关键改进：突破的是"最近一个"有效中继，而不是"历史上最高"的
        # 双重箱体回溯：记录这个中继窗口的局部最高，而不是全局最高
        df = df.with_columns(
            roll(
                pl.when(pl.col("raw_state") == 0)  # 只在 Relay 状态时记录
                .then(pl.col("high").rolling_max(5))  # 记录这个中继窗口的局部最高（5天窗口）
                .otherwise(None)
            ).alias("_relay_high_local")
        )
        df = df.with_columns(
            roll(pl.col("_relay_high_local").forward_fill()).alias("recent_relay_ceiling")
        )
        df = df.drop("_relay_high_local")
        # Box ceiling (大级别物理天花板) - 保留用于对比
        df = df.with_columns(
            pl.when(pl.col("raw_state") <= 0)  # 只要不是稳定上升态（D或R）
            .then(pl.col("high"))
            .otherwise(None)
            .rolling_max(window_size=self.box_ceiling_window)  # 最近N天非上升态形成的物理天花板
            .over(over)
            .alias("box_ceiling")
        )
        df = df.with_columns(
            roll(pl.col("box_ceiling").forward_fill()).alias("box_ceiling")
        )
        # Also keep last_relay_high for backward compatibility
        df = df.with_columns(
            pl.when(pl.col("raw_state") == 0).then(pl.col("high")).otherwise(None).alias("r_high")
        )
        df = df.with_columns(
            roll(pl.col("r_high").forward_fill()).alias("last_relay_high")
        )
        df = df.drop("r_high")

        # --- Step 4.5: Cross-period nesting validation (跨周期嵌套) ---
        # TODO: 关键改进4：日线中继应该在60分钟线上找DRU的连续组合，都在一个空间内
        # 注意：这需要额外的60分钟数据源，当前版本暂不实现

        # --- Step 4.6: Merged segments for signal (合并后的去判定信号) ---
        merged_s0_col = pl.lit(None).cast(pl.Int32)
        merged_s1_col = pl.lit(None).cast(pl.Int32)
        merged_s2_col = pl.lit(None).cast(pl.Int32)
        merged_s3_col = pl.lit(None).cast(pl.Int32)
        if self.use_merged_segments_for_signal:
            date_col = "trade_date"
            raw_seg = df.group_by(over + ["state_id"]).agg([
                pl.col(date_col).min().alias("开始时间"),
                pl.col(date_col).max().alias("结束时间"),
                pl.col("raw_state").first().alias("raw_state"),
            ]).with_columns(
                pl.when(pl.col("raw_state") == 1).then(pl.lit("上涨"))
                .when(pl.col("raw_state") == -1).then(pl.lit("下跌"))
                .otherwise(pl.lit("中继"))
                .alias("状态")
            )
            merged_rows = []
            if over:
                for tc in raw_seg[ts_code_col].unique().to_list():
                    sub = raw_seg.filter(pl.col(ts_code_col) == tc).sort("开始时间").select(["状态", "开始时间", "结束时间"]).to_dicts()
                    merged = _merge_single_day_relay_segments(sub, "开始时间", "结束时间")
                    for i, r in enumerate(merged):
                        state_int = -1 if r["状态"] == "下跌" else (1 if r["状态"] == "上涨" else 0)
                        merged_rows.append({ts_code_col: tc, "merged_seg_id": i + 1, "merged_state_int": state_int, "开始时间": r["开始时间"], "结束时间": r["结束时间"]})
            else:
                sub = raw_seg.sort("开始时间").select(["状态", "开始时间", "结束时间"]).to_dicts()
                merged = _merge_single_day_relay_segments(sub, "开始时间", "结束时间")
                for i, r in enumerate(merged):
                    state_int = -1 if r["状态"] == "下跌" else (1 if r["状态"] == "上涨" else 0)
                    merged_rows.append({"merged_seg_id": i + 1, "merged_state_int": state_int, "开始时间": r["开始时间"], "结束时间": r["结束时间"]})
            merged_seg_df = pl.DataFrame(merged_rows)
            if not merged_seg_df.is_empty():
                # Assign each (ts_code, trade_date) to merged_seg_id and merged_s0
                if over:
                    j = df.join(merged_seg_df, on=over, how="left")
                else:
                    j = df.join(merged_seg_df, how="cross")
                j = j.filter(
                    (pl.col(date_col) >= pl.col("开始时间")) & (pl.col(date_col) <= pl.col("结束时间"))
                )
                j = j.unique(subset=over + [date_col], keep="first")
                join_on = over + [date_col]
                df = df.join(
                    j.select(join_on + ["merged_seg_id", "merged_state_int"]),
                    on=join_on,
                    how="left",
                )
                df = df.with_columns(pl.col("merged_state_int").alias("merged_s0"))
                # merged_s1, s2, s3 = state of segment (merged_seg_id - 1, -2, -3)
                for n, name in [(1, "merged_s1"), (2, "merged_s2"), (3, "merged_s3")]:
                    prev_cols = over + [pl.col("merged_seg_id").add(n).alias("merged_seg_id"), pl.col("merged_state_int").alias(name)]
                    prev_df = merged_seg_df.select(prev_cols)
                    df = df.join(prev_df, on=over + ["merged_seg_id"], how="left")
                merged_s0_col = pl.col("merged_s0")
                merged_s1_col = pl.col("merged_s1")
                merged_s2_col = pl.col("merged_s2")
                merged_s3_col = pl.col("merged_s3")

        # --- Step 5: Pattern matching with physical breakout and early detection ---
        # Signal 4: raw [D, R, D, U] = s3=-1(D), s2=0(R), s1=-1(D), s0=1(U); merged [D, U, D, U] = merged_s3=-1, merged_s2=1, merged_s1=-1, merged_s0=1
        # 关键改进1：使用"准上升态" (Early Up) 捕捉大阳线突破瞬间，不等均线完全排好队
        # 关键改进2：突破"最近一个"有效中继（recent_relay_ceiling），而不是历史最高
        # 关键改进3：时序容错 - 允许下跌、中继、反转之间有重叠混杂
        pattern_4_strict = (
            (pl.col("s0") == 1)   # U (当前上升)
            & (pl.col("s1") == -1)  # D (上一个下跌)
            & (pl.col("s2") == 0)   # R (上上个中继)
            & (pl.col("s3") == -1)  # D (上上上个下跌)
        )
        # 时序容错版本：允许 s0=1 或 is_early_up，且之前确实跌过、盘过
        had_down = _rolling_any(
            (pl.col("raw_state") == -1).shift(5).cast(pl.Int32),
            15,
        )
        had_relay = _rolling_any(
            (pl.col("raw_state") == 0).shift(2).cast(pl.Int32),
            10,
        )
        df = df.with_columns([
            roll(had_down).alias("_had_down"),
            roll(had_relay).alias("_had_relay"),
        ])
        # 物理突破：close > recent_relay_ceiling (最近中继的局部最高，不是历史最高)
        physical_breakout_4 = pl.col("close") > roll(pl.col("recent_relay_ceiling").shift(1))
        # Signal 4: 严格模式（完美状态链）或容错模式（early_up + 时序容错）
        signal_4_strict = pattern_4_strict & physical_breakout_4
        signal_4_tolerant = (
            pl.col("is_early_up")
            & pl.col("_had_down")
            & pl.col("_had_relay")
            & physical_breakout_4
        )
        # 合并段判定：D-U-D-U（合并后无单日中继，与导出表一致）
        pattern_4_merged = (
            (merged_s0_col == 1)
            & (merged_s1_col == -1)
            & (merged_s2_col == 1)
            & (merged_s3_col == -1)
        )
        signal_4_merged = pattern_4_merged.fill_null(False) & physical_breakout_4
        if self.use_merged_segments_for_signal:
            df = df.with_columns(
                (signal_4_merged | signal_4_tolerant).alias("signal_4")
            )
        else:
            df = df.with_columns(
                (signal_4_strict | signal_4_tolerant).alias("signal_4")
            )
        df = df.drop(["_had_down", "_had_relay"])

        # s4_barrier: recent_relay_ceiling at signal_4 (最近中继天花板), forward_fill
        # 与 stage4 时对比的中继高点一致
        df = df.with_columns(
            pl.when(pl.col("signal_4"))
            .then(pl.col("recent_relay_ceiling"))
            .otherwise(None)
            .alias("_s4_barrier")
        )
        df = df.with_columns(
            roll(pl.col("_s4_barrier").forward_fill()).alias("s4_barrier")
        )
        df = df.drop("_s4_barrier")

        # Signal 6: [U, R, U] = s0=1, s1=0, s2=1, and close > s4_barrier (物理突破), and signal_4 was False
        pattern_6 = (
            (pl.col("s0") == 1)
            & (pl.col("s1") == 0)
            & (pl.col("s2") == 1)
        )
        # 物理突破：close > s4_barrier (与 stage4 一致的天花板)
        physical_breakout_6 = pl.col("close") > pl.col("s4_barrier")
        df = df.with_columns(
            (
                pattern_6
                & physical_breakout_6
                & (pl.col("signal_4") == False)
            ).alias("signal_6")
        )

        # current_stage
        df = df.with_columns(
            pl.when(pl.col("signal_6"))
            .then(pl.lit(6))
            .when(pl.col("signal_4"))
            .then(pl.lit(4))
            .otherwise(pl.lit(None))
            .cast(pl.Int32)
            .alias("current_stage")
        )

        # --- Stage interval dates: extract from state segments ---
        # For each state_id, get min/max trade_date (segment start/end)
        seg_dates = (
            df.group_by(over + ["state_id"])
            .agg([
                pl.col("trade_date").min().alias("seg_start_date"),
                pl.col("trade_date").max().alias("seg_end_date"),
            ])
            .sort(over + ["state_id"])
        )
        # For signal_4 pattern [D, R, D, U]: s3=D, s2=R, s1=D, s0=U
        # Join segment dates for state_id - 3, -2, -1
        for n in [3, 2, 1]:
            prev_dates = seg_dates.with_columns(
                (pl.col("state_id") + n).alias("_target_sid")
            )
            prev_cols = [ts_code_col, "_target_sid", "seg_start_date", "seg_end_date"] if over else ["_target_sid", "seg_start_date", "seg_end_date"]
            if n == 3:
                # s3: first D (第一个下跌段)
                prev_dates = prev_dates.select(prev_cols).rename({
                    "seg_start_date": "d_start",
                    "seg_end_date": "d_end",
                })
            elif n == 2:
                # s2: R (中继段)
                prev_dates = prev_dates.select(prev_cols).rename({
                    "seg_start_date": "r_start",
                    "seg_end_date": "r_end",
                })
            else:  # n == 1
                # s1: second D (第二个下跌段，signal_6 也会用到 s1，但 signal_6 的 s1 是 R)
                prev_dates = prev_dates.select(prev_cols).rename({
                    "seg_start_date": "d2_start",
                    "seg_end_date": "d2_end",
                })
            if over:
                df = df.join(
                    prev_dates,
                    left_on=over + ["state_id"],
                    right_on=[ts_code_col, "_target_sid"],
                    how="left",
                )
            else:
                df = df.join(
                    prev_dates,
                    left_on="state_id",
                    right_on="_target_sid",
                    how="left",
                )
            # Drop _target_sid if it exists
            try:
                df = df.drop("_target_sid")
            except pl.exceptions.ColumnNotFoundError:
                pass  # Column doesn't exist, skip
        # For signal_6: [U, R, U] = s2=U, s1=R, s0=U
        # s2 (U): u_start, u_end (上升段)
        # s1 (R): r2_start, r2_end (中继段，但需要单独获取，因为 signal_4 的 s1 是 D)
        # Get s2 (U) segment dates: state_id - 2
        u_dates = seg_dates.with_columns(
            (pl.col("state_id") + 2).alias("_target_sid")
        )
        u_cols = [ts_code_col, "_target_sid", "seg_start_date", "seg_end_date"] if over else ["_target_sid", "seg_start_date", "seg_end_date"]
        u_dates = u_dates.select(u_cols).rename({
            "seg_start_date": "u_start",
            "seg_end_date": "u_end",
        })
        if over:
            df = df.join(
                u_dates,
                left_on=over + ["state_id"],
                right_on=[ts_code_col, "_target_sid"],
                how="left",
            )
        else:
            df = df.join(
                u_dates,
                left_on="state_id",
                right_on="_target_sid",
                how="left",
            )
        try:
            df = df.drop("_target_sid")
        except Exception:
            pass
        # Get s1 (R) for signal_6: when s1 == 0 (Relay), get its segment dates
        r2_dates = seg_dates.with_columns(
            (pl.col("state_id") + 1).alias("_target_sid")
        )
        r2_cols = [ts_code_col, "_target_sid", "seg_start_date", "seg_end_date"] if over else ["_target_sid", "seg_start_date", "seg_end_date"]
        r2_dates = r2_dates.select(r2_cols).rename({
            "seg_start_date": "r2_start",
            "seg_end_date": "r2_end",
        })
        if over:
            df = df.join(
                r2_dates,
                left_on=over + ["state_id"],
                right_on=[ts_code_col, "_target_sid"],
                how="left",
            )
        else:
            df = df.join(
                r2_dates,
                left_on="state_id",
                right_on="_target_sid",
                how="left",
            )
        try:
            df = df.drop("_target_sid")
        except Exception:
            pass
        # signal_4 occurrence date (for signal_6)
        df = df.with_columns(
            pl.when(pl.col("signal_4"))
            .then(pl.col("trade_date"))
            .otherwise(None)
            .alias("_s4_date")
        )
        df = df.with_columns([
            roll(pl.col("_s4_date").forward_fill()).alias("s4_start"),
            roll(pl.col("_s4_date").forward_fill()).alias("s4_end"),
        ])
        df = df.drop("_s4_date")
        # Keep r2_start/r2_end for signal_6 (s1=R), but drop d2 (not needed in output)
        for col in ["d2_start", "d2_end"]:
            try:
                df = df.drop(col)
            except Exception:
                pass

        return df
