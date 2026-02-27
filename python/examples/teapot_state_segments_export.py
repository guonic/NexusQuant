# -*- coding: utf-8 -*-
"""
Export 下跌/中继/上涨 状态段为 CSV（状态, 开始时间, 结束时间）.

基于 TopologicalTrendScanner 的核心状态（raw_state: 1=上涨, -1=下跌, 0=中继），
在单只股票日线数据上跑算法，按 state_id 聚合得到每段的开始/结束时间，输出三列 CSV。

可选 --two-segment：合并为「一段下降 + 一段上升」。下跌段与紧随其后的中继合并为「下降」，
上涨段单独为「上升」，适合你图中「左侧下跌/盘整 + 右侧拉升」的表示。

Usage:
  PYTHONPATH=python python examples/teapot_state_segments_export.py --symbol 600487.SH --start-date 2024-01-01 --end-date 2024-12-31 --output outputs/teapot/state_segments_600487.csv
  PYTHONPATH=python python examples/teapot_state_segments_export.py --symbol 600487 --end-date 2024-12-31
  PYTHONPATH=python python examples/teapot_state_segments_export.py --symbol 600487.SH --end-date 2024-12-31 --two-segment --output outputs/teapot/state_two_segment.csv
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import polars as pl

from nq.config import DatabaseConfig, load_config
from nq.data.processor.teapot import TeapotDataLoader
from nq.trading.selector.teapot.topological_trend_scanner import TopologicalTrendScanner
from nq.utils.data_normalize import normalize_stock_code

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _is_single_day(r: dict, date_col_begin: str = "开始时间", date_col_end: str = "结束时间") -> bool:
    """判断该段是否仅一天（开始时间 == 结束时间）。"""
    b = r.get(date_col_begin)
    e = r.get(date_col_end)
    if b is None or e is None:
        return False
    return str(b) == str(e)


def _segment_days(r: dict, date_col_begin: str = "开始时间", date_col_end: str = "结束时间") -> int:
    """段内天数（近似：按日期字符串差）。"""
    b = r.get(date_col_begin)
    e = r.get(date_col_end)
    if b is None or e is None:
        return 0
    try:
        t0 = datetime.strptime(str(b)[:10], "%Y-%m-%d")
        t1 = datetime.strptime(str(e)[:10], "%Y-%m-%d")
        return max(0, (t1 - t0).days) + 1
    except Exception:
        return 1 if str(b) == str(e) else 2


def _is_short_segment(r: dict, max_days: int = 2, date_col_begin: str = "开始时间", date_col_end: str = "结束时间") -> bool:
    """是否短段（1 日或 2 日）。"""
    return _segment_days(r, date_col_begin, date_col_end) <= max_days


def _clean_single_day_relay_recursive(
    rows: list[dict],
    date_col_begin: str = "开始时间",
    date_col_end: str = "结束时间",
) -> list[dict]:
    """
    递归清理无效 R：当 R 为单日时，
    - DRD -> D（合并为一段下跌）
    - URU -> U（合并为一段上涨）
    - DRU -> DU（去掉 R，保留 D、U）
    直到无法再合并/删除为止。
    """
    if len(rows) < 3:
        return rows
    changed = False
    out: list[dict] = []
    i = 0
    while i < len(rows):
        r = rows[i]
        lab = r.get("状态")
        if lab != "中继" or not _is_single_day(r, date_col_begin, date_col_end):
            out.append(r)
            i += 1
            continue
        prev = out[-1] if out else None
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        if prev is None or nxt is None:
            out.append(r)
            i += 1
            continue
        plab = prev.get("状态")
        nlab = nxt.get("状态")
        if plab == "下跌" and nlab == "下跌":
            # DRD -> D
            out[-1] = {"状态": "下跌", date_col_begin: prev[date_col_begin], date_col_end: nxt[date_col_end]}
            i += 2
            changed = True
            continue
        if plab == "上涨" and nlab == "上涨":
            # URU -> U
            out[-1] = {"状态": "上涨", date_col_begin: prev[date_col_begin], date_col_end: nxt[date_col_end]}
            i += 2
            changed = True
            continue
        if (plab == "下跌" and nlab == "上涨") or (plab == "上涨" and nlab == "下跌"):
            # DRU -> DU 或 URD -> UD：去掉 R
            i += 1
            changed = True
            continue
        out.append(r)
        i += 1
    if changed:
        return _clean_single_day_relay_recursive(out, date_col_begin, date_col_end)
    return out


def _align_segment_boundaries(
    rows: list[dict],
    date_col_begin: str = "开始时间",
    date_col_end: str = "结束时间",
) -> list[dict]:
    """相邻段共享边界日：后一段的开始时间 = 前一段的结束时间（下跌到13号则上涨也从13号开始）。"""
    if len(rows) <= 1:
        return rows
    out = [dict(rows[0])]
    for i in range(1, len(rows)):
        r = dict(rows[i])
        r[date_col_begin] = out[-1][date_col_end]
        out.append(r)
    return out


def _clean_single_day_relay(seg: pl.DataFrame, date_col_begin: str = "开始时间", date_col_end: str = "结束时间") -> pl.DataFrame:
    """对分段表做递归清理单日中继，返回新分段表（三列：状态, 开始时间, 结束时间）。"""
    rows = seg.sort(date_col_begin).to_dicts()
    cleaned = _clean_single_day_relay_recursive(rows, date_col_begin, date_col_end)
    if not cleaned:
        return seg.select(["状态", date_col_begin, date_col_end]).sort(date_col_begin)
    return pl.DataFrame(cleaned).sort(date_col_begin)


def _merge_short_up_into_prev_relay_recursive(
    rows: list[dict],
    date_col_begin: str = "开始时间",
    date_col_end: str = "结束时间",
    max_short_days: int = 2,
) -> list[dict]:
    """
    递归：中继 + 短上涨(1～2 日) → 合并为一段 上升。
    这样 24（单日上涨）会与左侧中继合并为 上升；单日/两日 DU 噪声被消除。
    """
    if len(rows) < 2:
        return rows
    changed = False
    out: list[dict] = []
    i = 0
    while i < len(rows):
        r = rows[i]
        lab = r.get("状态")
        if lab == "上涨" and _is_short_segment(r, max_days=max_short_days, date_col_begin=date_col_begin, date_col_end=date_col_end):
            prev = out[-1] if out else None
            if prev is not None and prev.get("状态") == "中继":
                out[-1] = {"状态": "上升", date_col_begin: prev[date_col_begin], date_col_end: r[date_col_end]}
                changed = True
                i += 1
                continue
        out.append(r)
        i += 1
    if changed:
        return _merge_short_up_into_prev_relay_recursive(out, date_col_begin, date_col_end, max_short_days)
    return out


def _merge_short_up_into_prev_relay(seg: pl.DataFrame, date_col_begin: str = "开始时间", date_col_end: str = "结束时间") -> pl.DataFrame:
    """中继 + 短上涨(1～2 日) → 上升，递归直到不变。"""
    rows = seg.sort(date_col_begin).to_dicts()
    merged = _merge_short_up_into_prev_relay_recursive(rows, date_col_begin, date_col_end)
    if not merged:
        return seg.select(["状态", date_col_begin, date_col_end]).sort(date_col_begin)
    return pl.DataFrame(merged).sort(date_col_begin)


def _merge_relay_into_down(seg: pl.DataFrame, date_col_begin: str = "开始时间", date_col_end: str = "结束时间") -> pl.DataFrame:
    """
    将「下跌 + 中继 + 下跌」等连续下跌/中继合并为一段「下降」，上涨/上升段单独为「上升」。
    规则：凡连续出现的 下跌、中继 都合并成一段 下降，直到遇到 上涨/上升。
    """
    rows = seg.sort(date_col_begin).to_dicts()
    out = []
    for r in rows:
        lab = r.get("状态")
        if lab == "下跌":
            if out and out[-1]["状态"] == "下降":
                out[-1]["结束时间"] = r[date_col_end]
            else:
                out.append({"状态": "下降", "开始时间": r[date_col_begin], "结束时间": r[date_col_end]})
        elif lab == "中继":
            if out and out[-1]["状态"] == "下降":
                out[-1]["结束时间"] = r[date_col_end]
            else:
                out.append({"状态": "下降", "开始时间": r[date_col_begin], "结束时间": r[date_col_end]})
        else:  # 上涨 或 上升（前一步 中继+短上涨 合并得到）
            out.append({"状态": "上升", "开始时间": r[date_col_begin], "结束时间": r[date_col_end]})
    return pl.DataFrame(out).sort("开始时间")


def run_export(
    df: pl.DataFrame,
    scanner: TopologicalTrendScanner,
    ts_code_col: str = "ts_code",
    date_col: str = "trade_date",
    debug_date: str | None = None,
    two_segment: bool = False,
) -> pl.DataFrame:
    """
    在 df 上运行拓扑扫描，按 state_id 聚合出每段状态的 状态/开始时间/结束时间。

    If debug_date is set (YYYY-MM-DD), run with keep_relay_debug and log why that day got its state.
    If two_segment is True, merge 中继 into 下跌 as 下降, so output is only 下降/上升 (一段下降+一段上升).

    Returns:
        三列 DataFrame: 状态, 开始时间, 结束时间
    """
    keep_debug = bool(debug_date)
    analyzed = scanner.analyze(df, ts_code_col=ts_code_col, keep_relay_debug=keep_debug)
    # 按股票与 state_id 聚合
    over = [ts_code_col] if ts_code_col in analyzed.columns else []
    seg = analyzed.group_by(over + ["state_id"]).agg([
        pl.col("raw_state").first().alias("raw_state"),
        pl.col(date_col).min().alias("开始时间"),
        pl.col(date_col).max().alias("结束时间"),
    ])
    # 状态数字 -> 中文标签 (1=上涨, -1=下跌, 0=中继)
    seg = seg.with_columns(
        pl.when(pl.col("raw_state") == 1)
        .then(pl.lit("上涨"))
        .when(pl.col("raw_state") == -1)
        .then(pl.lit("下跌"))
        .otherwise(pl.lit("中继"))
        .alias("状态")
    )
    # 递归清理无效 R（单日中继）：DRD->D, URU->U, DRU->DU
    if over:
        cleaned_parts = []
        for tc in seg[ts_code_col].unique().to_list():
            sub = seg.filter(pl.col(ts_code_col) == tc).select(["状态", "开始时间", "结束时间"])
            cleaned = _clean_single_day_relay(sub)
            cleaned_parts.append(cleaned.with_columns(pl.lit(tc).alias(ts_code_col)))
        seg = pl.concat(cleaned_parts).sort("开始时间")
    else:
        seg = _clean_single_day_relay(seg)
    # 中继 + 短上涨(1～2 日) → 上升，避免单日/两日 DU 噪声（如 24 并入左侧上升，25/26 并入右侧下降）
    if over:
        merged_parts = []
        for tc in seg[ts_code_col].unique().to_list():
            sub = seg.filter(pl.col(ts_code_col) == tc).select(["状态", "开始时间", "结束时间"])
            merged = _merge_short_up_into_prev_relay(sub)
            merged_parts.append(merged.with_columns(pl.lit(tc).alias(ts_code_col)))
        seg = pl.concat(merged_parts).sort("开始时间")
    else:
        seg = _merge_short_up_into_prev_relay(seg)
    if two_segment:
        if over:
            out_parts = []
            for tc in seg[ts_code_col].unique().to_list():
                sub = seg.filter(pl.col(ts_code_col) == tc).select(["状态", "开始时间", "结束时间"])
                merged = _merge_relay_into_down(sub)
                rows = merged.sort("开始时间").to_dicts()
                aligned = _align_segment_boundaries(rows, "开始时间", "结束时间")
                out_parts.append(pl.DataFrame(aligned).sort("开始时间"))
            out = pl.concat(out_parts).sort("开始时间")
        else:
            merged = _merge_relay_into_down(seg)
            rows = merged.sort("开始时间").to_dicts()
            aligned = _align_segment_boundaries(rows, "开始时间", "结束时间")
            out = pl.DataFrame(aligned).sort("开始时间")
        out = out.select(["状态", "开始时间", "结束时间"])
    else:
        if over:
            out_parts = []
            for tc in seg[ts_code_col].unique().to_list():
                sub = seg.filter(pl.col(ts_code_col) == tc).select(["状态", "开始时间", "结束时间"]).sort("开始时间")
                rows = sub.to_dicts()
                aligned = _align_segment_boundaries(rows, "开始时间", "结束时间")
                out_parts.append(pl.DataFrame(aligned).sort("开始时间"))
            out = pl.concat(out_parts).sort("开始时间")
        else:
            out = seg.select(["状态", "开始时间", "结束时间"]).sort("开始时间")
            rows = out.to_dicts()
            aligned = _align_segment_boundaries(rows, "开始时间", "结束时间")
            out = pl.DataFrame(aligned).sort("开始时间")

    if debug_date and ts_code_col in analyzed.columns:
        day = analyzed.filter(pl.col(date_col).cast(pl.Utf8) == str(debug_date))
        if not day.is_empty():
            row = day.to_dicts()[0]
            logger.info(
                "【中继调试】%s 当日 raw_state=%s → %s",
                debug_date,
                row.get("raw_state"),
                "上涨" if row.get("raw_state") == 1 else "下跌" if row.get("raw_state") == -1 else "中继",
            )
            for k in ["_debug_center_drift", "_debug_price_to_center", "_debug_is_relay_recursive", "_debug_is_relay_traditional"]:
                if k in row and row[k] is not None:
                    logger.info("  %s = %s", k, row[k])
            # 若当日为中继，说明 is_relay 为 True：要么递归中继(中心漂移<0.3%%且价格距中心<1%%)，要么传统中继(均线纠缠)
            if row.get("raw_state") == 0:
                logger.info(
                    "  该日被标为中继的原因：当日 is_relay=True，"
                    "满足「递归中继」(中心漂移<0.003 且 价格距中心<0.01) 或 「传统中继」(ma2/ma3/ma5 纠缠)。"
                )
        else:
            logger.warning("未找到 %s 的日线数据，无法输出中继调试信息", debug_date)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export 下跌/中继/上涨 状态段为 CSV（状态, 开始时间, 结束时间）"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="股票代码，如 600487 或 600487.SH",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="开始日期 YYYY-MM-DD，不填则默认 end-date 往前 1 年",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="结束日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 CSV 路径，不填则 outputs/teapot/state_segments_{ts_code}.csv",
    )
    parser.add_argument("--debug-date", type=str, default=None, help="指定日期 YYYY-MM-DD，输出该日为何被标为 下跌/中继/上涨 的调试信息（如 2025-02-07）")
    parser.add_argument("--two-segment", action="store_true", help="合并为两段：下跌+中继→一段「下降」，上涨→一段「上升」，输出仅 下降/上升")
    parser.add_argument("--use-cache", action="store_true", help="使用本地缓存数据")
    parser.add_argument("--relay-std-ratio", type=float, default=0.007)
    parser.add_argument("--box-ceiling-window", type=int, default=15)
    parser.add_argument("--ma-diff-threshold", type=float, default=0.003)
    args = parser.parse_args()

    ts_code = normalize_stock_code(args.symbol.strip())
    if not ts_code:
        logger.error("无效股票代码: %s", args.symbol)
        return

    start_date = args.start_date
    end_date = args.end_date.strip()
    if not start_date:
        # 默认 1 年
        from datetime import datetime, timedelta
        try:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            start_dt = end_dt - timedelta(days=365)
            start_date = start_dt.strftime("%Y-%m-%d")
        except ValueError:
            logger.error("end-date 格式需为 YYYY-MM-DD: %s", end_date)
            return
    logger.info("股票=%s 区间=%s ~ %s", ts_code, start_date, end_date)

    try:
        config = load_config()
        db_config = config.database
    except Exception as e:
        logger.warning("Load config failed: %s", e)
        db_config = DatabaseConfig()

    loader = TeapotDataLoader(db_config=db_config, schema="quant", use_cache=args.use_cache)
    df = loader.load_daily_data(
        start_date=start_date,
        end_date=end_date,
        symbols=[ts_code],
    )
    if df.is_empty():
        logger.error("未加载到数据，请检查 ts_code 与日期范围")
        return

    scanner = TopologicalTrendScanner(
        relay_std_ratio=args.relay_std_ratio,
        box_ceiling_window=args.box_ceiling_window,
        ma_diff_threshold=args.ma_diff_threshold,
    )
    out_df = run_export(
        df,
        scanner,
        ts_code_col="ts_code",
        date_col="trade_date",
        debug_date=args.debug_date,
        two_segment=args.two_segment,
    )

    out_path = args.output
    if not out_path:
        out_path = f"outputs/teapot/state_segments_{ts_code.replace('.', '_')}.csv"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.write_csv(out_path)
    logger.info("已写入 %d 个状态段 -> %s", len(out_df), out_path)


if __name__ == "__main__":
    main()
