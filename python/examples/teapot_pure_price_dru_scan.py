# -*- coding: utf-8 -*-
"""
Pure-price DRU 测试脚本：加载日线、跑纯价格 DRU、输出 signal_4 与状态段。

无均线，仅用 high/low/close 的空间演进与 spatial_overlap 判定 D/R/U 与 signal_4。

Usage:
  PYTHONPATH=python python python/examples/teapot_pure_price_dru_scan.py --symbol 600487.SH --start-date 2024-01-01 --end-date 2024-12-31
  PYTHONPATH=python python python/examples/teapot_pure_price_dru_scan.py --symbol 600487.SH --start-date 2024-01-01 --end-date 2024-12-31 --output outputs/teapot/pure_price_dru.csv --segments outputs/teapot/pure_price_dru_segments.csv
"""

import argparse
import logging
from pathlib import Path

import polars as pl

from nq.config import DatabaseConfig, load_config
from nq.data.processor.teapot import TeapotDataLoader
from nq.trading.selector.teapot.pure_price_dru import PurePriceDRUScanner
from nq.utils.data_normalize import normalize_stock_code

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _is_single_day_seg(r: dict, date_col_begin: str = "开始时间", date_col_end: str = "结束时间") -> bool:
    """段是否仅一天（开始时间 == 结束时间）。"""
    b, e = r.get(date_col_begin), r.get(date_col_end)
    return b is not None and e is not None and str(b) == str(e)


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


def _clean_single_day_relay_segments(
    rows: list[dict],
    date_col_begin: str = "开始时间",
    date_col_end: str = "结束时间",
) -> list[dict]:
    """
    递归：单日中继时 DRD->D, URU->U, DRU->DU，使「下跌-中继(1日)-下跌」合并为一段下跌。
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
        return _clean_single_day_relay_segments(out, date_col_begin, date_col_end)
    return out


def _parse_symbols(raw: list) -> list:
    """Normalize and expand comma-separated symbols to ts_code list."""
    parts = []
    for s in raw:
        parts.extend(p.strip() for p in s.split(",") if p.strip())
    codes = [normalize_stock_code(p) for p in parts]
    seen = set()
    return [c for c in codes if c and c not in seen and not seen.add(c)]


def _first_of_run(df: pl.DataFrame, signal_col: str, ts_code_col: str = "ts_code", date_col: str = "trade_date") -> pl.DataFrame:
    """Keep only the first date of each consecutive run of True for signal_col per ts_code."""
    if df.is_empty():
        return df
    over = [ts_code_col] if ts_code_col in df.columns else []
    sort_cols = [ts_code_col, date_col] if over else [date_col]
    df = df.sort(sort_cols)
    if over:
        df = df.with_columns(pl.col(date_col).shift(1).over(over).alias("_prev_date"))
    else:
        df = df.with_columns(pl.col(date_col).shift(1).alias("_prev_date"))
    join_on = [ts_code_col, "_prev_date"] if over else ["_prev_date"]
    prev_cols = [ts_code_col, date_col, signal_col] if over else [date_col, signal_col]
    prev = df.select(prev_cols).rename({date_col: "_prev_date", signal_col: "_prev_sig"})
    df = df.join(prev, on=join_on, how="left")
    df = df.filter(pl.col(signal_col))
    df = df.filter(
        pl.col("_prev_date").is_null()
        | pl.col("_prev_sig").is_null()
        | (pl.col("_prev_sig") == False)
    )
    df = df.drop(["_prev_date", "_prev_sig"])
    return df


def _build_merged_segment_pattern(
    analyzed: pl.DataFrame,
    ts_code_col: str,
    date_col: str,
    clean_and_align_fn,
) -> pl.DataFrame:
    """
    Build merged segments (DRD->D, URU->U, DRU->DU), map each (ts_code, date) to
    merged_s0..merged_s3, return analyzed with pattern_4_merged (D-U-D-U) column.
    """
    over = [ts_code_col] if ts_code_col in analyzed.columns else []
    sort_cols = [ts_code_col, date_col] if over else [date_col]
    analyzed_sorted = analyzed.sort(sort_cols).with_columns(
        (pl.col("p_state") != pl.col("p_state").shift(1)).fill_null(True).over(over).alias("_chg")
    )
    analyzed_sorted = analyzed_sorted.with_columns(
        pl.col("_chg").cast(pl.Int32).cum_sum().over(over).alias("_sid")
    )
    raw_seg = analyzed_sorted.group_by(over + ["_sid"]).agg([
        pl.col("p_state").first().alias("p_state"),
        pl.col(date_col).min().alias("开始时间"),
        pl.col(date_col).max().alias("结束时间"),
    ])
    raw_seg = raw_seg.with_columns(
        pl.when(pl.col("p_state") == 1)
        .then(pl.lit("上涨"))
        .when(pl.col("p_state") == -1)
        .then(pl.lit("下跌"))
        .otherwise(pl.lit("中继"))
        .alias("状态")
    )

    merged_rows: list[dict] = []
    if over:
        for tc in raw_seg[ts_code_col].unique().to_list():
            sub = raw_seg.filter(pl.col(ts_code_col) == tc).sort("开始时间").select(["状态", "开始时间", "结束时间"])
            rows = sub.to_dicts()
            cleaned = clean_and_align_fn(rows)
            for i, r in enumerate(cleaned):
                state_int = -1 if r["状态"] == "下跌" else (1 if r["状态"] == "上涨" else 0)
                merged_rows.append({
                    ts_code_col: tc,
                    "merged_seg_id": i + 1,
                    "merged_state_int": state_int,
                    "开始时间": r["开始时间"],
                    "结束时间": r["结束时间"],
                })
    else:
        rows = raw_seg.sort("开始时间").select(["状态", "开始时间", "结束时间"]).to_dicts()
        cleaned = clean_and_align_fn(rows)
        for i, r in enumerate(cleaned):
            state_int = -1 if r["状态"] == "下跌" else (1 if r["状态"] == "上涨" else 0)
            merged_rows.append({
                "merged_seg_id": i + 1,
                "merged_state_int": state_int,
                "开始时间": r["开始时间"],
                "结束时间": r["结束时间"],
            })

    if not merged_rows:
        return analyzed.with_columns(pl.lit(False).alias("pattern_4_merged"))

    merged_seg_df = pl.DataFrame(merged_rows)
    join_on_over = [ts_code_col] if over else []
    j = analyzed.select(join_on_over + [date_col]).join(merged_seg_df, on=join_on_over, how="left")
    j = j.filter(
        (pl.col(date_col) >= pl.col("开始时间")) & (pl.col(date_col) <= pl.col("结束时间"))
    )
    j = j.unique(subset=join_on_over + [date_col], keep="first")
    j = j.select(join_on_over + [date_col, "merged_seg_id", "merged_state_int"])
    j = j.rename({"merged_state_int": "merged_s0"})

    for n, name in [(1, "merged_s1"), (2, "merged_s2"), (3, "merged_s3")]:
        prev = merged_seg_df.select(
            join_on_over + [pl.col("merged_seg_id").add(n).alias("merged_seg_id"), pl.col("merged_state_int").alias(name)]
        )
        j = j.join(prev, on=join_on_over + ["merged_seg_id"], how="left")

    # 合并段模式：D-R-D-U（当前段 U，前段 D，再前 R，再前 D）；3-21 等 D-U-D-U 无中继的不命中
    pattern_4_merged = (
        (pl.col("merged_s0") == 1)
        & (pl.col("merged_s1") == -1)
        & (pl.col("merged_s2") == 0)
        & (pl.col("merged_s3") == -1)
    ).fill_null(False)
    j = j.with_columns(pattern_4_merged.alias("pattern_4_merged"))
    j = j.select(join_on_over + [date_col, "pattern_4_merged"])

    analyzed = analyzed.join(j, on=join_on_over + [date_col], how="left")
    analyzed = analyzed.with_columns(pl.col("pattern_4_merged").fill_null(False))
    return analyzed


def run_scan(
    df: pl.DataFrame,
    scanner: PurePriceDRUScanner,
    output_path: str | None = None,
    segments_path: str | None = None,
    ts_code_col: str = "ts_code",
    date_col: str = "trade_date",
) -> None:
    """Run pure-price DRU scanner, log signal_4 (first of run), optionally write CSV and state segments.
    signal_4 is only True when merged segment pattern D-U-D-U holds (合并后的去判定信号).
    """
    analyzed = scanner.analyze(df, ts_code_col=ts_code_col)

    def clean_and_align(rows: list[dict]) -> list[dict]:
        cleaned = _clean_single_day_relay_segments(rows, "开始时间", "结束时间")
        return _align_segment_boundaries(cleaned, "开始时间", "结束时间")

    analyzed = _build_merged_segment_pattern(analyzed, ts_code_col, date_col, clean_and_align)
    analyzed = analyzed.with_columns(
        (pl.col("signal_4") & pl.col("pattern_4_merged")).alias("signal_4")
    )

    over = [ts_code_col] if ts_code_col in analyzed.columns else []
    if over:
        for tc in analyzed[ts_code_col].unique().to_list():
            sub = analyzed.filter(pl.col(ts_code_col) == tc).sort(date_col)
            s4 = _first_of_run(sub, "signal_4", ts_code_col, date_col)
            for row in s4.iter_rows(named=True):
                logger.info(
                    "Pure-price signal_4: %s %s close=%.4f rolling_h_min=%.4f",
                    row.get(ts_code_col),
                    row.get(date_col),
                    row.get("close"),
                    row.get("rolling_h_min"),
                )
    else:
        sub = analyzed.sort(date_col)
        s4 = _first_of_run(sub, "signal_4", date_col=date_col)
        for row in s4.iter_rows(named=True):
            logger.info(
                "Pure-price signal_4: %s close=%.4f rolling_h_min=%.4f",
                row.get(date_col),
                row.get("close"),
                row.get("rolling_h_min"),
            )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out_cols = [c for c in analyzed.columns if c in [ts_code_col, date_col, "open", "high", "low", "close", "p_state", "spatial_overlap", "is_turn_up", "signal_4", "pattern_4_merged", "rolling_h_min", "rolling_l_max", "relay_box_high", "relay_box_low"]]
        analyzed.select(out_cols).write_csv(output_path)
        logger.info("Wrote full result to %s", output_path)

    if segments_path:
        # p_state 状态段：按 p_state 变化切段，得到 状态(下跌/中继/上涨), 开始时间, 结束时间
        sort_cols = [ts_code_col, date_col] if over else [date_col]
        analyzed_sorted = analyzed.sort(sort_cols).with_columns(
            (pl.col("p_state") != pl.col("p_state").shift(1)).fill_null(True).over(over).alias("_chg")
        )
        analyzed_sorted = analyzed_sorted.with_columns(
            pl.col("_chg").cast(pl.Int32).cum_sum().over(over).alias("_sid")
        )
        seg = analyzed_sorted.group_by(over + ["_sid"]).agg([
            pl.col("p_state").first().alias("p_state"),
            pl.col(date_col).min().alias("开始时间"),
            pl.col(date_col).max().alias("结束时间"),
        ])
        seg = seg.with_columns(
            pl.when(pl.col("p_state") == 1)
            .then(pl.lit("上涨"))
            .when(pl.col("p_state") == -1)
            .then(pl.lit("下跌"))
            .otherwise(pl.lit("中继"))
            .alias("状态")
        )
        # 单日中继合并：下跌-中继(1日)-下跌 -> 一段下跌；URU -> U
        if over:
            parts = []
            for tc in seg[ts_code_col].unique().to_list():
                sub = seg.filter(pl.col(ts_code_col) == tc).select(["状态", "开始时间", "结束时间"])
                rows = sub.sort("开始时间").to_dicts()
                cleaned = _clean_single_day_relay_segments(rows, "开始时间", "结束时间")
                cleaned = _align_segment_boundaries(cleaned, "开始时间", "结束时间")
                if cleaned:
                    part = pl.DataFrame(cleaned).with_columns(pl.lit(tc).alias(ts_code_col))
                    parts.append(part)
            seg_out = pl.concat(parts).sort("开始时间").select(["状态", "开始时间", "结束时间", ts_code_col])
        else:
            rows = seg.sort("开始时间").select(["状态", "开始时间", "结束时间"]).to_dicts()
            cleaned = _clean_single_day_relay_segments(rows, "开始时间", "结束时间")
            cleaned = _align_segment_boundaries(cleaned, "开始时间", "结束时间")
            seg_out = pl.DataFrame(cleaned).sort("开始时间") if cleaned else seg.select(["状态", "开始时间", "结束时间"]).sort("开始时间")
        Path(segments_path).parent.mkdir(parents=True, exist_ok=True)
        seg_out.write_csv(segments_path)
        logger.info(
            "Wrote state segments to %s (single-day relay merged: D-R-D->D), %d segments",
            segments_path,
            len(seg_out),
        )
        # 终端可见：打印段数与前 5 行，便于确认输出有变化
        if len(seg_out) > 0:
            head = seg_out.head(5)
            for row in head.iter_rows(named=True):
                logger.info("  segment: %s %s ~ %s", row.get("状态"), row.get("开始时间"), row.get("结束时间"))

    # 汇总输出，便于确认「有变化」
    n_rows = len(analyzed)
    n_s4 = analyzed.filter(pl.col("signal_4")).height if "signal_4" in analyzed.columns else 0
    logger.info(
        "Pure-price DRU scan done: rows=%d signal_4=%d output=%s segments=%s",
        n_rows,
        n_s4,
        output_path or "-",
        segments_path or "-",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure-price DRU scanner test")
    parser.add_argument("--start-date", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--symbols", type=str, nargs="*", help="Stock codes (e.g. 600487 or 600487.SH)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (full result)")
    parser.add_argument("--segments", type=str, default=None, help="Output CSV path for state segments (状态, 开始时间, 结束时间)")
    parser.add_argument("--rolling-n", type=int, default=5, help="Spatial overlap window (default: 5)")
    parser.add_argument("--turn-lookback", type=int, default=3, help="Turn-up lookback for low (default: 3)")
    parser.add_argument("--use-cache", action="store_true")
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols) if args.symbols else None
    if args.symbols and not symbols:
        logger.error("No valid symbols: %s", args.symbols)
        return

    try:
        config = load_config()
        db_config = config.database
    except Exception as e:
        logger.warning("Load config failed: %s", e)
        db_config = DatabaseConfig()

    loader = TeapotDataLoader(db_config=db_config, schema="quant", use_cache=args.use_cache)
    logger.info("Loading data %s to %s", args.start_date, args.end_date)
    df = loader.load_daily_data(
        start_date=args.start_date,
        end_date=args.end_date,
        symbols=symbols,
    )
    if df.is_empty():
        logger.error("No data loaded")
        return

    scanner = PurePriceDRUScanner(rolling_n=args.rolling_n, turn_lookback=args.turn_lookback)
    run_scan(
        df,
        scanner,
        output_path=args.output,
        segments_path=args.segments,
        ts_code_col="ts_code",
        date_col="trade_date",
    )


if __name__ == "__main__":
    main()
