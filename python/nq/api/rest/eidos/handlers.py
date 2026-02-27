"""
Request handlers for Eidos API endpoints.
"""

import logging
from datetime import date, datetime
from decimal import Decimal
from typing import List, Optional, Dict, Any

from fastapi import HTTPException

from nq.api.rest.eidos.dependencies import get_eidos_repo, get_kline_service
from nq.api.rest.eidos.schemas import (
    ExperimentResponse,
    LedgerEntryResponse,
    TradeResponse,
    PerformanceMetricsResponse,
    TradeStatsResponse,
    BacktestReportResponse,
    DTWHitsPageResponse,
    DTWHitRow,
    DTWAnnotationRequest,
    DTWAnnotationResponse,
)

logger = logging.getLogger(__name__)


async def get_experiments_handler() -> List[ExperimentResponse]:
    """
    Get all experiments.
    
    Returns:
        List of experiments.
    """
    repo = get_eidos_repo()
    # Get all experiments (with pagination, but we'll fetch all)
    experiments = repo.experiment.list_experiments(limit=1000, offset=0)
    
    result = []
    for exp in experiments:
        try:
            # Convert date strings to date objects if needed
            if isinstance(exp.get("start_date"), str):
                exp["start_date"] = date.fromisoformat(exp["start_date"])
            if isinstance(exp.get("end_date"), str):
                exp["end_date"] = date.fromisoformat(exp["end_date"])
            result.append(ExperimentResponse(**exp))
        except Exception as e:
            logger.warning(f"Failed to parse experiment {exp.get('exp_id')}: {e}")
            continue
    
    return result


async def get_experiment_handler(exp_id: str) -> ExperimentResponse:
    """
    Get a single experiment by ID.
    
    Args:
        exp_id: Experiment ID.
    
    Returns:
        Experiment data.
    
    Raises:
        HTTPException: If experiment not found.
    """
    repo = get_eidos_repo()
    exp = repo.experiment.get_experiment(exp_id)
    
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment {exp_id} not found")
    
    # Convert date strings to date objects if needed
    if isinstance(exp.get("start_date"), str):
        exp["start_date"] = date.fromisoformat(exp["start_date"])
    if isinstance(exp.get("end_date"), str):
        exp["end_date"] = date.fromisoformat(exp["end_date"])
    
    return ExperimentResponse(**exp)


async def get_ledger_handler(
    exp_id: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> List[LedgerEntryResponse]:
    """
    Get ledger entries for an experiment.
    
    Args:
        exp_id: Experiment ID.
        start_date: Optional start date filter.
        end_date: Optional end date filter.
    
    Returns:
        List of ledger entries.
    """
    repo = get_eidos_repo()
    entries = repo.ledger.get_ledger(exp_id, start_date=start_date, end_date=end_date)
    
    result = []
    for entry in entries:
        try:
            # Convert date strings to date objects if needed
            if isinstance(entry.get("date"), str):
                entry["date"] = date.fromisoformat(entry["date"])
            result.append(LedgerEntryResponse(**entry))
        except Exception as e:
            logger.warning(f"Failed to parse ledger entry: {e}")
            continue
    
    return result


async def get_trades_handler(
    exp_id: str,
    symbol: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> List[TradeResponse]:
    """
    Get trades for an experiment.
    
    Args:
        exp_id: Experiment ID.
        symbol: Optional symbol filter.
        start_date: Optional start date filter.
        end_date: Optional end date filter.
    
    Returns:
        List of trades.
    """
    logger.info(f"Getting trades for exp_id: {exp_id}, symbol: {symbol}, start_date: {start_date}, end_date: {end_date}")
    
    repo = get_eidos_repo()
    trades = repo.trades.get_trades(
        exp_id, start_time=start_date, end_time=end_date, symbol=symbol
    )
    
    logger.info(f"Retrieved {len(trades)} trades from repository")
    
    result = []
    for idx, trade in enumerate(trades):
        try:
            # Ensure we have 'side' field for Pydantic (it uses alias="side")
            # The repository already converts 'side' to 'direction', but Pydantic needs 'side' due to alias
            if "direction" in trade and "side" not in trade:
                trade["side"] = trade["direction"]
            elif "side" in trade and "direction" not in trade:
                trade["direction"] = trade["side"]
            
            # Convert datetime strings if needed
            if isinstance(trade.get("deal_time"), str):
                trade["deal_time"] = datetime.fromisoformat(trade["deal_time"].replace("Z", "+00:00"))
            # Ensure exp_id is present
            if "exp_id" not in trade:
                trade["exp_id"] = exp_id
            result.append(TradeResponse(**trade))
        except Exception as e:
            logger.warning(f"Failed to parse trade {idx}: {e}, trade data: {trade}", exc_info=True)
            continue
    
    logger.info(f"Successfully parsed {len(result)} trades for exp_id: {exp_id}")
    return result


async def get_performance_metrics_handler(exp_id: str) -> PerformanceMetricsResponse:
    """
    Calculate performance metrics for an experiment.
    
    Args:
        exp_id: Experiment ID.
    
    Returns:
        Performance metrics.
    
    Raises:
        HTTPException: If experiment not found or insufficient data.
    """
    repo = get_eidos_repo()
    
    # Get experiment to verify it exists
    exp = repo.experiment.get_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment {exp_id} not found")
    
    # Get ledger entries
    entries = repo.ledger.get_ledger(exp_id)
    if not entries:
        raise HTTPException(status_code=404, detail=f"No ledger data found for experiment {exp_id}")
    
    # Sort by date
    entries.sort(key=lambda x: x.get("date", ""))
    
    # Calculate metrics
    initial_nav = entries[0].get("nav", 0)
    final_nav = entries[-1].get("nav", 0)
    
    if isinstance(initial_nav, (str, Decimal)):
        initial_nav = float(initial_nav)
    if isinstance(final_nav, (str, Decimal)):
        final_nav = float(final_nav)
    
    total_return = (final_nav - initial_nav) / initial_nav if initial_nav > 0 else 0.0
    
    # Calculate max drawdown
    max_nav = initial_nav
    max_drawdown = 0.0
    for entry in entries:
        nav = entry.get("nav", 0)
        if isinstance(nav, (str, Decimal)):
            nav = float(nav)
        if nav > max_nav:
            max_nav = nav
        drawdown = (max_nav - nav) / max_nav if max_nav > 0 else 0.0
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    trading_days = len(entries)
    
    # Get metrics from experiment if available
    metrics_summary = exp.get("metrics_summary", {}) or {}
    sharpe_ratio = metrics_summary.get("sharpe_ratio")
    annual_return = metrics_summary.get("annual_return")
    
    return PerformanceMetricsResponse(
        total_return=total_return,
        max_drawdown=max_drawdown,
        final_nav=float(final_nav),  # Convert to float for JSON serialization
        trading_days=trading_days,
        sharpe_ratio=sharpe_ratio,
        annual_return=annual_return,
    )


async def get_trade_stats_handler(exp_id: str) -> TradeStatsResponse:
    """
    Calculate trade statistics for an experiment.
    
    Args:
        exp_id: Experiment ID.
    
    Returns:
        Trade statistics.
    
    Raises:
        HTTPException: If experiment not found.
    """
    repo = get_eidos_repo()
    
    # Get experiment to verify it exists
    exp = repo.experiment.get_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment {exp_id} not found")
    
    # Get all trades
    trades = repo.trades.get_trades(exp_id)
    
    logger.info(f"Retrieved {len(trades)} trades for stats calculation")
    
    if not trades:
        return TradeStatsResponse(
            total_trades=0,
            buy_count=0,
            sell_count=0,
            win_rate=0.0,
            avg_hold_days=0.0,
        )
    
    # Get side/direction value - repository converts 'side' to 'direction', but may have both
    buy_count = 0
    sell_count = 0
    for t in trades:
        # Check both 'side' and 'direction' fields
        side_value = t.get("side") or t.get("direction")
        if side_value == 1:
            buy_count += 1
        elif side_value == -1:
            sell_count += 1
        else:
            logger.warning(f"Trade {t.get('trade_id')} has invalid side/direction: {side_value}")
    
    logger.info(f"Buy count: {buy_count}, Sell count: {sell_count}, Total: {len(trades)}")
    
    # Calculate win rate from trades with pnl_ratio
    winning_trades = [t for t in trades if t.get("pnl_ratio") and t.get("pnl_ratio", 0) > 0]
    win_rate = len(winning_trades) / len(trades) if trades else 0.0
    
    # Calculate average hold days
    hold_days_list = [t.get("hold_days") for t in trades if t.get("hold_days") is not None]
    avg_hold_days = sum(hold_days_list) / len(hold_days_list) if hold_days_list else 0.0
    
    return TradeStatsResponse(
        total_trades=len(trades),
        buy_count=buy_count,
        sell_count=sell_count,
        win_rate=win_rate,
        avg_hold_days=avg_hold_days,
    )


async def get_backtest_report_handler(
    exp_id: str,
    format: Optional[str] = "json",
    categories: Optional[str] = None,
    metrics: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate complete backtest report for an experiment.
    
    Args:
        exp_id: Experiment ID.
        format: Output format (currently only 'json' is supported for API).
        categories: Comma-separated list of metric categories.
        metrics: Comma-separated list of metric names.
    
    Returns:
        Backtest report data.
    
    Raises:
        HTTPException: If experiment not found or report generation fails.
    """
    from nq.analysis.backtest.report import BacktestReportGenerator, ReportConfig
    from nq.config import load_config
    
    try:
        # Load database config
        config = load_config("config/config.yaml")
        db_config = config.database
        
        # Create report generator
        generator = BacktestReportGenerator(db_config)
        
        # Parse categories and metrics
        metric_categories = None
        if categories:
            metric_categories = [c.strip() for c in categories.split(",")]
        
        metric_names = None
        if metrics:
            metric_names = [m.strip() for m in metrics.split(",")]
        
        # Create report config
        report_config = ReportConfig(
            metric_categories=metric_categories,
            metric_names=metric_names,
            output_format=format,
        )
        
        # Generate report
        report = generator.generate_report(exp_id, config=report_config)
        
        return report
        
    except ValueError as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Report generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


async def get_stock_kline_handler(
    exp_id: str,
    symbol: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    extend_days: int = 60,
    indicators: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """
    Get K-line (OHLCV) data for a stock symbol.
    
    This handler delegates to KlineService following the handler -> service -> repo -> model architecture.
    
    Args:
        exp_id: Experiment ID.
        symbol: Stock symbol (e.g., "000001.SZ").
        start_date: Optional start date filter.
        end_date: Optional end date filter.
        extend_days: Number of trading days to extend backward (default: 60).
        indicators: Dictionary of indicator names to boolean flags.
    
    Returns:
        Dictionary with 'kline_data', 'indicators', 'backtest_start', 'backtest_end'.
    
    Raises:
        HTTPException: If experiment not found or data retrieval fails.
    """
    try:
        kline_service = get_kline_service()
        result = kline_service.get_stock_kline(
            exp_id=exp_id,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            indicators=indicators,
            extend_days=extend_days,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get K-line data for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve K-line data: {str(e)}")


# DTW Labeling handlers
async def get_dtw_hits_handler(
    filename: str,
    page: int = 1,
    page_size: int = 50,
) -> DTWHitsPageResponse:
    """
    Get paginated DTW hits from CSV file.
    
    Args:
        filename: CSV filename (relative to outputs/ or absolute path).
        page: Page number (1-based).
        page_size: Number of rows per page.
    
    Returns:
        Paginated DTW hits.
    
    Raises:
        HTTPException: If file not found or read fails.
    """
    from pathlib import Path
    import pandas as pd

    # Resolve file path (try outputs/ first, then absolute)
    csv_path = Path("outputs") / filename
    if not csv_path.exists():
        csv_path = Path(filename)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"CSV file not found: {filename}")

    try:
        df = pd.read_csv(csv_path)
        total = len(df)
        total_pages = (total + page_size - 1) // page_size

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_df = df.iloc[start_idx:end_idx]

        hits = []
        for local_idx, (_, row) in enumerate(page_df.iterrows()):
            # Calculate actual row index in CSV (0-based)
            actual_row_index = start_idx + local_idx
            hit = DTWHitRow(
                template=str(row.get("template", "")),
                ts_code=str(row.get("ts_code", "")),
                start_date=str(row.get("start_date", "")),
                end_date=str(row.get("end_date", "")),
                score=float(row.get("score", 0.0)),
                hit_count=int(row.get("hit_count", 1)) if pd.notna(row.get("hit_count")) else 1,
                start_index=actual_row_index,  # Store actual CSV row index
                end_index=int(row.get("end_index")) if pd.notna(row.get("end_index")) else None,
                label=str(row.get("label", "")) if pd.notna(row.get("label")) else None,
                platform_start=str(row.get("platform_start", "")) if pd.notna(row.get("platform_start")) else None,
                platform_end=str(row.get("platform_end", "")) if pd.notna(row.get("platform_end")) else None,
                breakout_date=str(row.get("breakout_date", "")) if pd.notna(row.get("breakout_date")) else None,
                notes=str(row.get("notes", "")) if pd.notna(row.get("notes")) else None,
            )
            hits.append(hit)

        return DTWHitsPageResponse(
            hits=hits,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )
    except Exception as e:
        logger.error(f"Failed to read CSV {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to read CSV: {str(e)}")


async def get_dtw_kline_handler(
    symbol: str,
    start_date: str,
    end_date: str,
    extend_left_bars: int = 5,
    extend_right_bars: int = 5,
) -> Dict[str, Any]:
    """
    Get K-line data for DTW labeling (with MA5, MA10).
    
    Args:
        symbol: Stock code (e.g., "600487.SH").
        start_date: Start date YYYY-MM-DD.
        end_date: End date YYYY-MM-DD.
        extend_left_bars: Number of K-line bars to extend left (default 5).
        extend_right_bars: Number of K-line bars to extend right (default 5).
    
    Returns:
        K-line data with MA5, MA10 indicators.
    
    Raises:
        HTTPException: If data retrieval fails.
    """
    from datetime import timedelta
    import pandas as pd
    from nq.config import load_config, DatabaseConfig
    from nq.repo.kline_repo import StockKlineDayRepo

    try:
        start_d = date.fromisoformat(start_date)
        end_d = date.fromisoformat(end_date)

        # First, load a wider range to find actual trading days
        # Estimate: extend by ~2x the requested bars to account for weekends/holidays
        estimated_extend_days = max(extend_left_bars * 2, 20)
        extended_start = start_d - timedelta(days=estimated_extend_days)
        extended_end = end_d + timedelta(days=max(extend_right_bars * 2, 10))

        try:
            config = load_config()
            db_config = config.database
        except Exception:
            db_config = DatabaseConfig()

        kline_repo = StockKlineDayRepo(db_config, schema="quant")
        klines = kline_repo.get_by_ts_code(
            ts_code=symbol,
            start_time=datetime.combine(extended_start, datetime.min.time()),
            end_time=datetime.combine(extended_end, datetime.max.time()),
        )

        if not klines:
            raise HTTPException(status_code=404, detail=f"No K-line data for {symbol}")

        kline_data = []
        for k in klines:
            trade_date = k.trade_date
            if isinstance(trade_date, datetime):
                date_str = trade_date.strftime("%Y-%m-%d")
            elif isinstance(trade_date, date):
                date_str = trade_date.strftime("%Y-%m-%d")
            else:
                date_str = str(trade_date)

            kline_data.append({
                "date": date_str,
                "open": float(k.open) if k.open else 0.0,
                "high": float(k.high) if k.high else 0.0,
                "low": float(k.low) if k.low else 0.0,
                "close": float(k.close) if k.close else 0.0,
                "volume": float(k.volume) if k.volume else 0.0,
            })

        kline_data.sort(key=lambda x: x["date"])

        # Find indices for start_date and end_date
        start_idx = None
        end_idx = None
        for i, k in enumerate(kline_data):
            if k["date"] == start_date:
                start_idx = i
            if k["date"] == end_date:
                end_idx = i
        
        if start_idx is None or end_idx is None:
            raise HTTPException(
                status_code=404,
                detail=f"Could not find start_date {start_date} or end_date {end_date} in K-line data"
            )

        # Trim to ensure exactly extend_left_bars before start and extend_right_bars after end
        actual_start_idx = max(0, start_idx - extend_left_bars)
        actual_end_idx = min(len(kline_data) - 1, end_idx + extend_right_bars)
        
        trimmed_kline_data = kline_data[actual_start_idx:actual_end_idx + 1]

        closes = pd.Series([k["close"] for k in trimmed_kline_data])
        ma5 = closes.rolling(window=5).mean().tolist()
        ma10 = closes.rolling(window=10).mean().tolist()

        indicators = {
            "ma5": [float(x) if pd.notna(x) else None for x in ma5],
            "ma10": [float(x) if pd.notna(x) else None for x in ma10],
        }

        return {
            "kline_data": trimmed_kline_data,
            "indicators": indicators,
            "match_start": start_date,
            "match_end": end_date,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get DTW K-line for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve K-line: {str(e)}")


async def save_dtw_annotation_handler(
    request: DTWAnnotationRequest,
) -> DTWAnnotationResponse:
    """
    Save annotation for a DTW hit row back to CSV.
    
    Args:
        request: Annotation data.
    
    Returns:
        Success status.
    
    Raises:
        HTTPException: If file not found or write fails.
    """
    from pathlib import Path
    import pandas as pd

    csv_path = Path("outputs") / request.csv_file
    if not csv_path.exists():
        csv_path = Path(request.csv_file)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"CSV file not found: {request.csv_file}")

    try:
        df = pd.read_csv(csv_path)
        if request.row_index < 0 or request.row_index >= len(df):
            raise HTTPException(status_code=400, detail=f"Invalid row_index: {request.row_index}")

        df.at[request.row_index, "label"] = request.label
        if request.platform_start:
            df.at[request.row_index, "platform_start"] = request.platform_start
        if request.platform_end:
            df.at[request.row_index, "platform_end"] = request.platform_end
        if request.breakout_date:
            df.at[request.row_index, "breakout_date"] = request.breakout_date
        if request.notes:
            df.at[request.row_index, "notes"] = request.notes

        df.to_csv(csv_path, index=False)
        return DTWAnnotationResponse(success=True, message="Annotation saved")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save annotation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save annotation: {str(e)}")

