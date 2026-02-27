"""
FastAPI routes for Eidos API.
"""

from datetime import date, datetime
from typing import List, Optional

from fastapi import APIRouter, Query

from nq.api.rest.eidos import handlers
from nq.api.rest.eidos.schemas import (
    ExperimentResponse,
    LedgerEntryResponse,
    TradeResponse,
    PerformanceMetricsResponse,
    TradeStatsResponse,
    BacktestReportResponse,
    DTWHitsPageResponse,
    DTWAnnotationRequest,
    DTWAnnotationResponse,
)

router = APIRouter(prefix="/api/v1", tags=["eidos"])


@router.get("/experiments", response_model=List[ExperimentResponse])
async def get_experiments():
    """
    Get all experiments.
    
    Returns:
        List of all experiments.
    """
    return await handlers.get_experiments_handler()


@router.get("/experiments/{exp_id}", response_model=ExperimentResponse)
async def get_experiment(exp_id: str):
    """
    Get a single experiment by ID.
    
    Args:
        exp_id: Experiment ID (8-character hex string).
    
    Returns:
        Experiment data.
    """
    return await handlers.get_experiment_handler(exp_id)


@router.get("/experiments/{exp_id}/ledger", response_model=List[LedgerEntryResponse])
async def get_ledger(
    exp_id: str,
    start_date: Optional[date] = Query(None, description="Start date filter"),
    end_date: Optional[date] = Query(None, description="End date filter"),
):
    """
    Get ledger entries (daily NAV) for an experiment.
    
    Args:
        exp_id: Experiment ID.
        start_date: Optional start date filter.
        end_date: Optional end date filter.
    
    Returns:
        List of ledger entries.
    """
    return await handlers.get_ledger_handler(exp_id, start_date=start_date, end_date=end_date)


@router.get("/experiments/{exp_id}/trades", response_model=List[TradeResponse])
async def get_trades(
    exp_id: str,
    symbol: Optional[str] = Query(None, description="Symbol filter"),
    start_date: Optional[datetime] = Query(None, description="Start date/time filter"),
    end_date: Optional[datetime] = Query(None, description="End date/time filter"),
):
    """
    Get trades for an experiment.
    
    Args:
        exp_id: Experiment ID.
        symbol: Optional symbol filter.
        start_date: Optional start date/time filter.
        end_date: Optional end date/time filter.
    
    Returns:
        List of trades.
    """
    return await handlers.get_trades_handler(
        exp_id, symbol=symbol, start_date=start_date, end_date=end_date
    )


@router.get("/experiments/{exp_id}/metrics", response_model=PerformanceMetricsResponse)
async def get_performance_metrics(exp_id: str):
    """
    Get performance metrics for an experiment.
    
    Calculates:
    - Total return
    - Maximum drawdown
    - Final NAV
    - Trading days
    - Sharpe ratio (if available)
    - Annual return (if available)
    
    Args:
        exp_id: Experiment ID.
    
    Returns:
        Performance metrics.
    """
    return await handlers.get_performance_metrics_handler(exp_id)


@router.get("/experiments/{exp_id}/trade-stats", response_model=TradeStatsResponse)
async def get_trade_stats(exp_id: str):
    """
    Get trade statistics for an experiment.
    
    Calculates:
    - Total trades
    - Buy/Sell counts
    - Win rate
    - Average holding days
    
    Args:
        exp_id: Experiment ID.
    
    Returns:
        Trade statistics.
    """
    return await handlers.get_trade_stats_handler(exp_id)


@router.get("/experiments/{exp_id}/kline/{symbol}")
async def get_stock_kline(
    exp_id: str,
    symbol: str,
    start_date: Optional[date] = Query(None, description="Start date filter"),
    end_date: Optional[date] = Query(None, description="End date filter"),
    indicators: Optional[str] = Query(None, description="Comma-separated list of indicators to calculate. Supported: ma5,ma10,ma20,ma30,ma60,ma120,ema,wma,rsi,kdj,cci,wr,obv,macd,dmi,bollinger,envelope,atr,bbw,vwap"),
):
    """
    Get K-line (OHLCV) data for a stock symbol within an experiment's date range.
    Optionally calculate technical indicators on the backend.
    
    Args:
        exp_id: Experiment ID.
        symbol: Stock symbol (e.g., "000001.SZ").
        start_date: Optional start date filter.
        end_date: Optional end date filter.
        indicators: Optional comma-separated list of indicators to calculate.
    
    Returns:
        Dictionary with 'kline_data' (list of K-line points) and 'indicators' (calculated indicators).
    """
    # Parse indicators parameter
    indicators_dict = None
    if indicators:
        indicator_list = [i.strip() for i in indicators.split(",")]
        indicators_dict = {
            # Trend indicators
            "ma5": "ma5" in indicator_list,
            "ma10": "ma10" in indicator_list,
            "ma20": "ma20" in indicator_list,
            "ma30": "ma30" in indicator_list,
            "ma60": "ma60" in indicator_list,
            "ma120": "ma120" in indicator_list,
            "ema": "ema" in indicator_list,
            "wma": "wma" in indicator_list,
            # Oscillator indicators
            "rsi": "rsi" in indicator_list,
            "kdj": "kdj" in indicator_list,
            "cci": "cci" in indicator_list,
            "wr": "wr" in indicator_list,
            "obv": "obv" in indicator_list,
            # Trend + Oscillator indicators
            "macd": "macd" in indicator_list,
            "dmi": "dmi" in indicator_list,
            # Channel indicators
            "bollinger": "bollinger" in indicator_list,
            "envelope": "envelope" in indicator_list,
            # Volatility indicators
            "atr": "atr" in indicator_list,
            "bbw": "bbw" in indicator_list,
            # Volume indicators
            "vwap": "vwap" in indicator_list,
        }
    
    return await handlers.get_stock_kline_handler(
        exp_id, symbol, start_date=start_date, end_date=end_date, indicators=indicators_dict
    )


@router.get("/experiments/{exp_id}/report", response_model=BacktestReportResponse)
async def get_backtest_report(
    exp_id: str,
    format: Optional[str] = Query("json", description="Output format (json, console, html, markdown)"),
    categories: Optional[str] = Query(None, description="Comma-separated metric categories (portfolio, trading, turnover, risk, model)"),
    metrics: Optional[str] = Query(None, description="Comma-separated metric names"),
):
    """
    Get complete backtest report for an experiment.
    
    Generates a comprehensive report with metrics organized by category:
    - Portfolio metrics (returns, Sharpe ratio, drawdown, etc.)
    - Trading statistics (win rate, profit factor, etc.)
    - Turnover statistics (turnover rate, position count, etc.)
    - Risk metrics (if available)
    - Model performance metrics (if available)
    
    Args:
        exp_id: Experiment ID.
        format: Output format (currently only 'json' is supported for API).
        categories: Optional comma-separated list of metric categories to include.
        metrics: Optional comma-separated list of specific metric names to include.
    
    Returns:
        Complete backtest report with all metrics.
    """
    return await handlers.get_backtest_report_handler(
        exp_id, format=format, categories=categories, metrics=metrics
    )


# DTW Labeling routes
@router.get("/dtw/labeling/csv/{filename:path}", response_model=DTWHitsPageResponse)
async def get_dtw_hits(
    filename: str,
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(50, ge=1, le=500, description="Page size"),
):
    """
    Get paginated DTW hits from CSV file.
    
    Args:
        filename: CSV filename (e.g., "dtw_hits_600487.SH_2024-01-01_2024-12-31.csv").
        page: Page number (1-based).
        page_size: Number of rows per page.
    
    Returns:
        Paginated DTW hits.
    """
    return await handlers.get_dtw_hits_handler(filename, page=page, page_size=page_size)


@router.get("/dtw/labeling/kline/{symbol}")
async def get_dtw_kline(
    symbol: str,
    start_date: str = Query(..., description="Start date YYYY-MM-DD"),
    end_date: str = Query(..., description="End date YYYY-MM-DD"),
    extend_left_bars: int = Query(5, description="Number of K-line bars to extend left"),
    extend_right_bars: int = Query(5, description="Number of K-line bars to extend right"),
):
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
    """
    return await handlers.get_dtw_kline_handler(
        symbol, start_date, end_date, extend_left_bars, extend_right_bars
    )


@router.post("/dtw/labeling/annotate", response_model=DTWAnnotationResponse)
async def save_dtw_annotation(request: DTWAnnotationRequest):
    """
    Save annotation for a DTW hit row.
    
    Args:
        request: Annotation data (CSV file, row index, label, dates, notes).
    
    Returns:
        Success status.
    """
    return await handlers.save_dtw_annotation_handler(request)

