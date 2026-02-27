"""
Pydantic schemas for Eidos API requests and responses.
"""

from __future__ import annotations

from datetime import date as Date, datetime as DateTime
from typing import Any, Dict, List, Optional

from pydantic import Field, BaseModel

from nq.models.eidos import (
    Experiment,
    LedgerEntry,
    Trade,
    ModelOutput,
)


# Request schemas
class ExperimentCreateRequest(BaseModel):
    """Request schema for creating a new experiment."""

    name: str = Field(..., description="Experiment name")
    model_type: Optional[str] = Field(None, description="Model type")
    engine_type: Optional[str] = Field(None, description="Engine type")
    start_date: Date = Field(..., description="Start date")
    end_date: Date = Field(..., description="End date")
    config: Dict[str, Any] = Field(default_factory=dict, description="Configuration")
    metrics_summary: Optional[Dict[str, Any]] = Field(None, description="Metrics summary")


# Response schemas
class ExperimentResponse(Experiment):
    """Response schema for experiment data."""

    created_at: Optional[DateTime] = Field(None, description="Creation timestamp")
    updated_at: Optional[DateTime] = Field(None, description="Update timestamp")


class LedgerEntryResponse(LedgerEntry):
    """Response schema for ledger entry."""

    pass


class TradeResponse(Trade):
    """Response schema for trade data."""

    pass


class PerformanceMetricsResponse(BaseModel):
    """Response schema for performance metrics."""

    total_return: float = Field(..., description="Total return")
    annualized_return: Optional[float] = Field(None, description="Annualized return")
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    final_nav: float = Field(..., description="Final NAV")
    initial_cash: float = Field(..., description="Initial cash")
    total_trades: int = Field(..., description="Total number of trades")
    win_rate: Optional[float] = Field(None, description="Win rate")
    profit_factor: Optional[float] = Field(None, description="Profit factor")


class TradeStatsResponse(BaseModel):
    """Response schema for trade statistics."""

    total_trades: int = Field(..., description="Total number of trades")
    buy_trades: int = Field(..., description="Number of buy trades")
    sell_trades: int = Field(..., description="Number of sell trades")
    win_trades: int = Field(..., description="Number of winning trades")
    loss_trades: int = Field(..., description="Number of losing trades")
    win_rate: float = Field(..., description="Win rate")
    avg_profit: Optional[float] = Field(None, description="Average profit")
    avg_loss: Optional[float] = Field(None, description="Average loss")
    profit_factor: Optional[float] = Field(None, description="Profit factor")
    max_consecutive_wins: int = Field(..., description="Maximum consecutive wins")
    max_consecutive_losses: int = Field(..., description="Maximum consecutive losses")


class BacktestReportResponse(BaseModel):
    """Response schema for backtest report."""

    exp_id: str = Field(..., description="Experiment ID")
    report: Dict[str, Any] = Field(..., description="Report data")
    format: str = Field(default="json", description="Report format")


# DTW Labeling schemas
class DTWHitRow(BaseModel):
    """Single row from DTW hits CSV."""

    template: str = Field(..., description="Template name")
    ts_code: str = Field(..., description="Stock code")
    start_date: str = Field(..., description="Start date YYYY-MM-DD")
    end_date: str = Field(..., description="End date YYYY-MM-DD")
    score: float = Field(..., description="DTW score")
    hit_count: Optional[int] = Field(1, description="Number of hits merged")
    start_index: Optional[int] = Field(None, description="Start index")
    end_index: Optional[int] = Field(None, description="End index")
    # Annotation fields (optional, added after labeling)
    label: Optional[str] = Field(None, description="Label: positive/negative")
    platform_start: Optional[str] = Field(None, description="Platform start date YYYY-MM-DD")
    platform_end: Optional[str] = Field(None, description="Platform end date YYYY-MM-DD")
    breakout_date: Optional[str] = Field(None, description="Breakout date YYYY-MM-DD")
    notes: Optional[str] = Field(None, description="Notes")


class DTWHitsPageResponse(BaseModel):
    """Paginated DTW hits response."""

    hits: List[DTWHitRow] = Field(..., description="List of hits")
    total: int = Field(..., description="Total number of hits")
    page: int = Field(..., description="Current page (1-based)")
    page_size: int = Field(..., description="Page size")
    total_pages: int = Field(..., description="Total number of pages")


class DTWAnnotationRequest(BaseModel):
    """Request to save annotation for a hit."""

    csv_file: str = Field(..., description="CSV filename")
    row_index: int = Field(..., description="Row index (0-based)")
    label: str = Field(..., description="Label: positive or negative")
    platform_start: Optional[str] = Field(None, description="Platform start date YYYY-MM-DD")
    platform_end: Optional[str] = Field(None, description="Platform end date YYYY-MM-DD")
    breakout_date: Optional[str] = Field(None, description="Breakout date YYYY-MM-DD")
    notes: Optional[str] = Field(None, description="Notes")


class DTWAnnotationResponse(BaseModel):
    """Response after saving annotation."""

    success: bool = Field(..., description="Success status")
    message: str = Field(..., description="Message")
