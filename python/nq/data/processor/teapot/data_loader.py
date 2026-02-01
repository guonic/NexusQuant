"""
Data loader for Teapot pattern recognition.

Provides data loading from PostgreSQL and Parquet cache.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import polars as pl

from nq.config import DatabaseConfig
from nq.data.processor.teapot.cache_manager import CacheManager
from nq.data.processor.teapot.exceptions import CacheIncompleteError, CacheNotFoundError
from nq.repo.kline_repo import StockKlineDayRepo
from nq.repo.stock_repo import StockBasicRepo

logger = logging.getLogger(__name__)


class TeapotDataLoader:
    """
    Data loader for Teapot pattern recognition.

    Supports loading from PostgreSQL (production) or Parquet cache (backtesting/development).
    """

    def __init__(
        self,
        db_config: DatabaseConfig,
        schema: str = "quant",
        use_cache: bool = False,
        cache_dir: Optional[Path] = None,
        strict_cache: bool = False,
    ):
        """
        Initialize data loader.

        Args:
            db_config: Database configuration.
            schema: Database schema name.
            use_cache: Whether to use Parquet cache.
            cache_dir: Cache directory path (if None, uses default).
            strict_cache: Strict cache mode for backtesting. If True, raises exceptions when cache is missing or incomplete.
        """
        self.db_config = db_config
        self.schema = schema
        self.use_cache = use_cache
        self.strict_cache = strict_cache

        if strict_cache:
            self.use_cache = True  # Force enable cache in strict mode

        if self.use_cache:
            cache_path = cache_dir or Path("storage/teapot_cache")
            self.cache_manager = CacheManager(
                cache_dir=str(cache_path),
                strict_mode=strict_cache,
            )
        else:
            self.cache_manager = None

        self.kline_repo = StockKlineDayRepo(db_config, schema)
        self.stock_basic_repo = StockBasicRepo(db_config, schema)

    def load_daily_data(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """
        Load daily K-line data.

        Strategy:
        - If strict_cache=True: Force load from cache, raise exception on failure.
        - If use_cache=True: Try cache first, fallback to PostgreSQL.
        - Otherwise: Load directly from PostgreSQL.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            symbols: Optional list of stock codes to filter.

        Returns:
            Polars DataFrame with columns: ts_code, trade_date, open, high, low, close, volume, amount.

        Raises:
            CacheNotFoundError: If cache not found (strict_cache=True).
            CacheIncompleteError: If cache incomplete (strict_cache=True).
        """
        if self.strict_cache:
            # Strict mode: Force use cache
            return self.cache_manager.load_from_cache(
                start_date=start_date,
                end_date=end_date,
                symbols=symbols,
            )
        else:
            # Normal mode: Try cache first, fallback to PostgreSQL
            if self.use_cache and self.cache_manager:
                cached_data = self.cache_manager.load_from_cache(
                    start_date=start_date,
                    end_date=end_date,
                    symbols=symbols,
                )
                if cached_data is not None and not cached_data.is_empty():
                    return cached_data

            # Load from PostgreSQL
            return self.load_from_postgresql(
                start_date=start_date,
                end_date=end_date,
                symbols=symbols,
            )

    def load_from_postgresql(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """
        Load data directly from PostgreSQL.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            symbols: Optional list of stock codes to filter.

        Returns:
            Polars DataFrame with columns: ts_code, trade_date, open, high, low, close, volume, amount.
        """
        logger.info(
            f"Loading data from PostgreSQL: {start_date} to {end_date}"
        )

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Load data for all symbols or specific symbols
        if symbols:
            all_data = []
            for ts_code in symbols:
                klines = self.kline_repo.get_by_ts_code(
                    ts_code=ts_code,
                    start_time=start_dt,
                    end_time=end_dt,
                )
                for kline in klines:
                    all_data.append(
                        {
                            "ts_code": kline.ts_code,
                            "trade_date": kline.trade_date.strftime("%Y-%m-%d")
                            if isinstance(kline.trade_date, datetime)
                            else str(kline.trade_date),
                            "open": float(kline.open) if kline.open else None,
                            "high": float(kline.high) if kline.high else None,
                            "low": float(kline.low) if kline.low else None,
                            "close": float(kline.close) if kline.close else None,
                            "volume": int(kline.volume) if kline.volume else 0,
                            "amount": float(kline.amount) if kline.amount else None,
                        }
                    )
        else:
            # Load all stocks (need to query all stocks first)
            logger.warning(
                "Loading all stocks - this may be slow. Consider specifying symbols."
            )
            all_data = []

            # Get all stock codes from stock_basic table
            try:
                all_stocks = self.stock_basic_repo.get_all()
                stock_codes = [stock.ts_code for stock in all_stocks if stock.ts_code]
                logger.info(f"Found {len(stock_codes)} stocks in stock_basic table")
                
                if not stock_codes:
                    logger.warning("No stocks found in stock_basic table")
                    return pl.DataFrame()
                
                # Load data for all stocks using direct SQL query (more efficient)
                try:
                    from sqlalchemy import text
                    engine = self.kline_repo._get_engine()
                    table_name = self.kline_repo._get_full_table_name()
                    
                    # Build SQL query
                    query = f"""
                    SELECT ts_code, trade_date, open, high, low, close, volume, amount
                    FROM {table_name}
                    WHERE trade_date >= :start_date AND trade_date <= :end_date
                    ORDER BY ts_code, trade_date
                    """
                    
                    logger.info("Executing SQL query for all stocks...")
                    with engine.connect() as conn:
                        result = conn.execute(
                            text(query),
                            {
                                "start_date": start_dt.date(),
                                "end_date": end_dt.date(),
                            },
                        )
                        
                        for row in result:
                            all_data.append(
                                {
                                    "ts_code": row.ts_code,
                                    "trade_date": (
                                        row.trade_date.strftime("%Y-%m-%d")
                                        if isinstance(row.trade_date, datetime)
                                        else str(row.trade_date)
                                    ),
                                    "open": float(row.open) if row.open else None,
                                    "high": float(row.high) if row.high else None,
                                    "low": float(row.low) if row.low else None,
                                    "close": float(row.close) if row.close else None,
                                    "volume": int(row.volume) if row.volume else 0,
                                    "amount": float(row.amount) if row.amount else None,
                                }
                            )
                    
                    logger.info(f"Loaded {len(all_data)} records from SQL query")
                except Exception as e:
                    logger.warning(
                        f"Failed to load data using SQL query, falling back to batch processing: {e}"
                    )
                    # Fallback to batch processing
                    batch_size = 100
                    for i in range(0, len(stock_codes), batch_size):
                        batch_codes = stock_codes[i : i + batch_size]
                        logger.info(
                            f"Loading batch {i // batch_size + 1}/{(len(stock_codes) + batch_size - 1) // batch_size}: "
                            f"{len(batch_codes)} stocks"
                        )
                        
                        for ts_code in batch_codes:
                            try:
                                klines = self.kline_repo.get_by_ts_code(
                                    ts_code=ts_code,
                                    start_time=start_dt,
                                    end_time=end_dt,
                                )
                                for kline in klines:
                                    all_data.append(
                                        {
                                            "ts_code": kline.ts_code,
                                            "trade_date": kline.trade_date.strftime("%Y-%m-%d")
                                            if isinstance(kline.trade_date, datetime)
                                            else str(kline.trade_date),
                                            "open": float(kline.open) if kline.open else None,
                                            "high": float(kline.high) if kline.high else None,
                                            "low": float(kline.low) if kline.low else None,
                                            "close": float(kline.close) if kline.close else None,
                                            "volume": int(kline.volume) if kline.volume else 0,
                                            "amount": float(kline.amount) if kline.amount else None,
                                        }
                                    )
                            except Exception as e2:
                                logger.warning(
                                    f"Failed to load data for {ts_code}: {e2}"
                                )
                                continue
                            
            except Exception as e:
                logger.error(f"Failed to load stock list: {e}")
                return pl.DataFrame()

        if not all_data:
            logger.warning("No data found in PostgreSQL")
            return pl.DataFrame()

        df = pl.DataFrame(all_data)
        df = self.validate_data(df)

        logger.info(f"Loaded {len(df)} records from PostgreSQL")
        return df.sort(["ts_code", "trade_date"])

    def validate_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Validate data quality.

        Args:
            df: Input DataFrame.

        Returns:
            Validated DataFrame.
        """
        # Remove rows with missing critical data
        df = df.filter(
            pl.col("ts_code").is_not_null()
            & pl.col("trade_date").is_not_null()
            & pl.col("close").is_not_null()
            & (pl.col("close") > 0)
        )

        # Fill missing amount with close * volume
        df = df.with_columns(
            pl.when(pl.col("amount").is_null() | (pl.col("amount") == 0))
            .then(pl.col("close") * pl.col("volume"))
            .otherwise(pl.col("amount"))
            .alias("amount")
        )

        return df
