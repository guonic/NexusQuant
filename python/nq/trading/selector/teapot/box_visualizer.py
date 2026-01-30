"""
Box visualization for Teapot pattern recognition.

Generates K-line charts with box annotations for manual verification.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available, visualization will be disabled")

try:
    import kaleido
    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False
    logger.warning("Kaleido not available, image export may fail. Install with: pip install kaleido")


class BoxVisualizer:
    """
    Visualizer for box detection results.

    Generates K-line charts with box annotations and moving averages.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize visualizer.

        Args:
            output_dir: Output directory for charts (default: current directory).
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization. Install with: pip install plotly")
        
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/teapot/charts")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_box_id(
        self,
        ts_code: str,
        box_start_date: str,
        box_end_date: str,
    ) -> str:
        """
        Generate unique box ID.

        Format: {ts_code}.{start_date}.{end_date}

        Args:
            ts_code: Stock code.
            box_start_date: Box start date (YYYY-MM-DD).
            box_end_date: Box end date (YYYY-MM-DD).

        Returns:
            Unique box ID.
        """
        return f"{ts_code}.{box_start_date}.{box_end_date}"

    def calculate_moving_averages(
        self,
        df: pl.DataFrame,
        periods: list[int] = [5, 10, 30, 60],
    ) -> pl.DataFrame:
        """
        Calculate moving averages.

        Args:
            df: DataFrame with close prices, sorted by trade_date.
            periods: List of periods for moving averages.

        Returns:
            DataFrame with moving average columns added.
        """
        df = df.sort("trade_date")
        
        for period in periods:
            df = df.with_columns([
                pl.col("close")
                .rolling_mean(window_size=period)
                .over("ts_code")
                .alias(f"ma{period}"),
            ])
        
        return df

    def plot_box_chart(
        self,
        df: pl.DataFrame,
        ts_code: str,
        box_start_date: str,
        box_end_date: str,
        box_h: float,
        box_l: float,
        box_id: Optional[str] = None,
        context_days: int = 30,
    ) -> Optional[str]:
        """
        Plot K-line chart with box annotation.

        Args:
            df: DataFrame with K-line data (ts_code, trade_date, open, high, low, close, volume).
            ts_code: Stock code.
            box_start_date: Box start date (YYYY-MM-DD).
            box_end_date: Box end date (YYYY-MM-DD).
            box_h: Box upper bound.
            box_l: Box lower bound.
            box_id: Optional box ID. If None, will be generated.
            context_days: Number of days before and after box to show.

        Returns:
            Path to saved chart file, or None if failed.
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available")
            return None

        # Generate box ID if not provided
        if box_id is None:
            box_id = self.generate_box_id(ts_code, box_start_date, box_end_date)

        # Filter data for this stock
        stock_df = df.filter(pl.col("ts_code") == ts_code).sort("trade_date")

        if stock_df.is_empty():
            logger.warning(f"No data found for {ts_code}")
            return None

        # Parse dates
        try:
            box_start = datetime.strptime(box_start_date, "%Y-%m-%d")
            box_end = datetime.strptime(box_end_date, "%Y-%m-%d")
        except ValueError:
            logger.error(f"Invalid date format: {box_start_date} or {box_end_date}")
            return None

        # Extend date range for context
        chart_start = box_start - timedelta(days=context_days)
        chart_end = box_end + timedelta(days=context_days)

        # Filter data within chart range
        stock_df = stock_df.filter(
            (pl.col("trade_date") >= chart_start.strftime("%Y-%m-%d")) &
            (pl.col("trade_date") <= chart_end.strftime("%Y-%m-%d"))
        )

        if stock_df.is_empty():
            logger.warning(f"No data in range for {ts_code}")
            return None

        # Calculate moving averages
        stock_df = self.calculate_moving_averages(stock_df)

        # Convert to pandas for plotting
        df_plot = stock_df.to_pandas()
        df_plot["trade_date"] = pd.to_datetime(df_plot["trade_date"])

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"{ts_code} - Box ID: {box_id}", "Volume"),
        )

        # Plot candlestick
        fig.add_trace(
            go.Candlestick(
                x=df_plot["trade_date"],
                open=df_plot["open"],
                high=df_plot["high"],
                low=df_plot["low"],
                close=df_plot["close"],
                name="K-line",
            ),
            row=1,
            col=1,
        )

        # Plot moving averages
        ma_periods = [5, 10, 30, 60]
        ma_colors = ["blue", "orange", "green", "red"]
        for period, color in zip(ma_periods, ma_colors):
            if f"ma{period}" in df_plot.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_plot["trade_date"],
                        y=df_plot[f"ma{period}"],
                        mode="lines",
                        name=f"MA{period}",
                        line=dict(color=color, width=1),
                    ),
                    row=1,
                    col=1,
                )

        # Draw box boundaries
        box_start_dt = pd.to_datetime(box_start_date)
        box_end_dt = pd.to_datetime(box_end_date)

        # Box area (rectangle) - draw first so it's behind other elements
        fig.add_shape(
            type="rect",
            x0=box_start_dt,
            y0=box_l,
            x1=box_end_dt,
            y1=box_h,
            fillcolor="rgba(255, 165, 0, 0.2)",  # Orange with 20% opacity - more visible
            line=dict(color="red", width=2),  # Thicker border
            layer="below",  # Draw below other elements
            row=1,
            col=1,
        )

        # Box upper bound (horizontal line) - draw on top
        fig.add_shape(
            type="line",
            x0=box_start_dt,
            y0=box_h,
            x1=box_end_dt,
            y1=box_h,
            line=dict(color="red", width=3, dash="solid"),  # Solid line, thicker
            layer="above",  # Draw above rectangle
            row=1,
            col=1,
        )

        # Box lower bound (horizontal line) - draw on top
        fig.add_shape(
            type="line",
            x0=box_start_dt,
            y0=box_l,
            x1=box_end_dt,
            y1=box_l,
            line=dict(color="green", width=3, dash="solid"),  # Solid line, thicker
            layer="above",  # Draw above rectangle
            row=1,
            col=1,
        )

        # Add annotation for box
        box_center = (box_h + box_l) / 2
        fig.add_annotation(
            x=box_start_dt + (box_end_dt - box_start_dt) / 2,
            y=box_center,
            text=f"Box: {box_l:.2f} - {box_h:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="red",
            bgcolor="rgba(255, 255, 255, 0.8)",
            row=1,
            col=1,
        )

        # Plot volume
        fig.add_trace(
            go.Bar(
                x=df_plot["trade_date"],
                y=df_plot["volume"],
                name="Volume",
                marker_color="lightblue",
            ),
            row=2,
            col=1,
        )

        # Update layout
        fig.update_layout(
            title=f"{ts_code} - Box Detection Chart (Box ID: {box_id})",
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            hovermode="x unified",
        )

        # Update x-axis
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        # Save chart as HTML
        chart_file = self.output_dir / f"{box_id}.html"
        fig.write_html(str(chart_file))
        
        logger.info(f"Saved chart: {chart_file}")
        return str(chart_file)

    def batch_plot_boxes(
        self,
        boxes_df: pl.DataFrame,
        kline_df: pl.DataFrame,
        context_days: int = 30,
        max_boxes: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Batch plot boxes from DataFrame.

        Args:
            boxes_df: DataFrame with box information. Must contain columns:
                ts_code, box_start_date, box_end_date, box_h, box_l, box_id (optional)
            kline_df: DataFrame with K-line data.
            context_days: Number of days before and after box to show.
            max_boxes: Maximum number of boxes to plot (None for all).

        Returns:
            DataFrame with box_id and chart_path columns added.
        """
        if boxes_df.is_empty():
            logger.warning("No boxes to plot")
            return pl.DataFrame()

        # Limit number of boxes if specified
        if max_boxes:
            boxes_df = boxes_df.head(max_boxes)

        results = []
        
        for row in boxes_df.iter_rows(named=True):
            ts_code = row["ts_code"]
            box_start_date = row.get("box_start_date", row.get("start_date"))
            box_end_date = row.get("box_end_date", row.get("end_date"))
            box_h = row["box_h"]
            box_l = row["box_l"]
            box_id = row.get("box_id")

            try:
                chart_path = self.plot_box_chart(
                    df=kline_df,
                    ts_code=ts_code,
                    box_start_date=str(box_start_date),
                    box_end_date=str(box_end_date),
                    box_h=float(box_h),
                    box_l=float(box_l),
                    box_id=box_id,
                    context_days=context_days,
                )

                # Calculate box duration days
                try:
                    from datetime import datetime
                    start_dt = datetime.strptime(str(box_start_date), "%Y-%m-%d")
                    end_dt = datetime.strptime(str(box_end_date), "%Y-%m-%d")
                    duration_days = (end_dt - start_dt).days + 1  # +1 to include both start and end days
                except Exception:
                    duration_days = 0
                
                results.append({
                    "box_id": box_id or self.generate_box_id(ts_code, str(box_start_date), str(box_end_date)),
                    "ts_code": ts_code,
                    "box_start_date": box_start_date,
                    "box_end_date": box_end_date,
                    "box_duration_days": duration_days,
                    "chart_path": chart_path or "",
                    "status": "success" if chart_path else "failed",
                })
            except Exception as e:
                logger.error(f"Failed to plot box for {ts_code}: {e}")
                # Calculate box duration days even for errors
                try:
                    from datetime import datetime
                    start_dt = datetime.strptime(str(box_start_date), "%Y-%m-%d")
                    end_dt = datetime.strptime(str(box_end_date), "%Y-%m-%d")
                    duration_days = (end_dt - start_dt).days + 1
                except Exception:
                    duration_days = 0
                
                results.append({
                    "box_id": box_id or self.generate_box_id(ts_code, str(box_start_date), str(box_end_date)),
                    "ts_code": ts_code,
                    "box_start_date": box_start_date,
                    "box_end_date": box_end_date,
                    "box_duration_days": duration_days,
                    "chart_path": "",
                    "status": "error",
                })

        return pl.DataFrame(results)
    
    def create_summary_html(
        self,
        results_df: pl.DataFrame,
        output_file: Optional[Path] = None,
        title: str = "Box Detection Charts Summary",
    ) -> str:
        """
        Create a summary HTML file with all chart images.
        
        Args:
            results_df: DataFrame with chart results (from batch_plot_boxes).
            output_file: Output HTML file path (default: summary.html in output_dir).
            title: Title for the summary page.
            
        Returns:
            Path to saved summary HTML file.
        """
        if results_df.is_empty():
            logger.warning("No results to create summary")
            return ""
        
        if output_file is None:
            output_file = self.output_dir / "summary.html"
        
        # Filter successful charts
        successful = results_df.filter(pl.col("status") == "success")
        
        if successful.is_empty():
            logger.warning("No successful charts to include in summary")
            return ""
        
        # Calculate box duration days if not present
        if "box_duration_days" not in successful.columns:
            successful = successful.with_columns([
                (
                    pl.col("box_end_date").cast(pl.Date) - pl.col("box_start_date").cast(pl.Date)
                ).dt.total_days().alias("box_duration_days")
            ])
        
        # Sort by box_duration_days (descending - longest boxes first)
        successful = successful.sort("box_duration_days", descending=True)
        
        # Generate HTML content
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #333;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .stats {{
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chart-list {{
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .chart-item {{
            padding: 15px;
            border-bottom: 1px solid #eee;
            transition: background-color 0.2s;
        }}
        .chart-item:hover {{
            background-color: #f8f9fa;
        }}
        .chart-item:last-child {{
            border-bottom: none;
        }}
        .chart-link {{
            display: block;
            color: #007bff;
            text-decoration: none;
            font-size: 16px;
            font-weight: 500;
            margin-bottom: 5px;
        }}
        .chart-link:hover {{
            color: #0056b3;
            text-decoration: underline;
        }}
        .chart-info {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
        .chart-info strong {{
            color: #333;
        }}
        .chart-info span {{
            margin-right: 15px;
        }}
        .search-box {{
            margin-bottom: 20px;
            padding: 10px;
            width: 100%;
            max-width: 400px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="stats">
        <h2>Statistics</h2>
        <p><strong>Total Charts:</strong> {len(successful)}</p>
        <p><strong>Failed:</strong> {len(results_df) - len(successful)}</p>
    </div>
    
    <input type="text" id="searchBox" class="search-box" placeholder="Search by stock code or box ID...">
    
    <div class="chart-list" id="chartList">
"""
        
        # Add chart items as links, sorted by box_duration_days
        for row in successful.iter_rows(named=True):
            box_id = row["box_id"]
            ts_code = row["ts_code"]
            box_start = row.get("box_start_date", "")
            box_end = row.get("box_end_date", "")
            box_duration = row.get("box_duration_days", 0)
            chart_path = row["chart_path"]
            
            # Get relative path for HTML file
            if chart_path:
                chart_file = Path(chart_path)
                summary_html_dir = Path(output_file).parent.resolve()
                
                # Calculate relative path from summary HTML location to chart file
                try:
                    chart_file_resolved = chart_file.resolve()
                    # Get relative path from summary_html_dir to chart_file
                    rel_path = chart_file_resolved.relative_to(summary_html_dir)
                    # Convert to forward slashes for web compatibility
                    rel_path = str(rel_path).replace("\\", "/")
                except (ValueError, RuntimeError) as e:
                    # If paths are on different drives or can't compute relative path
                    # Try to extract just the relative part from chart_path
                    # Chart files are typically in a "charts" subdirectory
                    if "charts" in str(chart_file):
                        # Extract the part after "charts/"
                        chart_name = chart_file.name
                        rel_path = f"charts/{chart_name}"
                    else:
                        # Fallback: use filename only
                        logger.warning(f"Could not compute relative path for {chart_path}: {e}, using filename only")
                        rel_path = chart_file.name
                
                # Display as link only (no thumbnail)
                html_content += f"""
        <div class="chart-item" data-ts-code="{ts_code}" data-box-id="{box_id}" data-duration="{box_duration}">
            <a href="{rel_path}" target="_blank" class="chart-link">{box_id}</a>
            <div class="chart-info">
                <span><strong>Stock:</strong> {ts_code}</span>
                <span><strong>Period:</strong> {box_start} to {box_end}</span>
                <span><strong>Duration:</strong> {int(box_duration)} days</span>
            </div>
        </div>
"""
        
        html_content += """
    </div>
    
    <script>
        // Search functionality
        document.getElementById("searchBox").addEventListener("input", function(e) {
            var searchTerm = e.target.value.toLowerCase();
            var items = document.getElementsByClassName("chart-item");
            
            for (var i = 0; i < items.length; i++) {
                var item = items[i];
                var tsCode = item.getAttribute("data-ts-code").toLowerCase();
                var boxId = item.getAttribute("data-box-id").toLowerCase();
                
                if (tsCode.includes(searchTerm) || boxId.includes(searchTerm)) {
                    item.style.display = "block";
                } else {
                    item.style.display = "none";
                }
            }
        });
        
        // Sort functionality (already sorted by server, but can add client-side sorting if needed)
        // Charts are already sorted by box_duration_days (descending) from the server
    </script>
</body>
</html>
"""
        
        # Write HTML file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logger.info(f"Created summary HTML: {output_file}")
        return str(output_file)