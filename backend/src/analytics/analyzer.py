
"""
Analytics Analyzer - Hybrid Engine
Combines deterministic analytics (100% accurate historical) with ML predictions (future forecasting)

Features:
- Deterministic: Time series, breakdowns, KPIs (always available)
- ML Optional: Forecasting, anomaly detection, trend prediction
- AUTO SCHEMA DETECTION: If schema is None, automatically detects it!
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Literal, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.analytics.schema import SemanticSchema, MetricDefinition, DimensionDefinition
from src.data.smart_detector import SmartSchemaDetector
from src.data.normalizer import DataNormalizer


class AnalyticsAnalyzer:
    """
    Hybrid Analytics Engine
    
    Deterministic Methods (Core):
    - time_series_analysis() - Historical trends
    - dimension_breakdown() - Category analysis
    - kpi_analysis() - Current vs previous period
    
    ML Methods (Optional):
    - forecast_metric() - Future predictions
    - detect_anomalies() - Outlier detection
    - analyze_seasonality() - Pattern detection
    """
    
    def __init__(self, enable_ml: bool = True):
        """
        Initialize analyzer
        
        Args:
            enable_ml: Enable ML features (requires prophet library)
        """
        self.cache = {}
        self.enable_ml = enable_ml
        self.models = {}
        
        # Initialize smart detector and normalizer
        self.schema_detector = SmartSchemaDetector()
        self.normalizer = DataNormalizer()
        
        # Check if ML libraries are available
        if self.enable_ml:
            try:
                from prophet import Prophet
                self.ml_available = True
                print("✅ ML forecasting enabled (Prophet available)")
            except ImportError:
                self.ml_available = False
                self.enable_ml = False
                print("⚠️ ML forecasting disabled (Prophet not installed)")
                print("   Install with: pip install prophet")
    
    async def _ensure_schema(
        self, 
        df: pd.DataFrame, 
        schema: Optional[SemanticSchema],
        dataset_id: str = "auto_detected"
    ) -> SemanticSchema:
        """
        Ensure we have a valid schema - auto-detect if None
        
        Args:
            df: DataFrame to analyze
            schema: Existing schema or None
            dataset_id: ID for auto-detected schema
        
        Returns:
            Valid SemanticSchema
        """
        if schema is not None:
            return schema
        
        print(f"📊 No schema provided - auto-detecting schema...")
        
        # Clean the dataframe first
        df_cleaned = self.normalizer.clean_dataframe(df)
        
        # Auto-detect schema using smart detector
        detected_schema = await self.schema_detector.detect_schema(
            df=df_cleaned,
            dataset_id=dataset_id,
            dataset_name=f"Auto-detected from {dataset_id}"
        )
        
        print(f"✅ Schema auto-detected successfully!")
        print(f"   Time column: {detected_schema.time_column.column_name}")
        print(f"   Metrics: {[m.name for m in detected_schema.metrics]}")
        print(f"   Dimensions: {[d.name for d in detected_schema.dimensions]}")
        
        return detected_schema
    
    # ============================================
    # DETERMINISTIC ANALYTICS (Core - Always Available)
    # ============================================
    
    async def time_series_analysis(
        self,
        df: pd.DataFrame,
        schema: Optional[SemanticSchema] = None,
        metric_names: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        granularity: Optional[str] = None,
        dataset_id: str = "auto_detected"
    ) -> Dict[str, Any]:
        """
        Analyze metrics over time (100% accurate historical data)
        
        Args:
            df: Source DataFrame
            schema: Semantic schema (auto-detects if None)
            metric_names: List of metric names to analyze (uses all if None)
            start_date: Optional start date filter
            end_date: Optional end date filter
            granularity: Override time granularity (day, week, month)
            dataset_id: ID for auto-detection
        
        Returns:
            Time series analysis results
        """
        # Auto-detect schema if not provided
        schema = await self._ensure_schema(df, schema, dataset_id)
        
        # Use all metrics if none specified
        if metric_names is None:
            metric_names = [m.name for m in schema.metrics]
        
        # 1. Validate metrics
        metrics = self._get_metrics_by_names(schema, metric_names)
        if not metrics:
            raise ValueError(f"No valid metrics found for: {metric_names}")
        
        # 2. Get time column
        time_col = schema.time_column.column_name
        if time_col not in df.columns:
            raise ValueError(f"Time column '{time_col}' not found in dataset")
        
        # 3. Filter by date range
        df_filtered = df.copy()
        df_filtered[time_col] = pd.to_datetime(df_filtered[time_col])
        
        if start_date:
            df_filtered = df_filtered[df_filtered[time_col] >= start_date]
        if end_date:
            df_filtered = df_filtered[df_filtered[time_col] <= end_date]
        
        # 4. Set granularity
        gran = granularity or schema.time_column.granularity
        
        # 5. Resample/group by time
        df_grouped = self._group_by_time(df_filtered, time_col, gran)
        
        # 6. Aggregate metrics
        result_data = []
        for date, group in df_grouped:
            row = {time_col: date.isoformat() if isinstance(date, datetime) else str(date)}
            for metric in metrics:
                value = self._aggregate_metric(group, metric)
                row[metric.name] = round(value, 2) if value else 0
            result_data.append(row)
        
        # 7. Calculate summary statistics
        summary = {}
        for metric in metrics:
            values = [row[metric.name] for row in result_data if row[metric.name] is not None]
            summary[metric.name] = {
                "total": round(sum(values), 2) if metric.aggregation == "sum" else None,
                "avg": round(np.mean(values), 2) if values else 0,
                "min": round(min(values), 2) if values else 0,
                "max": round(max(values), 2) if values else 0,
                "trend": self._calculate_trend(values),
                "direction": metric.direction,
                "unit": metric.unit,
                "data_points": len(values)
            }
        
        return {
            "analysis_type": "time_series",
            "method": "deterministic",
            "accuracy": "100% (historical data)",
            "time_column": time_col,
            "granularity": gran,
            "date_range": {
                "start": result_data[0][time_col] if result_data else None,
                "end": result_data[-1][time_col] if result_data else None
            },
            "data": result_data,
            "summary": summary,
            "row_count": len(result_data),
            "schema_auto_detected": schema.dataset_id == "auto_detected"
        }
    
    async def dimension_breakdown(
        self,
        df: pd.DataFrame,
        schema: Optional[SemanticSchema] = None,
        metric_names: Optional[List[str]] = None,
        dimension_name: Optional[str] = None,
        top_n: int = 10,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        dataset_id: str = "auto_detected"
    ) -> Dict[str, Any]:
        """
        Break down metrics by a dimension (100% accurate historical data)
        
        Args:
            df: Source DataFrame
            schema: Semantic schema (auto-detects if None)
            metric_names: Metrics to analyze (uses all if None)
            dimension_name: Dimension to group by (uses first if None)
            top_n: Return top N dimension values
            start_date: Optional start date filter
            end_date: Optional end date filter
            dataset_id: ID for auto-detection
        
        Returns:
            Dimension breakdown results
        """
        # Auto-detect schema if not provided
        schema = await self._ensure_schema(df, schema, dataset_id)
        
        # Use all metrics if none specified
        if metric_names is None:
            metric_names = [m.name for m in schema.metrics]
        
        # Use first dimension if none specified
        if dimension_name is None:
            if not schema.dimensions:
                raise ValueError("No dimensions found in schema")
            dimension_name = schema.dimensions[0].name
        
        # 1. Validate inputs
        metrics = self._get_metrics_by_names(schema, metric_names)
        dimension = self._get_dimension_by_name(schema, dimension_name)
        
        if not dimension:
            raise ValueError(f"Dimension '{dimension_name}' not found in schema")
        
        # 2. Filter by date if provided
        df_filtered = df.copy()
        if start_date or end_date:
            time_col = schema.time_column.column_name
            df_filtered[time_col] = pd.to_datetime(df_filtered[time_col])
            if start_date:
                df_filtered = df_filtered[df_filtered[time_col] >= start_date]
            if end_date:
                df_filtered = df_filtered[df_filtered[time_col] <= end_date]
        
        # 3. Group by dimension
        dim_col = dimension.source_column
        if dim_col not in df_filtered.columns:
            raise ValueError(f"Dimension column '{dim_col}' not found")
        
        grouped = df_filtered.groupby(dim_col)
        
        # 4. Aggregate metrics
        result_data = []
        for dim_value, group in grouped:
            row = {dimension.name: str(dim_value)}
            for metric in metrics:
                value = self._aggregate_metric(group, metric)
                row[metric.name] = round(value, 2) if value else 0
            result_data.append(row)
        
        # 5. Sort by first metric (descending) and take top N
        if result_data and metrics:
            primary_metric = metrics[0].name
            result_data = sorted(
                result_data,
                key=lambda x: x.get(primary_metric, 0),
                reverse=True
            )[:top_n]
        
        # 6. Calculate summary
        summary = {
            "total_dimension_values": len(grouped),
            "showing_top_n": min(top_n, len(result_data)),
            "metrics": {}
        }
        
        for metric in metrics:
            values = [row[metric.name] for row in result_data if row[metric.name] is not None]
            summary["metrics"][metric.name] = {
                "total": round(sum(values), 2) if metric.aggregation == "sum" else None,
                "avg": round(np.mean(values), 2) if values else 0,
                "top_contributor": result_data[0][dimension.name] if result_data else None,
                "top_value": result_data[0][metric.name] if result_data else 0,
                "unit": metric.unit
            }
        
        return {
            "analysis_type": "dimension_breakdown",
            "method": "deterministic",
            "accuracy": "100% (historical data)",
            "dimension": dimension.name,
            "metrics": [m.name for m in metrics],
            "data": result_data,
            "summary": summary,
            "schema_auto_detected": schema.dataset_id == "auto_detected"
        }
    
    async def kpi_analysis(
        self,
        df: pd.DataFrame,
        schema: Optional[SemanticSchema] = None,
        metric_names: Optional[List[str]] = None,
        comparison_period: Literal["previous_period", "previous_year", "none"] = "previous_period",
        dataset_id: str = "auto_detected"
    ) -> Dict[str, Any]:
        """
        Calculate current KPI values with comparison (100% accurate)
        
        Args:
            df: Source DataFrame
            schema: Semantic schema (auto-detects if None)
            metric_names: Metrics to calculate (uses all if None)
            comparison_period: What to compare against
            dataset_id: ID for auto-detection
        
        Returns:
            KPI analysis results
        """
        # Auto-detect schema if not provided
        schema = await self._ensure_schema(df, schema, dataset_id)
        
        # Use all metrics if none specified
        if metric_names is None:
            metric_names = [m.name for m in schema.metrics]
        
        # 1. Validate metrics
        metrics = self._get_metrics_by_names(schema, metric_names)
        time_col = schema.time_column.column_name
        
        # 2. Prepare data
        df_sorted = df.copy()
        df_sorted[time_col] = pd.to_datetime(df_sorted[time_col])
        df_sorted = df_sorted.sort_values(time_col)
        
        # 3. Get latest date
        latest_date = df_sorted[time_col].max()
        earliest_date = df_sorted[time_col].min()
        date_range_days = (latest_date - earliest_date).days
        
        # 4. Determine comparison period
        if comparison_period == "previous_period":
            # Use last 30 days as current, previous 30 as comparison
            current_start = latest_date - timedelta(days=30)
            comparison_start = current_start - timedelta(days=30)
            comparison_end = current_start
        elif comparison_period == "previous_year":
            current_start = latest_date - timedelta(days=30)
            comparison_start = latest_date - timedelta(days=365)
            comparison_end = latest_date - timedelta(days=335)
        else:
            current_start = earliest_date
            comparison_start = None
            comparison_end = None
        
        # 5. Calculate KPIs
        kpis = []
        for metric in metrics:
            # Current period
            current_df = df_sorted[df_sorted[time_col] >= current_start]
            current_value = self._aggregate_metric(current_df, metric)
            
            # Previous period
            if comparison_start:
                previous_df = df_sorted[
                    (df_sorted[time_col] >= comparison_start) &
                    (df_sorted[time_col] <= comparison_end)
                ]
                previous_value = self._aggregate_metric(previous_df, metric)
            else:
                previous_value = None
            
            # Calculate change
            if previous_value and previous_value != 0:
                change = current_value - previous_value
                change_percent = (change / previous_value) * 100
            else:
                change = None
                change_percent = None
            
            kpis.append({
                "metric": metric.name,
                "current_value": round(current_value, 2),
                "previous_value": round(previous_value, 2) if previous_value else None,
                "change": round(change, 2) if change else None,
                "change_percent": round(change_percent, 2) if change_percent else None,
                "direction": metric.direction,
                "unit": metric.unit,
                "metric_type": metric.metric_type,
                "trend": "up" if (change and change > 0) else "down" if (change and change < 0) else "flat"
            })
        
        return {
            "analysis_type": "kpi",
            "method": "deterministic",
            "accuracy": "100% (historical data)",
            "as_of_date": latest_date.isoformat(),
            "comparison_period": comparison_period,
            "current_period_days": 30,
            "kpis": kpis,
            "schema_auto_detected": schema.dataset_id == "auto_detected"
        }
    
    # ============================================
    # ML PREDICTIVE ANALYTICS (Optional Enhancement)
    # ============================================
    
    async def forecast_metric(
        self,
        df: pd.DataFrame,
        schema: Optional[SemanticSchema] = None,
        metric_name: Optional[str] = None,
        periods_ahead: int = 30,
        confidence_interval: float = 0.95,
        include_historical: bool = True,
        dataset_id: str = "auto_detected"
    ) -> Dict[str, Any]:
        """
        Predict future metric values using Facebook Prophet ML
        
        Args:
            df: Source DataFrame
            schema: Semantic schema (auto-detects if None)
            metric_name: Metric to forecast (uses first if None)
            periods_ahead: Number of periods to forecast
            confidence_interval: Confidence level
            include_historical: Include actual historical data
            dataset_id: ID for auto-detection
        
        Returns:
            Forecast results
        """
        # Auto-detect schema if not provided
        schema = await self._ensure_schema(df, schema, dataset_id)
        
        # Use first metric if none specified
        if metric_name is None:
            if not schema.metrics:
                raise ValueError("No metrics found in schema")
            metric_name = schema.metrics[0].name
        
        # Check if ML is enabled
        if not self.enable_ml or not self.ml_available:
            return self._simple_trend_forecast(df, schema, metric_name, periods_ahead)
        
        # Validate metric
        metric = self._get_metrics_by_names(schema, [metric_name])
        if not metric:
            raise ValueError(f"Metric '{metric_name}' not found in schema")
        metric = metric[0]
        
        try:
            from prophet import Prophet
            
            # 1. Prepare data for Prophet
            time_col = schema.time_column.column_name
            metric_col = metric.source_column
            
            # Create Prophet format (ds, y)
            forecast_df = df[[time_col, metric_col]].copy()
            forecast_df[time_col] = pd.to_datetime(forecast_df[time_col])
            
            # Aggregate by day
            forecast_df = forecast_df.groupby(time_col)[metric_col].sum().reset_index()
            forecast_df.columns = ['ds', 'y']
            
            # Remove any NaN values
            forecast_df = forecast_df.dropna()
            
            if len(forecast_df) < 30:
                raise ValueError("Need at least 30 days of data for accurate forecasting")
            
            # 2. Train Prophet model
            model = Prophet(
                interval_width=confidence_interval,
                yearly_seasonality=True if len(forecast_df) > 365 else False,
                weekly_seasonality=True if len(forecast_df) > 14 else False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            model.fit(forecast_df)
            
            # 3. Generate future dates
            future = model.make_future_dataframe(periods=periods_ahead)
            
            # 4. Make predictions
            forecast = model.predict(future)
            
            # 5. Split historical vs future
            historical_size = len(forecast_df)
            historical_forecast = forecast.head(historical_size)
            future_forecast = forecast.tail(periods_ahead)
            
            # 6. Calculate accuracy metrics
            actual = forecast_df['y'].values
            predicted = historical_forecast['yhat'].values
            mae = np.mean(np.abs(actual - predicted))
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            
            # 7. Format historical data
            historical_data = []
            if include_historical:
                for idx, row in forecast_df.iterrows():
                    historical_data.append({
                        "date": row['ds'].strftime('%Y-%m-%d'),
                        "actual_value": round(row['y'], 2),
                        "is_prediction": False
                    })
            
            # 8. Format forecast data
            forecast_data = []
            for idx, row in future_forecast.iterrows():
                forecast_data.append({
                    "date": row['ds'].strftime('%Y-%m-%d'),
                    "predicted_value": round(row['yhat'], 2),
                    "lower_bound": round(row['yhat_lower'], 2),
                    "upper_bound": round(row['yhat_upper'], 2),
                    "is_prediction": True
                })
            
            # 9. Detect seasonality
            seasonality_components = []
            if model.yearly_seasonality:
                seasonality_components.append("yearly")
            if model.weekly_seasonality:
                seasonality_components.append("weekly")
            
            return {
                "analysis_type": "forecast",
                "method": "ml_prophet",
                "metric": metric.name,
                "forecast_horizon": periods_ahead,
                "confidence_level": confidence_interval,
                "historical_data": historical_data if include_historical else None,
                "forecast": forecast_data,
                "accuracy_metrics": {
                    "mape": round(mape, 2),
                    "rmse": round(rmse, 2),
                    "mae": round(mae, 2),
                    "description": f"Model tested on {historical_size} historical data points"
                },
                "model_info": {
                    "algorithm": "Facebook Prophet (Time Series ML)",
                    "training_samples": len(forecast_df),
                    "features": ["trend"] + [f"{s}_seasonality" for s in seasonality_components],
                    "changepoint_detection": "automatic"
                },
                "warnings": [
                    "⚠️ This is a statistical prediction and may not reflect actual future performance",
                    "⚠️ Accuracy decreases for longer forecast horizons",
                    "⚠️ External factors (marketing campaigns, economic changes) are not included",
                    f"⚠️ Based on historical patterns from {forecast_df['ds'].min().strftime('%Y-%m-%d')} to {forecast_df['ds'].max().strftime('%Y-%m-%d')}"
                ],
                "recommendation": "Use forecasts for planning purposes only. Validate with business context and recent trends.",
                "schema_auto_detected": schema.dataset_id == "auto_detected"
            }
            
        except ImportError:
            return self._simple_trend_forecast(df, schema, metric_name, periods_ahead)
        except Exception as e:
            print(f"ML forecast failed: {str(e)}. Using simple trend forecast.")
            return self._simple_trend_forecast(df, schema, metric_name, periods_ahead)
    
    async def detect_anomalies(
        self,
        df: pd.DataFrame,
        schema: Optional[SemanticSchema] = None,
        metric_name: Optional[str] = None,
        sensitivity: float = 2.0,
        min_anomaly_score: float = 0.7,
        dataset_id: str = "auto_detected"
    ) -> Dict[str, Any]:
        """
        Detect unusual/anomalous values using statistical methods
        
        Args:
            df: Source DataFrame
            schema: Semantic schema (auto-detects if None)
            metric_name: Metric to analyze (uses first if None)
            sensitivity: Standard deviation threshold
            min_anomaly_score: Minimum anomaly score to report
            dataset_id: ID for auto-detection
        
        Returns:
            Anomaly detection results
        """
        # Auto-detect schema if not provided
        schema = await self._ensure_schema(df, schema, dataset_id)
        
        # Use first metric if none specified
        if metric_name is None:
            if not schema.metrics:
                raise ValueError("No metrics found in schema")
            metric_name = schema.metrics[0].name
        
        # Validate metric
        metric = self._get_metrics_by_names(schema, [metric_name])
        if not metric:
            raise ValueError(f"Metric '{metric_name}' not found in schema")
        metric = metric[0]
        
        # 1. Prepare time series data
        time_col = schema.time_column.column_name
        metric_col = metric.source_column
        
        df_sorted = df[[time_col, metric_col]].copy()
        df_sorted[time_col] = pd.to_datetime(df_sorted[time_col])
        df_sorted = df_sorted.sort_values(time_col)
        
        # Aggregate by day
        daily_df = df_sorted.groupby(time_col)[metric_col].sum().reset_index()
        daily_df.columns = ['date', 'value']
        
        # 2. Calculate statistical bounds (Z-score method)
        mean_value = daily_df['value'].mean()
        std_value = daily_df['value'].std()
        
        # Handle zero variance
        if std_value == 0:
            return {
                "analysis_type": "anomaly_detection",
                "method": "statistical_zscore",
                "metric": metric.name,
                "anomalies": [],
                "total_anomalies": 0,
                "total_data_points": len(daily_df),
                "anomaly_rate": 0.0,
                "warning": "No variation in data - cannot detect anomalies",
                "baseline_statistics": {
                    "mean": round(mean_value, 2),
                    "std_dev": 0,
                    "note": "Data has zero variance"
                },
                "schema_auto_detected": schema.dataset_id == "auto_detected"
            }
        
        upper_bound = mean_value + (sensitivity * std_value)
        lower_bound = mean_value - (sensitivity * std_value)
        
        # 3. Detect anomalies
        anomalies = []
        for idx, row in daily_df.iterrows():
            value = row['value']
            date = row['date']
            is_anomaly = False
            deviation = 0
            direction = None
            
            if value > upper_bound:
                is_anomaly = True
                deviation = value - upper_bound
                direction = "above"
            elif value < lower_bound:
                is_anomaly = True
                deviation = lower_bound - value
                direction = "below"
            
            if is_anomaly:
                # Calculate anomaly score (0-1)
                z_score = abs((value - mean_value) / std_value)
                anomaly_score = min(z_score / 5.0, 1.0)
                
                if anomaly_score >= min_anomaly_score:
                    # Calculate severity
                    if anomaly_score > 0.9:
                        severity = "critical"
                    elif anomaly_score > 0.8:
                        severity = "high"
                    elif anomaly_score > 0.7:
                        severity = "medium"
                    else:
                        severity = "low"
                    
                    # Calculate percent deviation
                    percent_deviation = (deviation / mean_value) * 100
                    
                    anomalies.append({
                        "date": date.strftime('%Y-%m-%d'),
                        "actual_value": round(value, 2),
                        "expected_range": [round(lower_bound, 2), round(upper_bound, 2)],
                        "expected_value": round(mean_value, 2),
                        "deviation": round(deviation, 2),
                        "deviation_percent": round(percent_deviation, 2),
                        "direction": direction,
                        "severity": severity,
                        "anomaly_score": round(anomaly_score, 2),
                        "description": f"Value is {round(percent_deviation, 1)}% {direction} normal range"
                    })
        
        # 4. Sort by severity
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        anomalies = sorted(anomalies, key=lambda x: severity_order.get(x["severity"], 0), reverse=True)
        
        return {
            "analysis_type": "anomaly_detection",
            "method": "statistical_zscore",
            "metric": metric.name,
            "anomalies": anomalies,
            "total_anomalies": len(anomalies),
            "total_data_points": len(daily_df),
            "anomaly_rate": round((len(anomalies) / len(daily_df)) * 100, 2),
            "detection_settings": {
                "sensitivity": sensitivity,
                "sensitivity_description": "2.0 = moderate (95% confidence interval)",
                "method": "Z-Score (Standard Deviation)",
                "min_anomaly_score": min_anomaly_score
            },
            "baseline_statistics": {
                "mean": round(mean_value, 2),
                "std_dev": round(std_value, 2),
                "expected_range": [round(lower_bound, 2), round(upper_bound, 2)]
            },
            "schema_auto_detected": schema.dataset_id == "auto_detected"
        }
    
    async def analyze_seasonality(
        self,
        df: pd.DataFrame,
        schema: Optional[SemanticSchema] = None,
        metric_name: Optional[str] = None,
        dataset_id: str = "auto_detected"
    ) -> Dict[str, Any]:
        """
        Detect seasonal patterns in data (day of week, month, etc.)
        
        Args:
            df: Source DataFrame
            schema: Semantic schema (auto-detects if None)
            metric_name: Metric to analyze (uses first if None)
            dataset_id: ID for auto-detection
        
        Returns:
            Seasonality analysis results
        """
        # Auto-detect schema if not provided
        schema = await self._ensure_schema(df, schema, dataset_id)
        
        # Use first metric if none specified
        if metric_name is None:
            if not schema.metrics:
                raise ValueError("No metrics found in schema")
            metric_name = schema.metrics[0].name
        
        # Validate metric
        metric = self._get_metrics_by_names(schema, [metric_name])
        if not metric:
            raise ValueError(f"Metric '{metric_name}' not found in schema")
        metric = metric[0]
        
        # Prepare data
        time_col = schema.time_column.column_name
        metric_col = metric.source_column
        
        df_sorted = df[[time_col, metric_col]].copy()
        df_sorted[time_col] = pd.to_datetime(df_sorted[time_col])
        
        # Overall average
        overall_avg = df_sorted[metric_col].mean()
        
        # Day of week analysis
        df_sorted['day_of_week'] = df_sorted[time_col].dt.day_name()
        day_of_week_stats = df_sorted.groupby('day_of_week')[metric_col].mean().to_dict()
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_of_week_analysis = []
        for day in day_order:
            if day in day_of_week_stats:
                avg_val = day_of_week_stats[day]
                vs_overall = ((avg_val - overall_avg) / overall_avg) * 100
                day_of_week_analysis.append({
                    "day": day,
                    "avg_value": round(avg_val, 2),
                    "vs_overall_percent": round(vs_overall, 2),
                    "trend": "above" if vs_overall > 0 else "below" if vs_overall < 0 else "equal"
                })
        
        # Month analysis
        df_sorted['month'] = df_sorted[time_col].dt.month_name()
        month_stats = df_sorted.groupby('month')[metric_col].mean().to_dict()
        
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        month_analysis = []
        for month in month_order:
            if month in month_stats:
                avg_val = month_stats[month]
                vs_overall = ((avg_val - overall_avg) / overall_avg) * 100
                month_analysis.append({
                    "month": month,
                    "avg_value": round(avg_val, 2),
                    "vs_overall_percent": round(vs_overall, 2),
                    "trend": "above" if vs_overall > 0 else "below" if vs_overall < 0 else "equal"
                })
        
        return {
            "analysis_type": "seasonality",
            "method": "statistical_aggregation",
            "metric": metric.name,
            "overall_average": round(overall_avg, 2),
            "patterns": {
                "day_of_week": day_of_week_analysis,
                "month": month_analysis
            },
            "insights": self._generate_seasonality_insights(day_of_week_analysis, month_analysis),
            "schema_auto_detected": schema.dataset_id == "auto_detected"
        }
    
    # ============================================
    # HELPER METHODS
    # ============================================
    
    def _get_metrics_by_names(
        self,
        schema: SemanticSchema,
        metric_names: List[str]
    ) -> List[MetricDefinition]:
        """Get metric definitions by name"""
        return [m for m in schema.metrics if m.name in metric_names]
    
    def _get_dimension_by_name(
        self,
        schema: SemanticSchema,
        dimension_name: str
    ) -> Optional[DimensionDefinition]:
        """Get dimension definition by name"""
        for dim in schema.dimensions:
            if dim.name == dimension_name:
                return dim
        return None
    
    def _aggregate_metric(
        self,
        df: pd.DataFrame,
        metric: MetricDefinition
    ) -> float:
        """Aggregate a metric according to its definition"""
        col = metric.source_column
        if col not in df.columns:
            return 0
        
        values = df[col].dropna()
        if len(values) == 0:
            return 0
        
        if metric.aggregation == "sum":
            return float(values.sum())
        elif metric.aggregation == "avg":
            return float(values.mean())
        elif metric.aggregation == "count":
            return float(len(values))
        elif metric.aggregation == "min":
            return float(values.min())
        elif metric.aggregation == "max":
            return float(values.max())
        elif metric.aggregation == "count_distinct":
            return float(values.nunique())
        else:
            return float(values.sum())
    
    def _group_by_time(
        self,
        df: pd.DataFrame,
        time_col: str,
        granularity: str
    ) -> pd.core.groupby.DataFrameGroupBy:
        """Group DataFrame by time granularity - OPTIMIZED VERSION"""
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        
        if granularity == "day":
            df['_period'] = df[time_col].dt.date
        elif granularity == "week":
            # Vectorized: much faster than .apply()
            df['_period'] = df[time_col].dt.to_period('W').dt.start_time.dt.date
        elif granularity == "month":
            df['_period'] = df[time_col].dt.to_period('M').dt.start_time.dt.date
        elif granularity == "quarter":
            df['_period'] = df[time_col].dt.to_period('Q').dt.start_time.dt.date
        elif granularity == "year":
            df['_period'] = df[time_col].dt.to_period('Y').dt.start_time.dt.date
        else:
            df['_period'] = df[time_col].dt.date
        
        return df.groupby('_period')
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from time series values"""
        if len(values) < 2:
            return "flat"
        
        # Simple trend: compare first half vs second half
        mid = len(values) // 2
        first_half = np.mean(values[:mid])
        second_half = np.mean(values[mid:])
        
        if second_half > first_half * 1.05:  # 5% threshold
            return "up"
        elif second_half < first_half * 0.95:
            return "down"
        else:
            return "flat"
    
    def _simple_trend_forecast(
        self,
        df: pd.DataFrame,
        schema: SemanticSchema,
        metric_name: str,
        periods_ahead: int
    ) -> Dict[str, Any]:
        """Fallback: Simple linear trend forecast when ML is unavailable"""
        # Get metric
        metric = self._get_metrics_by_names(schema, [metric_name])[0]
        
        # Prepare data
        time_col = schema.time_column.column_name
        metric_col = metric.source_column
        
        df_sorted = df[[time_col, metric_col]].copy()
        df_sorted[time_col] = pd.to_datetime(df_sorted[time_col])
        df_sorted = df_sorted.sort_values(time_col)
        
        # Aggregate by day
        daily_df = df_sorted.groupby(time_col)[metric_col].sum().reset_index()
        daily_df.columns = ['date', 'value']
        
        # Calculate simple linear trend
        x = np.arange(len(daily_df))
        y = daily_df['value'].values
        
        # Linear regression (y = mx + b)
        if len(x) > 1:
            m, b = np.polyfit(x, y, 1)
        else:
            m, b = 0, y[0] if len(y) > 0 else 0
        
        # Generate forecast
        last_date = daily_df['date'].max()
        forecast_data = []
        
        for i in range(1, periods_ahead + 1):
            forecast_date = last_date + timedelta(days=i)
            predicted_value = m * (len(x) + i - 1) + b
            
            # Simple confidence interval (±20%)
            lower_bound = predicted_value * 0.8
            upper_bound = predicted_value * 1.2
            
            forecast_data.append({
                "date": forecast_date.strftime('%Y-%m-%d'),
                "predicted_value": round(max(0, predicted_value), 2),
                "lower_bound": round(max(0, lower_bound), 2),
                "upper_bound": round(upper_bound, 2),
                "is_prediction": True
            })
        
        return {
            "analysis_type": "forecast",
            "method": "simple_linear_trend",
            "metric": metric.name,
            "forecast_horizon": periods_ahead,
            "confidence_level": 0.80,
            "forecast": forecast_data,
            "model_info": {
                "algorithm": "Linear Regression (Simple Trend)",
                "training_samples": len(daily_df),
                "features": ["time_trend"]
            },
            "warnings": [
                "⚠️ This is a simple linear trend forecast (ML unavailable)",
                "⚠️ Install 'prophet' for more accurate ML forecasting: pip install prophet",
                "⚠️ Simple trend does not account for seasonality or complex patterns",
                "⚠️ Use for rough estimates only"
            ],
            "recommendation": "For production forecasting, install Facebook Prophet for ML-based predictions",
            "schema_auto_detected": schema.dataset_id == "auto_detected"
        }
    
    def _generate_seasonality_insights(
        self,
        day_of_week_analysis: List[Dict],
        month_analysis: List[Dict]
    ) -> List[str]:
        """Generate insights from seasonality analysis"""
        insights = []
        
        # Find best/worst days
        if day_of_week_analysis:
            best_day = max(day_of_week_analysis, key=lambda x: x['avg_value'])
            worst_day = min(day_of_week_analysis, key=lambda x: x['avg_value'])
            
            if best_day['vs_overall_percent'] > 10:
                insights.append(f"{best_day['day']} performs {best_day['vs_overall_percent']:.1f}% above average")
            if worst_day['vs_overall_percent'] < -10:
                insights.append(f"{worst_day['day']} performs {abs(worst_day['vs_overall_percent']):.1f}% below average")
        
        # Find best/worst months
        if month_analysis:
            best_month = max(month_analysis, key=lambda x: x['avg_value'])
            worst_month = min(month_analysis, key=lambda x: x['avg_value'])
            
            if best_month['vs_overall_percent'] > 15:
                insights.append(f"{best_month['month']} is the strongest month (+{best_month['vs_overall_percent']:.1f}%)")
            if worst_month['vs_overall_percent'] < -15:
                insights.append(f"{worst_month['month']} is the weakest month ({worst_month['vs_overall_percent']:.1f}%)")
        
        if not insights:
            insights.append("No significant seasonal patterns detected")
        
        return insights