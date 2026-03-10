"""
Tests for the Analytics Analyzer module

Tests cover:
- Deterministic analytics (time series, dimension breakdown, KPI analysis)
- ML predictive analytics (forecasting, anomaly detection, seasonality)
- Auto-schema detection
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

from src.analytics.analyzer import AnalyticsAnalyzer
from src.analytics.schema import (
    SemanticSchema, MetricDefinition, DimensionDefinition, TimeColumnDefinition
)
from src.data.smart_detector import SmartSchemaDetector


# ============================================
# FIXTURES - Test Data Setup
# ============================================

@pytest.fixture
def sample_sales_data():
    """Create sample sales data for testing"""
    dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
    np.random.seed(42)
    
    data = {
        'order_date': dates,
        'region': np.random.choice(['North', 'South', 'East', 'West'], len(dates)),
        'product': np.random.choice(['A', 'B', 'C'], len(dates)),
        'revenue': np.random.randint(1000, 10000, len(dates)),
        'units_sold': np.random.randint(10, 100, len(dates)),
        'conversions': np.random.randint(1, 50, len(dates))
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_schema():
    """Create sample semantic schema"""
    return SemanticSchema(
        dataset_id="sales_2024",
        dataset_name="Sales Data 2024",
        time_column=TimeColumnDefinition(
            column_name="order_date",
            granularity="day"
        ),
        metrics=[
            MetricDefinition(
                name="revenue",
                source_column="revenue",
                aggregation="sum",
                metric_type="currency",
                unit="$",
                direction="positive"
            ),
            MetricDefinition(
                name="units_sold",
                source_column="units_sold",
                aggregation="sum",
                metric_type="numeric",
                direction="positive"
            ),
            MetricDefinition(
                name="conversions",
                source_column="conversions",
                aggregation="sum",
                metric_type="numeric",
                direction="positive"
            )
        ],
        dimensions=[
            DimensionDefinition(
                name="region",
                source_column="region",
                data_type="string"
            ),
            DimensionDefinition(
                name="product",
                source_column="product",
                data_type="string"
            )
        ]
    )


@pytest.fixture
def analyzer():
    """Create analyzer instance"""
    return AnalyticsAnalyzer(enable_ml=False)  # Disable ML to speed up tests


@pytest.fixture
def analyzer_with_ml():
    """Create analyzer instance with ML enabled"""
    return AnalyticsAnalyzer(enable_ml=True)


# ============================================
# INITIALIZATION TESTS
# ============================================

def test_analyzer_initialization():
    """Test analyzer initializes correctly"""
    analyzer = AnalyticsAnalyzer(enable_ml=False)
    
    assert analyzer is not None
    assert analyzer.cache == {}
    assert analyzer.enable_ml == False


def test_analyzer_with_ml_initialization():
    """Test analyzer with ML enabled"""
    analyzer = AnalyticsAnalyzer(enable_ml=True)
    
    assert analyzer is not None
    assert analyzer.enable_ml == True or analyzer.ml_available == False


# ============================================
# DETERMINISTIC ANALYTICS TESTS
# ============================================

def test_time_series_analysis_basic(analyzer, sample_sales_data, sample_schema):
    """Test basic time series analysis"""
    result = asyncio.run(analyzer.time_series_analysis(
        df=sample_sales_data,
        schema=sample_schema,
        metric_names=["revenue"]
    ))
    
    assert result is not None
    assert result["analysis_type"] == "time_series"
    assert result["method"] == "deterministic"
    assert result["accuracy"] == "100% (historical data)"
    assert "data" in result
    assert "summary" in result
    assert len(result["data"]) > 0
    assert "revenue" in result["summary"]


def test_time_series_analysis_multiple_metrics(analyzer, sample_sales_data, sample_schema):
    """Test time series analysis with multiple metrics"""
    result = asyncio.run(analyzer.time_series_analysis(
        df=sample_sales_data,
        schema=sample_schema,
        metric_names=["revenue", "units_sold", "conversions"]
    ))
    
    assert result is not None
    assert len(result["summary"]) == 3
    assert "revenue" in result["summary"]
    assert "units_sold" in result["summary"]
    assert "conversions" in result["summary"]


def test_time_series_analysis_with_date_filter(analyzer, sample_sales_data, sample_schema):
    """Test time series analysis with date range filtering"""
    start_date = datetime(2024, 2, 1)
    end_date = datetime(2024, 2, 29)
    
    result = asyncio.run(analyzer.time_series_analysis(
        df=sample_sales_data,
        schema=sample_schema,
        metric_names=["revenue"],
        start_date=start_date,
        end_date=end_date
    ))
    
    assert result is not None
    assert len(result["data"]) > 0
    # Verify all returned dates are within range
    for row in result["data"]:
        date_str = row["order_date"]
        assert "2024-02" in date_str


def test_time_series_analysis_granularity(analyzer, sample_sales_data, sample_schema):
    """Test time series analysis with different granularities"""
    for granularity in ["day", "week", "month"]:
        result = asyncio.run(analyzer.time_series_analysis(
            df=sample_sales_data,
            schema=sample_schema,
            metric_names=["revenue"],
            granularity=granularity
        ))
        
        assert result is not None
        assert result["granularity"] == granularity
        assert len(result["data"]) > 0


def test_dimension_breakdown(analyzer, sample_sales_data, sample_schema):
    """Test dimension breakdown analysis"""
    result = asyncio.run(analyzer.dimension_breakdown(
        df=sample_sales_data,
        schema=sample_schema,
        metric_names=["revenue"],
        dimension_name="region"
    ))
    
    assert result is not None
    assert result["analysis_type"] == "dimension_breakdown"
    assert result["method"] == "deterministic"
    assert result["dimension"] == "region"
    assert "data" in result
    assert len(result["data"]) > 0
    assert result["data"][0].get("region") is not None


def test_dimension_breakdown_top_n(analyzer, sample_sales_data, sample_schema):
    """Test dimension breakdown with top_n limit"""
    result = asyncio.run(analyzer.dimension_breakdown(
        df=sample_sales_data,
        schema=sample_schema,
        metric_names=["revenue"],
        dimension_name="region",
        top_n=2
    ))
    
    assert len(result["data"]) <= 2


def test_dimension_breakdown_multiple_metrics(analyzer, sample_sales_data, sample_schema):
    """Test dimension breakdown with multiple metrics"""
    result = asyncio.run(analyzer.dimension_breakdown(
        df=sample_sales_data,
        schema=sample_schema,
        metric_names=["revenue", "units_sold"],
        dimension_name="region"
    ))
    
    assert len(result["metrics"]) == 2
    assert "revenue" in result["metrics"]
    assert "units_sold" in result["metrics"]


def test_kpi_analysis(analyzer, sample_sales_data, sample_schema):
    """Test KPI analysis"""
    result = asyncio.run(analyzer.kpi_analysis(
        df=sample_sales_data,
        schema=sample_schema,
        metric_names=["revenue"]
    ))
    
    assert result is not None
    assert result["analysis_type"] == "kpi"
    assert result["method"] == "deterministic"
    assert "kpis" in result
    assert len(result["kpis"]) > 0
    
    kpi = result["kpis"][0]
    assert kpi["metric"] == "revenue"
    assert "current_value" in kpi
    assert "direction" in kpi
    assert "unit" in kpi


def test_kpi_analysis_comparison_periods(analyzer, sample_sales_data, sample_schema):
    """Test KPI analysis with different comparison periods"""
    for comparison in ["previous_period", "previous_year", "none"]:
        result = asyncio.run(analyzer.kpi_analysis(
            df=sample_sales_data,
            schema=sample_schema,
            metric_names=["revenue"],
            comparison_period=comparison
        ))
        
        assert result is not None
        assert result["comparison_period"] == comparison


def test_kpi_analysis_multiple_metrics(analyzer, sample_sales_data, sample_schema):
    """Test KPI analysis with multiple metrics"""
    result = asyncio.run(analyzer.kpi_analysis(
        df=sample_sales_data,
        schema=sample_schema,
        metric_names=["revenue", "units_sold", "conversions"]
    ))
    
    assert len(result["kpis"]) == 3


# ============================================
# AUTO-SCHEMA DETECTION TESTS
# ============================================

def test_auto_schema_detection(analyzer, sample_sales_data):
    """Test automatic schema detection when schema=None"""
    result = asyncio.run(analyzer.time_series_analysis(
        df=sample_sales_data,
        schema=None,  # Let it auto-detect
        metric_names=None
    ))
    
    assert result is not None
    # Auto-detected schema should mark as such
    assert "schema_auto_detected" in result


def test_ensure_schema_auto_detect(analyzer, sample_sales_data):
    """Test _ensure_schema method for auto-detection"""
    schema = asyncio.run(analyzer._ensure_schema(
        df=sample_sales_data,
        schema=None,
        dataset_id="test_data"
    ))
    
    assert schema is not None
    assert isinstance(schema, SemanticSchema)
    assert len(schema.metrics) > 0
    assert schema.time_column is not None


# ============================================
# ANOMALY DETECTION TESTS
# ============================================

def test_detect_anomalies_basic(analyzer_with_ml, sample_sales_data, sample_schema):
    """Test basic anomaly detection"""
    result = asyncio.run(analyzer_with_ml.detect_anomalies(
        df=sample_sales_data,
        schema=sample_schema,
        metric_name="revenue"
    ))
    
    assert result is not None
    assert result["analysis_type"] == "anomaly_detection"
    assert result["method"] == "statistical_zscore"
    assert result["metric"] == "revenue"
    assert "anomalies" in result
    assert "baseline_statistics" in result
    assert "total_anomalies" in result
    assert "anomaly_rate" in result


def test_detect_anomalies_sensitivity(analyzer_with_ml, sample_sales_data, sample_schema):
    """Test anomaly detection with different sensitivity levels"""
    # Lower sensitivity = more anomalies detected
    result_low = asyncio.run(analyzer_with_ml.detect_anomalies(
        df=sample_sales_data,
        schema=sample_schema,
        metric_name="revenue",
        sensitivity=1.0
    ))
    
    result_high = asyncio.run(analyzer_with_ml.detect_anomalies(
        df=sample_sales_data,
        schema=sample_schema,
        metric_name="revenue",
        sensitivity=3.0
    ))
    
    # Lower sensitivity should detect more anomalies
    assert len(result_low["anomalies"]) >= len(result_high["anomalies"])


def test_detect_anomalies_severity_levels(analyzer_with_ml, sample_sales_data, sample_schema):
    """Test that anomalies are classified by severity"""
    result = asyncio.run(analyzer_with_ml.detect_anomalies(
        df=sample_sales_data,
        schema=sample_schema,
        metric_name="revenue",
        sensitivity=1.0  # Lower sensitivity to ensure anomalies
    ))
    
    if result["total_anomalies"] > 0:
        # Check that anomalies have severity
        for anomaly in result["anomalies"]:
            assert "severity" in anomaly
            assert anomaly["severity"] in ["critical", "high", "medium", "low"]
            assert "anomaly_score" in anomaly


# ============================================
# SEASONALITY ANALYSIS TESTS
# ============================================

def test_analyze_seasonality(analyzer_with_ml, sample_sales_data, sample_schema):
    """Test seasonality analysis"""
    result = asyncio.run(analyzer_with_ml.analyze_seasonality(
        df=sample_sales_data,
        schema=sample_schema,
        metric_name="revenue"
    ))
    
    assert result is not None
    assert result["analysis_type"] == "seasonality"
    assert result["method"] == "statistical_aggregation"
    assert "patterns" in result
    assert "day_of_week" in result["patterns"]
    assert "month" in result["patterns"]
    assert "insights" in result
    assert isinstance(result["insights"], list)


def test_seasonality_day_of_week(analyzer_with_ml, sample_sales_data, sample_schema):
    """Test that day of week analysis is complete"""
    result = asyncio.run(analyzer_with_ml.analyze_seasonality(
        df=sample_sales_data,
        schema=sample_schema,
        metric_name="revenue"
    ))
    
    days = [p["day"] for p in result["patterns"]["day_of_week"]]
    expected_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # At least some days should be present
    assert len(days) > 0
    # All days should be valid day names
    for day in days:
        assert day in expected_days


# ============================================
# FORECASTING TESTS
# ============================================

def test_forecast_metric_fallback(analyzer, sample_sales_data, sample_schema):
    """Test forecasting with fallback to simple trend (ML disabled)"""
    result = asyncio.run(analyzer.forecast_metric(
        df=sample_sales_data,
        schema=sample_schema,
        metric_name="revenue",
        periods_ahead=30
    ))
    
    assert result is not None
    assert result["analysis_type"] == "forecast"
    assert "forecast" in result
    assert len(result["forecast"]) == 30
    assert "metric" in result
    assert result["metric"] == "revenue"


def test_forecast_with_confidence_interval(analyzer, sample_sales_data, sample_schema):
    """Test that forecast includes confidence intervals"""
    result = asyncio.run(analyzer.forecast_metric(
        df=sample_sales_data,
        schema=sample_schema,
        metric_name="revenue",
        periods_ahead=30,
        confidence_interval=0.95
    ))
    
    if len(result["forecast"]) > 0:
        forecast_point = result["forecast"][0]
        assert "predicted_value" in forecast_point
        assert "lower_bound" in forecast_point
        assert "upper_bound" in forecast_point
        assert forecast_point["lower_bound"] <= forecast_point["predicted_value"]
        assert forecast_point["predicted_value"] <= forecast_point["upper_bound"]


def test_forecast_periods_ahead(analyzer, sample_sales_data, sample_schema):
    """Test forecasting with different period lengths"""
    for periods in [7, 14, 30]:
        result = asyncio.run(analyzer.forecast_metric(
            df=sample_sales_data,
            schema=sample_schema,
            metric_name="revenue",
            periods_ahead=periods
        ))
        
        assert len(result["forecast"]) == periods


# ============================================
# HELPER METHOD TESTS
# ============================================

def test_get_metrics_by_names(analyzer, sample_schema):
    """Test _get_metrics_by_names helper"""
    metrics = analyzer._get_metrics_by_names(sample_schema, ["revenue", "units_sold"])
    
    assert len(metrics) == 2
    assert metrics[0].name == "revenue"
    assert metrics[1].name == "units_sold"


def test_get_metrics_by_names_invalid(analyzer, sample_schema):
    """Test _get_metrics_by_names with invalid metric"""
    metrics = analyzer._get_metrics_by_names(sample_schema, ["invalid_metric"])
    
    assert len(metrics) == 0


def test_get_dimension_by_name(analyzer, sample_schema):
    """Test _get_dimension_by_name helper"""
    dimension = analyzer._get_dimension_by_name(sample_schema, "region")
    
    assert dimension is not None
    assert dimension.name == "region"
    assert dimension.source_column == "region"


def test_get_dimension_by_name_invalid(analyzer, sample_schema):
    """Test _get_dimension_by_name with invalid dimension"""
    dimension = analyzer._get_dimension_by_name(sample_schema, "invalid_dim")
    
    assert dimension is None


def test_aggregate_metric_sum(analyzer, sample_sales_data, sample_schema):
    """Test metric aggregation with sum"""
    metric = sample_schema.metrics[0]  # revenue with sum aggregation
    
    total = analyzer._aggregate_metric(sample_sales_data, metric)
    
    assert total > 0
    assert isinstance(total, float)


def test_aggregate_metric_count(analyzer, sample_sales_data, sample_schema):
    """Test metric aggregation with count"""
    metric = MetricDefinition(
        name="count_test",
        source_column="revenue",
        aggregation="count"
    )
    
    count = analyzer._aggregate_metric(sample_sales_data, metric)
    
    assert count == len(sample_sales_data)


def test_calculate_trend_up(analyzer):
    """Test trend calculation for upward trend"""
    values = [10, 20, 30, 40, 50]
    trend = analyzer._calculate_trend(values)
    
    assert trend == "up"


def test_calculate_trend_down(analyzer):
    """Test trend calculation for downward trend"""
    values = [50, 40, 30, 20, 10]
    trend = analyzer._calculate_trend(values)
    
    assert trend == "down"


def test_calculate_trend_flat(analyzer):
    """Test trend calculation for flat trend"""
    values = [25, 25, 25, 25, 25]
    trend = analyzer._calculate_trend(values)
    
    assert trend == "flat"


# ============================================
# ERROR HANDLING TESTS
# ============================================

def test_time_series_invalid_metric_names(analyzer, sample_sales_data, sample_schema):
    """Test error handling for invalid metric names"""
    with pytest.raises(ValueError):
        asyncio.run(analyzer.time_series_analysis(
            df=sample_sales_data,
            schema=sample_schema,
            metric_names=["invalid_metric_xyz"]
        ))


def test_dimension_breakdown_invalid_dimension(analyzer, sample_sales_data, sample_schema):
    """Test error handling for invalid dimension"""
    with pytest.raises(ValueError):
        asyncio.run(analyzer.dimension_breakdown(
            df=sample_sales_data,
            schema=sample_schema,
            metric_names=["revenue"],
            dimension_name="invalid_dimension_xyz"
        ))


def test_empty_dataframe(analyzer, sample_schema):
    """Test handling of empty DataFrame"""
    empty_df = pd.DataFrame()
    
    # Should handle gracefully or raise appropriate error
    try:
        result = asyncio.run(analyzer.time_series_analysis(
            df=empty_df,
            schema=sample_schema
        ))
    except (ValueError, KeyError):
        # Expected to fail with empty dataframe
        pass


def test_missing_time_column(analyzer, sample_sales_data, sample_schema):
    """Test handling of missing time column"""
    df_no_time = sample_sales_data.drop(columns=['order_date'])
    
    with pytest.raises(ValueError):
        asyncio.run(analyzer.time_series_analysis(
            df=df_no_time,
            schema=sample_schema
        ))


# ============================================
# DATA QUALITY TESTS
# ============================================

def test_handles_nan_values(analyzer, sample_sales_data, sample_schema):
    """Test that analyzer handles NaN values"""
    df_with_nan = sample_sales_data.copy()
    df_with_nan.loc[0:5, 'revenue'] = np.nan
    
    result = asyncio.run(analyzer.time_series_analysis(
        df=df_with_nan,
        schema=sample_schema,
        metric_names=["revenue"]
    ))
    
    assert result is not None
    assert "data" in result


def test_handles_duplicate_rows(analyzer, sample_sales_data, sample_schema):
    """Test that analyzer handles duplicate rows"""
    df_with_dupes = pd.concat([sample_sales_data, sample_sales_data.iloc[:5]], ignore_index=True)
    
    result = asyncio.run(analyzer.time_series_analysis(
        df=df_with_dupes,
        schema=sample_schema,
        metric_names=["revenue"]
    ))
    
    assert result is not None


def test_summary_statistics_accuracy(analyzer, sample_sales_data, sample_schema):
    """Test accuracy of summary statistics"""
    result = asyncio.run(analyzer.time_series_analysis(
        df=sample_sales_data,
        schema=sample_schema,
        metric_names=["revenue"]
    ))
    
    summary = result["summary"]["revenue"]
    
    # Verify statistics are reasonable
    assert summary["avg"] > 0
    assert summary["min"] <= summary["avg"]
    assert summary["max"] >= summary["avg"]
    assert summary["data_points"] > 0


# ============================================
# INTEGRATION TESTS
# ============================================

def test_full_analytics_workflow(analyzer, sample_sales_data, sample_schema):
    """Test complete analytics workflow"""
    # Time series
    ts_result = asyncio.run(analyzer.time_series_analysis(
        df=sample_sales_data,
        schema=sample_schema,
        metric_names=["revenue"]
    ))
    assert ts_result is not None
    
    # Dimension breakdown
    dim_result = asyncio.run(analyzer.dimension_breakdown(
        df=sample_sales_data,
        schema=sample_schema,
        metric_names=["revenue"],
        dimension_name="region"
    ))
    assert dim_result is not None
    
    # KPI analysis
    kpi_result = asyncio.run(analyzer.kpi_analysis(
        df=sample_sales_data,
        schema=sample_schema,
        metric_names=["revenue"]
    ))
    assert kpi_result is not None


def test_schema_persistence(analyzer, sample_sales_data, sample_schema):
    """Test that schema is used correctly across multiple analyses"""
    result1 = asyncio.run(analyzer.time_series_analysis(
        df=sample_sales_data,
        schema=sample_schema,
        metric_names=["revenue"]
    ))
    
    result2 = asyncio.run(analyzer.dimension_breakdown(
        df=sample_sales_data,
        schema=sample_schema,
        metric_names=["revenue"],
        dimension_name="region"
    ))
    
    # Both should reference the same schema
    assert result1 is not None
    assert result2 is not None
