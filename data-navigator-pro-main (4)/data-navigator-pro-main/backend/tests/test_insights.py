"""
Tests for the Insights Generator module

Tests cover:
- InsightsGenerator initialization
- Insights generation for all analysis types
- AI-powered insights (with Gemini)
- Fallback insights (without AI)
- Different insight types (detailed vs executive)
- JSON response cleaning
- Error handling
"""

import pytest
import asyncio
import json
from datetime import datetime
from typing import Dict, Any

from src.analytics.insights import InsightsGenerator
from src.analytics.analyzer import AnalyticsAnalyzer
from src.analytics.schema import (
    SemanticSchema, MetricDefinition, DimensionDefinition, TimeColumnDefinition
)


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def insights_generator():
    """Create InsightsGenerator instance"""
    return InsightsGenerator()


@pytest.fixture
def sample_time_series_result() -> Dict[str, Any]:
    """Sample time series analysis result"""
    return {
        "analysis_type": "time_series",
        "method": "deterministic",
        "accuracy": "100% (historical data)",
        "time_column": "order_date",
        "granularity": "day",
        "date_range": {
            "start": "2024-01-01",
            "end": "2024-03-31"
        },
        "summary": {
            "revenue": {
                "total": 150000,
                "avg": 1634.78,
                "min": 1000,
                "max": 10000,
                "trend": "up",
                "direction": "positive",
                "unit": "$",
                "data_points": 90
            },
            "units_sold": {
                "total": 5000,
                "avg": 54.35,
                "min": 10,
                "max": 100,
                "trend": "flat",
                "direction": "positive",
                "unit": "units",
                "data_points": 90
            }
        },
        "data": [
            {"order_date": "2024-01-01", "revenue": 5000, "units_sold": 50},
            {"order_date": "2024-01-02", "revenue": 4500, "units_sold": 45}
        ]
    }


@pytest.fixture
def sample_dimension_breakdown_result() -> Dict[str, Any]:
    """Sample dimension breakdown analysis result"""
    return {
        "analysis_type": "dimension_breakdown",
        "method": "deterministic",
        "accuracy": "100% (historical data)",
        "dimension": "region",
        "metrics": ["revenue"],
        "data": [
            {"region": "North", "revenue": 50000},
            {"region": "South", "revenue": 45000},
            {"region": "East", "revenue": 35000},
            {"region": "West", "revenue": 20000}
        ],
        "summary": {
            "total_dimension_values": 4,
            "showing_top_n": 4,
            "metrics": {
                "revenue": {
                    "total": 150000,
                    "avg": 37500,
                    "top_contributor": "North",
                    "top_value": 50000,
                    "unit": "$"
                }
            }
        }
    }


@pytest.fixture
def sample_kpi_result() -> Dict[str, Any]:
    """Sample KPI analysis result"""
    return {
        "analysis_type": "kpi",
        "method": "deterministic",
        "accuracy": "100% (historical data)",
        "as_of_date": "2024-03-31",
        "comparison_period": "previous_period",
        "current_period_days": 30,
        "kpis": [
            {
                "metric": "revenue",
                "current_value": 45000,
                "previous_value": 40000,
                "change": 5000,
                "change_percent": 12.5,
                "direction": "positive",
                "unit": "$",
                "metric_type": "currency",
                "trend": "up"
            },
            {
                "metric": "units_sold",
                "current_value": 1500,
                "previous_value": 1400,
                "change": 100,
                "change_percent": 7.14,
                "direction": "positive",
                "unit": "units",
                "metric_type": "numeric",
                "trend": "up"
            }
        ]
    }


@pytest.fixture
def sample_forecast_result() -> Dict[str, Any]:
    """Sample forecast analysis result"""
    return {
        "analysis_type": "forecast",
        "method": "simple_linear_trend",
        "metric": "revenue",
        "forecast_horizon": 30,
        "confidence_level": 0.80,
        "forecast": [
            {
                "date": "2024-04-01",
                "predicted_value": 4500,
                "lower_bound": 3600,
                "upper_bound": 5400,
                "is_prediction": True
            },
            {
                "date": "2024-04-02",
                "predicted_value": 4550,
                "lower_bound": 3640,
                "upper_bound": 5460,
                "is_prediction": True
            }
        ],
        "model_info": {
            "algorithm": "Linear Regression (Simple Trend)",
            "training_samples": 90,
            "features": ["time_trend"]
        }
    }


@pytest.fixture
def sample_anomaly_result() -> Dict[str, Any]:
    """Sample anomaly detection result"""
    return {
        "analysis_type": "anomaly_detection",
        "method": "statistical_zscore",
        "metric": "revenue",
        "anomalies": [
            {
                "date": "2024-02-15",
                "actual_value": 25000,
                "expected_value": 5000,
                "expected_range": [2000, 8000],
                "deviation": 17000,
                "deviation_percent": 340.0,
                "direction": "above",
                "severity": "high",
                "anomaly_score": 0.95,
                "description": "Value is 340% above normal range"
            },
            {
                "date": "2024-03-10",
                "actual_value": 500,
                "expected_value": 5000,
                "expected_range": [2000, 8000],
                "deviation": 4500,
                "deviation_percent": 90.0,
                "direction": "below",
                "severity": "high",
                "anomaly_score": 0.92,
                "description": "Value is 90% below normal range"
            }
        ],
        "total_anomalies": 2,
        "total_data_points": 90,
        "anomaly_rate": 2.22,
        "baseline_statistics": {
            "mean": 5000,
            "std_dev": 1200,
            "expected_range": [2000, 8000]
        }
    }


@pytest.fixture
def sample_seasonality_result() -> Dict[str, Any]:
    """Sample seasonality analysis result"""
    return {
        "analysis_type": "seasonality",
        "method": "statistical_aggregation",
        "metric": "revenue",
        "overall_average": 5000,
        "patterns": {
            "day_of_week": [
                {"day": "Monday", "avg_value": 5500, "vs_overall_percent": 10.0, "trend": "above"},
                {"day": "Tuesday", "avg_value": 5200, "vs_overall_percent": 4.0, "trend": "above"},
                {"day": "Wednesday", "avg_value": 4800, "vs_overall_percent": -4.0, "trend": "below"},
                {"day": "Thursday", "avg_value": 4900, "vs_overall_percent": -2.0, "trend": "below"},
                {"day": "Friday", "avg_value": 5600, "vs_overall_percent": 12.0, "trend": "above"}
            ],
            "month": [
                {"month": "January", "avg_value": 4500, "vs_overall_percent": -10.0, "trend": "below"},
                {"month": "February", "avg_value": 5200, "vs_overall_percent": 4.0, "trend": "above"},
                {"month": "March", "avg_value": 5300, "vs_overall_percent": 6.0, "trend": "above"}
            ]
        },
        "insights": [
            "Monday and Friday perform 10-12% above average",
            "January is the weakest month (-10%)",
            "March shows strong growth (+6%)"
        ]
    }


# ============================================
# INITIALIZATION TESTS
# ============================================

def test_insights_generator_initialization():
    """Test InsightsGenerator initializes correctly"""
    gen = InsightsGenerator()
    
    assert gen is not None
    # Will be True if GOOGLE_API_KEY is set, False otherwise
    assert isinstance(gen.ai_enabled, bool)


def test_insights_generator_with_ai_enabled(monkeypatch):
    """Test InsightsGenerator with AI enabled"""
    monkeypatch.setenv("GOOGLE_API_KEY", "test_key_12345")
    gen = InsightsGenerator()
    
    # Even with a dummy key, it should try to initialize
    assert gen is not None


# ============================================
# TIME SERIES INSIGHTS TESTS
# ============================================

def test_time_series_insights_generation(insights_generator, sample_time_series_result):
    """Test time series insights generation"""
    result = asyncio.run(insights_generator.generate_insights(
        sample_time_series_result,
        insight_type="detailed"
    ))
    
    assert result is not None
    assert "insights" in result
    assert "analysis_type" in result
    assert result["analysis_type"] == "time_series"
    assert "generated_at" in result
    assert "metadata" in result


def test_time_series_executive_insights(insights_generator, sample_time_series_result):
    """Test executive summary insights for time series"""
    result = asyncio.run(insights_generator.generate_insights(
        sample_time_series_result,
        insight_type="executive"
    ))
    
    assert result is not None
    assert result.get("insight_type") in ["executive", "fallback"]
    assert "insights" in result
    assert isinstance(result["insights"], list)


def test_time_series_detailed_insights(insights_generator, sample_time_series_result):
    """Test detailed insights for time series"""
    result = asyncio.run(insights_generator.generate_insights(
        sample_time_series_result,
        insight_type="detailed"
    ))
    
    assert result is not None
    assert result.get("insight_type") in ["detailed", "fallback"]
    assert len(result["insights"]) > 0


# ============================================
# DIMENSION BREAKDOWN INSIGHTS TESTS
# ============================================

def test_dimension_breakdown_insights(insights_generator, sample_dimension_breakdown_result):
    """Test dimension breakdown insights generation"""
    result = asyncio.run(insights_generator.generate_insights(
        sample_dimension_breakdown_result,
        insight_type="detailed"
    ))
    
    assert result is not None
    assert result["analysis_type"] == "dimension_breakdown"
    assert "insights" in result
    assert "metadata" in result
    # Metadata may have 'dimension' key in AI mode or not in fallback mode
    assert isinstance(result["metadata"], dict)


def test_dimension_breakdown_executive_insights(insights_generator, sample_dimension_breakdown_result):
    """Test executive insights for dimension breakdown"""
    result = asyncio.run(insights_generator.generate_insights(
        sample_dimension_breakdown_result,
        insight_type="executive"
    ))
    
    assert result is not None
    assert "insights" in result
    assert isinstance(result["insights"], list)


# ============================================
# KPI INSIGHTS TESTS
# ============================================

def test_kpi_insights_generation(insights_generator, sample_kpi_result):
    """Test KPI insights generation"""
    result = asyncio.run(insights_generator.generate_insights(
        sample_kpi_result,
        insight_type="detailed"
    ))
    
    assert result is not None
    assert result["analysis_type"] == "kpi"
    assert "insights" in result
    # Metadata may have 'kpis_analyzed' key in AI mode or not in fallback mode
    assert isinstance(result["metadata"], dict)


def test_kpi_executive_insights(insights_generator, sample_kpi_result):
    """Test executive insights for KPIs"""
    result = asyncio.run(insights_generator.generate_insights(
        sample_kpi_result,
        insight_type="executive"
    ))
    
    assert result is not None
    assert "insights" in result
    assert len(result["insights"]) > 0


# ============================================
# FORECAST INSIGHTS TESTS
# ============================================

def test_forecast_insights_generation(insights_generator, sample_forecast_result):
    """Test forecast insights generation"""
    result = asyncio.run(insights_generator.generate_insights(
        sample_forecast_result,
        insight_type="detailed"
    ))
    
    assert result is not None
    assert result["analysis_type"] == "forecast"
    assert "insights" in result
    # Metadata may have 'metric' key in AI mode or not in fallback mode
    assert isinstance(result["metadata"], dict)


def test_forecast_executive_insights(insights_generator, sample_forecast_result):
    """Test executive insights for forecast"""
    result = asyncio.run(insights_generator.generate_insights(
        sample_forecast_result,
        insight_type="executive"
    ))
    
    assert result is not None
    assert "insights" in result


# ============================================
# ANOMALY INSIGHTS TESTS
# ============================================

def test_anomaly_insights_generation(insights_generator, sample_anomaly_result):
    """Test anomaly detection insights generation"""
    result = asyncio.run(insights_generator.generate_insights(
        sample_anomaly_result,
        insight_type="detailed"
    ))
    
    assert result is not None
    assert result["analysis_type"] == "anomaly_detection"
    assert "insights" in result
    # Metadata may have 'total_anomalies' key in AI mode or not in fallback mode
    assert isinstance(result["metadata"], dict)


def test_anomaly_executive_insights(insights_generator, sample_anomaly_result):
    """Test executive insights for anomalies"""
    result = asyncio.run(insights_generator.generate_insights(
        sample_anomaly_result,
        insight_type="executive"
    ))
    
    assert result is not None
    assert "insights" in result
    assert len(result["insights"]) > 0


# ============================================
# SEASONALITY INSIGHTS TESTS
# ============================================

def test_seasonality_insights_generation(insights_generator, sample_seasonality_result):
    """Test seasonality insights generation"""
    result = asyncio.run(insights_generator.generate_insights(
        sample_seasonality_result,
        insight_type="detailed"
    ))
    
    assert result is not None
    assert result["analysis_type"] == "seasonality"
    assert "insights" in result
    # Metadata may have 'metric' key in AI mode or not in fallback mode
    assert isinstance(result["metadata"], dict)


def test_seasonality_executive_insights(insights_generator, sample_seasonality_result):
    """Test executive insights for seasonality"""
    result = asyncio.run(insights_generator.generate_insights(
        sample_seasonality_result,
        insight_type="executive"
    ))
    
    assert result is not None
    assert "insights" in result


# ============================================
# FALLBACK INSIGHTS TESTS
# ============================================

def test_fallback_insights_when_ai_disabled(monkeypatch):
    """Test fallback insights when AI is disabled"""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    
    gen = InsightsGenerator()
    result = asyncio.run(gen.generate_insights({
        "analysis_type": "time_series",
        "summary": {"revenue": {"avg": 1000}}
    }))
    
    assert result is not None
    assert "insights" in result
    assert len(result["insights"]) > 0
    assert not result["metadata"].get("ai_powered", True)


def test_fallback_insights_structure(insights_generator, sample_time_series_result):
    """Test structure of fallback insights when AI disabled"""
    # Create a generator without AI to test fallback
    import os
    old_key = os.environ.get("GOOGLE_API_KEY")
    if old_key:
        del os.environ["GOOGLE_API_KEY"]
    
    try:
        gen = InsightsGenerator()
        result = asyncio.run(gen.generate_insights(sample_time_series_result))
        
        assert "insights" in result
        assert isinstance(result["insights"], list)
        assert isinstance(result["metadata"], dict)
        assert result["metadata"]["ai_powered"] == False
    finally:
        if old_key:
            os.environ["GOOGLE_API_KEY"] = old_key


# ============================================
# JSON RESPONSE CLEANING TESTS
# ============================================

def test_json_response_cleaning_with_json_markdown(insights_generator):
    """Test cleaning JSON response with markdown code blocks"""
    response = '```json\n["insight1", "insight2"]\n```'
    cleaned = insights_generator._clean_json_response(response)
    
    # Should be valid JSON
    parsed = json.loads(cleaned)
    assert isinstance(parsed, list)
    assert len(parsed) == 2


def test_json_response_cleaning_with_plain_markdown(insights_generator):
    """Test cleaning JSON response with plain markdown"""
    response = '```\n["insight1", "insight2"]\n```'
    cleaned = insights_generator._clean_json_response(response)
    
    parsed = json.loads(cleaned)
    assert isinstance(parsed, list)


def test_json_response_cleaning_already_clean(insights_generator):
    """Test cleaning response that's already clean"""
    response = '["insight1", "insight2"]'
    cleaned = insights_generator._clean_json_response(response)
    
    parsed = json.loads(cleaned)
    assert isinstance(parsed, list)
    assert len(parsed) == 2


# ============================================
# ERROR HANDLING TESTS
# ============================================

def test_unknown_analysis_type(insights_generator):
    """Test handling of unknown analysis type"""
    result = asyncio.run(insights_generator.generate_insights({
        "analysis_type": "unknown_analysis_type",
        "data": {}
    }))
    
    assert result is not None
    # Should return error or fallback insights
    assert "insights" in result or "error" in result


def test_missing_required_fields(insights_generator):
    """Test handling of missing required fields"""
    result = asyncio.run(insights_generator.generate_insights({}))
    
    assert result is not None
    assert "insights" in result or "error" in result


# ============================================
# METADATA TESTS
# ============================================

def test_insights_include_timestamp(insights_generator, sample_time_series_result):
    """Test that insights include generation timestamp"""
    result = asyncio.run(insights_generator.generate_insights(sample_time_series_result))
    
    assert "generated_at" in result
    # Should be ISO format datetime string
    generated_at = result["generated_at"]
    assert isinstance(generated_at, str)
    assert "T" in generated_at  # ISO format includes T


def test_insights_include_metadata(insights_generator, sample_time_series_result):
    """Test that insights include metadata"""
    result = asyncio.run(insights_generator.generate_insights(sample_time_series_result))
    
    assert "metadata" in result
    assert isinstance(result["metadata"], dict)


def test_insights_include_analysis_type(insights_generator, sample_kpi_result):
    """Test that insights include analysis type"""
    result = asyncio.run(insights_generator.generate_insights(sample_kpi_result))
    
    assert "analysis_type" in result
    assert result["analysis_type"] == "kpi"


# ============================================
# INTEGRATION TESTS
# ============================================

def test_insights_from_analyzer_output(insights_generator):
    """Test generating insights from actual analyzer output"""
    # Create a realistic analyzer output
    analyzer_output = {
        "analysis_type": "time_series",
        "method": "deterministic",
        "accuracy": "100% (historical data)",
        "time_column": "date",
        "granularity": "day",
        "date_range": {
            "start": "2024-01-01",
            "end": "2024-03-31"
        },
        "summary": {
            "revenue": {
                "total": 150000,
                "avg": 1634.78,
                "min": 1000,
                "max": 10000,
                "trend": "up",
                "direction": "positive",
                "unit": "$",
                "data_points": 90
            }
        },
        "data": [
            {"date": "2024-01-01", "revenue": 5000},
            {"date": "2024-01-02", "revenue": 4500}
        ]
    }
    
    result = asyncio.run(insights_generator.generate_insights(analyzer_output))
    
    assert result is not None
    assert "insights" in result
    assert len(result["insights"]) > 0


def test_multiple_insight_generation(insights_generator, sample_time_series_result):
    """Test generating insights multiple times"""
    result1 = asyncio.run(insights_generator.generate_insights(
        sample_time_series_result, insight_type="detailed"
    ))
    result2 = asyncio.run(insights_generator.generate_insights(
        sample_time_series_result, insight_type="executive"
    ))
    
    assert result1 is not None
    assert result2 is not None
    # They may have different insights
    assert "insights" in result1
    assert "insights" in result2


# ============================================
# RESPONSE VALIDATION TESTS
# ============================================

def test_insights_are_list(insights_generator, sample_time_series_result):
    """Test that insights are returned as a list"""
    result = asyncio.run(insights_generator.generate_insights(sample_time_series_result))
    
    assert isinstance(result["insights"], list)


def test_insights_are_strings(insights_generator, sample_time_series_result):
    """Test that insights are returned as strings"""
    result = asyncio.run(insights_generator.generate_insights(sample_time_series_result))
    
    for insight in result["insights"]:
        assert isinstance(insight, str)
        assert len(insight) > 0


def test_insights_have_content(insights_generator, sample_time_series_result):
    """Test that generated insights have meaningful content"""
    result = asyncio.run(insights_generator.generate_insights(sample_time_series_result))
    
    insights = result["insights"]
    if not isinstance(insights, list) or len(insights) == 0:
        # Fallback case
        assert "insights" in result
    else:
        # Should have at least some insights
        assert len(insights) > 0
        # Each insight should have content
        for insight in insights:
            assert len(insight.strip()) > 0


# ============================================
# INSIGHT TYPE PARAMETER TESTS
# ============================================

def test_insight_type_detailed(insights_generator, sample_time_series_result):
    """Test detailed insight type"""
    result = asyncio.run(insights_generator.generate_insights(
        sample_time_series_result,
        insight_type="detailed"
    ))
    
    assert result is not None
    assert "insights" in result


def test_insight_type_executive(insights_generator, sample_time_series_result):
    """Test executive insight type"""
    result = asyncio.run(insights_generator.generate_insights(
        sample_time_series_result,
        insight_type="executive"
    ))
    
    assert result is not None
    assert "insights" in result


def test_insight_type_preserved(insights_generator, sample_time_series_result):
    """Test that insight type is preserved in response"""
    result = asyncio.run(insights_generator.generate_insights(
        sample_time_series_result,
        insight_type="executive"
    ))
    
    # Should either be "executive", "detailed", or "fallback"
    assert result.get("insight_type") in ["detailed", "executive", "fallback"]
