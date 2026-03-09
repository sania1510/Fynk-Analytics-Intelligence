import asyncio
import json
from datetime import datetime

import pandas as pd
import pytest

from src.data.smart_detector import SmartSchemaDetector
from src.analytics.schema import SemanticSchema, TimeColumnDefinition, MetricDefinition, DimensionDefinition


# Behaviors covered in this suite
# 1. Falls back to rule-based detection when AI is disabled or fails
# 2. Rule-based detection raises when no time/date column is found
# 3. Rule-based detection identifies metrics (numeric) and dimensions (categorical)
# 4. Date range fields are computed from detected time column
# 5. _prepare_sample returns a compact JSON preview with stringified datetimes
# 6. _get_date_min/_get_date_max safely return None on parsing failure
# 7. AI path: when enabled and returning valid JSON, result is mapped to SemanticSchema
# 8. AI path: markdown-wrapped JSON is cleaned and parsed


@pytest.fixture()
def simple_df():
    return pd.DataFrame({
        "order_date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-02"]),
        "amount": [10, 20, 30],
        "product": ["A", "B", "A"],
    })


def test_fallback_to_rules_when_ai_disabled(monkeypatch, simple_df):
    det = SmartSchemaDetector()
    # Force AI disabled regardless of env
    det.ai_enabled = False

    schema = asyncio.run(det.detect_schema(simple_df, dataset_id="ds1", dataset_name="Sample"))
    assert isinstance(schema, SemanticSchema)
    assert schema.dataset_id == "ds1"
    assert schema.time_column.column_name in simple_df.columns
    # Should include numeric metric and string dimension
    assert any(m.source_column == "amount" for m in schema.metrics)
    assert any(d.source_column == "product" for d in schema.dimensions)


def test_rule_based_raises_when_no_time_column():
    det = SmartSchemaDetector()
    # Frame with no date-like column
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    with pytest.raises(ValueError):
        det._detect_with_rules(df, dataset_id="none", dataset_name="NoTime")


def test_rule_based_builds_metrics_dimensions_and_date_range(simple_df):
    det = SmartSchemaDetector()
    schema = det._detect_with_rules(simple_df, dataset_id="ds2", dataset_name="Rules")

    # Metrics and dimensions
    assert any(m.source_column == "amount" for m in schema.metrics)
    assert any(d.source_column == "product" for d in schema.dimensions)

    # Date range computed from order_date
    assert isinstance(schema.date_range_start, datetime)
    assert isinstance(schema.date_range_end, datetime)
    assert schema.date_range_start <= schema.date_range_end


def test_prepare_sample_and_date_helpers(simple_df):
    det = SmartSchemaDetector()

    # prepare sample returns JSON
    sample = det._prepare_sample(simple_df, n=2)
    parsed = json.loads(sample)
    assert isinstance(parsed, list) and len(parsed) == 2
    # timestamps serialized as strings
    assert isinstance(parsed[0]["order_date"], str)

    # date min/max
    dmin = det._get_date_min(simple_df, "order_date")
    dmax = det._get_date_max(simple_df, "order_date")
    assert isinstance(dmin, datetime) and isinstance(dmax, datetime)

    # invalid column returns None safely
    assert det._get_date_min(simple_df, "missing") is None
    assert det._get_date_max(simple_df, "missing") is None


def test_ai_path_parses_valid_json(monkeypatch):
    det = SmartSchemaDetector()
    det.ai_enabled = True

    # Minimal frame to compute row_count and date range
    # Convert ts column to datetime to ensure date range works
    df = pd.DataFrame({
        "ts": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "rev": [100, 200],
        "prod": ["A", "B"],
    })

    # Fake Gemini response object
    class DummyResp:
        def __init__(self, text):
            self.text = text

    payload = {
        "time_column": {"column_name": "ts", "granularity": "day", "format": None},
        "metrics": [
            {"name": "revenue", "source_column": "rev", "aggregation": "sum", "metric_type": "currency", "unit": "$", "direction": "positive"}
        ],
        "dimensions": [
            {"name": "product", "source_column": "prod", "data_type": "string"}
        ],
    }

    def fake_generate_content(prompt):
        return DummyResp(json.dumps(payload))

    # Create a mock model object with generate_content method
    class MockModel:
        def generate_content(self, prompt):
            return DummyResp(json.dumps(payload))
    
    det.model = MockModel()

    schema = asyncio.run(det._detect_with_ai(df, dataset_id="aid", dataset_name="AI"))
    assert isinstance(schema, SemanticSchema)
    assert schema.time_column.column_name == "ts"
    assert any(m.name == "revenue" and m.source_column == "rev" for m in schema.metrics)
    assert any(d.name == "product" and d.source_column == "prod" for d in schema.dimensions)
    # Row count and date range
    assert schema.row_count == 2
    assert isinstance(schema.date_range_start, datetime)
    assert isinstance(schema.date_range_end, datetime)


def test_ai_path_cleans_markdown_wrapped_json(monkeypatch):
    det = SmartSchemaDetector()
    det.ai_enabled = True

    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "amount": [1, 2],
    })

    class DummyResp:
        def __init__(self, text):
            self.text = text

    payload = {
        "time_column": {"column_name": "date", "granularity": "day", "format": None},
        "metrics": [
            {"name": "amount", "source_column": "amount", "aggregation": "sum", "metric_type": "numeric", "unit": None, "direction": "neutral"}
        ],
        "dimensions": [],
    }

    # Create a mock model object with generate_content method
    class MockModel:
        def generate_content(self, prompt):
            return DummyResp("```json\n" + json.dumps(payload) + "\n```")
    
    det.model = MockModel()

    schema = asyncio.run(det._detect_with_ai(df, dataset_id="aid2", dataset_name="AI2"))
    assert isinstance(schema, SemanticSchema)
    assert schema.time_column.column_name == "date"
    assert len(schema.metrics) == 1
    assert schema.metrics[0].source_column == "amount"