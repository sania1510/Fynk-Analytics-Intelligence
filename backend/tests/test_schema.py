import pytest
from datetime import datetime, timedelta
from pydantic import ValidationError

from src.analytics.schema import (
    MetricDefinition,
    DimensionDefinition,
    TimeColumnDefinition,
    SemanticSchema,
    AnalysisRequest,
    AnalysisResult,
)


# Behaviors:
# 1. MetricDefinition validates allowed aggregation and metric_type and sets defaults.
# 2. DimensionDefinition defaults data_type to 'string' and enforces allowed literals.
# 3. SemanticSchema enforces at least one metric and constructs metadata fields with defaults.
# 4. TimeColumnDefinition defaults granularity to 'day' and accepts optional format.
# 5. AnalysisRequest accepts optional dimension_name, time_range, and filters.
# 6. AnalysisResult auto-populates generated_at and preserves provided metadata/data.
# 7. SemanticSchema datetime fields accept datetime instances and are optional.
# 8. MetricDefinition unit can be None and direction defaults to 'positive'.
# 9. Invalid literals raise ValidationError across models.
# 10. SemanticSchema dimensions default to empty list when not provided.


def test_metric_definition_defaults_and_validation():
    m = MetricDefinition(name="revenue", source_column="amount")
    assert m.aggregation == "sum"
    assert m.metric_type == "numeric"
    assert m.direction == "positive"
    assert m.unit is None

    # invalid aggregation
    with pytest.raises(ValidationError):
        MetricDefinition(name="revenue", source_column="amount", aggregation="median")

    # invalid metric_type
    with pytest.raises(ValidationError):
        MetricDefinition(name="revenue", source_column="amount", metric_type="foo")


def test_dimension_definition_defaults_and_validation():
    d = DimensionDefinition(name="product", source_column="product_name")
    assert d.data_type == "string"

    with pytest.raises(ValidationError):
        DimensionDefinition(name="product", source_column="product_name", data_type="datetime")


def test_time_column_definition_defaults_and_optional_format():
    t = TimeColumnDefinition(column_name="order_date")
    assert t.granularity == "day"
    assert t.format is None

    t2 = TimeColumnDefinition(column_name="created_at", granularity="month", format="%Y-%m")
    assert t2.granularity == "month"
    assert t2.format == "%Y-%m"


def test_semantic_schema_requires_metrics_and_sets_defaults():
    time_col = TimeColumnDefinition(column_name="order_date")

    # requires at least one metric
    with pytest.raises(ValidationError):
        SemanticSchema(dataset_id="sales", dataset_name="Sales Data", time_column=time_col, metrics=[])

    metric = MetricDefinition(name="revenue", source_column="amount", aggregation="sum")
    schema = SemanticSchema(
        dataset_id="sales_2024",
        dataset_name="Sales Data 2024",
        time_column=time_col,
        metrics=[metric],
    )

    assert schema.dimensions == []
    assert isinstance(schema.created_at, datetime)
    assert schema.row_count is None
    assert schema.date_range_start is None
    assert schema.date_range_end is None


def test_semantic_schema_accepts_optional_datetime_metadata():
    time_col = TimeColumnDefinition(column_name="order_date")
    metric = MetricDefinition(name="revenue", source_column="amount")

    start = datetime.utcnow() - timedelta(days=30)
    end = datetime.utcnow()

    schema = SemanticSchema(
        dataset_id="sales_2024",
        dataset_name="Sales Data 2024",
        time_column=time_col,
        metrics=[metric],
        row_count=1000,
        date_range_start=start,
        date_range_end=end,
    )

    assert schema.row_count == 1000
    assert schema.date_range_start == start
    assert schema.date_range_end == end


def test_analysis_request_optional_fields_and_literals():
    metric_names = ["revenue", "orders"]
    req = AnalysisRequest(
        dataset_id="sales_2024",
        analysis_type="time_series",
        metric_names=metric_names,
    )
    assert req.dimension_name is None
    assert req.time_range is None
    assert req.filters is None

    # with optional fields populated
    start = datetime.utcnow() - timedelta(days=7)
    end = datetime.utcnow()
    req2 = AnalysisRequest(
        dataset_id="sales_2024",
        analysis_type="breakdown",
        metric_names=metric_names,
        dimension_name="product",
        time_range=(start, end),
        filters={"region": "US"},
    )
    assert req2.dimension_name == "product"
    assert req2.time_range == (start, end)
    assert req2.filters == {"region": "US"}

    with pytest.raises(ValidationError):
        AnalysisRequest(dataset_id="sales_2024", analysis_type="invalid", metric_names=metric_names)


def test_analysis_result_defaults_and_content():
    data = {"points": [1, 2, 3]}
    metadata = {"note": "sample"}
    result = AnalysisResult(analysis_type="kpi", data=data, metadata=metadata)

    assert result.analysis_type == "kpi"
    assert result.data == data
    assert result.metadata == metadata
    assert isinstance(result.generated_at, datetime)
