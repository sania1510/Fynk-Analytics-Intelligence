from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime
class MetricDefinition(BaseModel):
    """Defines how a metric is calculated"""
    name: str = Field(..., description="Metric name (e.g., 'revenue', 'conversions')")
    source_column: str = Field(..., description="Column name in the data")
    aggregation: Literal["sum", "avg", "count", "min", "max", "count_distinct"] = Field(
        default="sum",
        description="How to aggregate this metric"
    )
    metric_type: Literal["numeric", "currency", "percentage"] = Field(
        default="numeric",
        description="Display type for formatting"
    )
    unit: Optional[str] = Field(None, description="Unit symbol (e.g., '$', '%', 'units')")
    direction: Literal["positive", "negative", "neutral"] = Field(
        default="positive",
        description="Whether increase is good/bad/neutral"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "revenue",
                "source_column": "amount",
                "aggregation": "sum",
                "metric_type": "currency",
                "unit": "$",
                "direction": "positive"
            }
        }
class DimensionDefinition(BaseModel):
    """Defines a dimension for grouping/filtering"""
    name: str = Field(..., description="Dimension name (e.g., 'product', 'region')")
    source_column: str = Field(..., description="Column name in the data")
    data_type: Literal["string", "numeric", "boolean", "date"] = Field(
        default="string",
        description="Data type of this dimension"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "product",
                "source_column": "product_name",
                "data_type": "string"
            }
        }


class TimeColumnDefinition(BaseModel):
    """Defines the time column for time-series analysis"""
    column_name: str = Field(..., description="Name of the datetime column")
    granularity: Literal["hour", "day", "week", "month", "quarter", "year"] = Field(
        default="day",
        description="Time granularity for analysis"
    )
    format: Optional[str] = Field(
        None,
        description="DateTime format if parsing is needed (e.g., '%Y-%m-%d')"
    )


class SemanticSchema(BaseModel):
    """
    The complete semantic schema for a dataset.
    This is what the AI uses to understand your data.
    """
    dataset_id: str = Field(..., description="Unique identifier for this dataset")
    dataset_name: str = Field(..., description="Human-readable name")
    time_column: TimeColumnDefinition
    metrics: List[MetricDefinition] = Field(..., min_length=1)
    dimensions: List[DimensionDefinition] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    row_count: Optional[int] = None
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "dataset_id": "sales_2024",
                "dataset_name": "Sales Data 2024",
                "time_column": {
                    "column_name": "order_date",
                    "granularity": "day"
                },
                "metrics": [
                    {
                        "name": "revenue",
                        "source_column": "amount",
                        "aggregation": "sum",
                        "metric_type": "currency",
                        "unit": "$",
                        "direction": "positive"
                    }
                ],
                "dimensions": [
                    {
                        "name": "product",
                        "source_column": "product_name",
                        "data_type": "string"
                    }
                ]
            }
        }


class AnalysisRequest(BaseModel):
    """Request model for running analytics"""
    dataset_id: str
    analysis_type: Literal["time_series", "breakdown", "comparison", "kpi"]
    metric_names: List[str]
    dimension_name: Optional[str] = None
    time_range: Optional[tuple[datetime, datetime]] = None
    filters: Optional[dict] = None


class AnalysisResult(BaseModel):
    """Standardized result from any analysis"""
    analysis_type: str
    data: dict
    metadata: dict
    generated_at: datetime = Field(default_factory=datetime.utcnow)
