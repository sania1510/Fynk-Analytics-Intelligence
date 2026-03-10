"""
Smart Schema Detector - AI-powered column detection using Gemini
Analyzes DataFrames and auto-generates semantic schemas
"""

import pandas as pd
import google.generativeai as genai
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.analytics.schema import (
    SemanticSchema,
    MetricDefinition,
    DimensionDefinition,
    TimeColumnDefinition
)
from src.data.column_matcher import ColumnMatcher


class SmartSchemaDetector:
    """
    Uses Gemini AI to intelligently detect schema from DataFrames
    Falls back to rule-based matching if AI fails
    """
    
    def __init__(self):
        # Initialize Gemini - Fixed: properly read from environment variable
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            self.ai_enabled = True
        else:
            self.ai_enabled = False
            self.model = None
            print("Warning: GOOGLE_API_KEY not found. Using rule-based detection only.")
        
        # Fallback matcher
        self.matcher = ColumnMatcher()
    
    async def detect_schema(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        dataset_name: str
    ) -> SemanticSchema:
        """
        Auto-detect schema from DataFrame
        
        Args:
            df: pandas DataFrame to analyze
            dataset_id: Unique identifier for dataset
            dataset_name: Human-readable name
        
        Returns:
            SemanticSchema object
        """
        
        # Try AI detection first
        if self.ai_enabled:
            try:
                schema = await self._detect_with_ai(df, dataset_id, dataset_name)
                if schema:
                    return schema
            except Exception as e:
                print(f"AI detection failed: {e}. Falling back to rule-based.")
        
        # Fallback to rule-based detection
        return self._detect_with_rules(df, dataset_id, dataset_name)
    
    async def _detect_with_ai(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        dataset_name: str
    ) -> Optional[SemanticSchema]:
        """
        Use Gemini to detect schema
        """
        
        # Prepare data sample for AI
        sample_data = self._prepare_sample(df)
        
        # Create prompt
        prompt = f"""
Analyze this dataset and identify the schema structure.

Dataset Name: {dataset_name}
Column Names: {list(df.columns)}
Data Types: {df.dtypes.to_dict()}

Sample Data (first 5 rows):
{sample_data}

Your task:
1. Identify the TIME/DATE column (there should be exactly one)
2. Identify METRICS - numeric columns that should be aggregated (sum, avg, count, etc.)
   - For each metric, determine:
     * Standard name (e.g., "revenue", "cost", "quantity", "conversions")
     * How to aggregate it (sum, avg, count, min, max, count_distinct)
     * Type: "currency", "numeric", or "percentage"
     * Direction: "positive" (higher is better), "negative" (lower is better), or "neutral"
3. Identify DIMENSIONS - categorical columns for grouping/filtering

Return ONLY valid JSON in this EXACT format (no markdown, no extra text):
{{
  "time_column": {{
    "column_name": "actual_column_name_from_data",
    "granularity": "day",
    "format": null
  }},
  "metrics": [
    {{
      "name": "revenue",
      "source_column": "actual_column_name",
      "aggregation": "sum",
      "metric_type": "currency",
      "unit": "$",
      "direction": "positive"
    }}
  ],
  "dimensions": [
    {{
      "name": "product",
      "source_column": "actual_column_name",
      "data_type": "string"
    }}
  ]
}}

Rules:
- source_column MUST match exact column names from the dataset
- If a column name is "rev" or "revenue_amount", map it to metric name "revenue"
- If a column name is "prod" or "product_name", map it to dimension name "product"
- Normalize similar column names to standard metric names
- Only include numeric columns as metrics
- Only include string/categorical columns as dimensions
"""
        
        # Call Gemini
        response = self.model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean response (remove markdown if present)
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON
        schema_dict = json.loads(response_text)
        
        # Build SemanticSchema object
        schema = SemanticSchema(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            time_column=TimeColumnDefinition(**schema_dict["time_column"]),
            metrics=[MetricDefinition(**m) for m in schema_dict["metrics"]],
            dimensions=[DimensionDefinition(**d) for d in schema_dict["dimensions"]],
            row_count=len(df),
            date_range_start=self._get_date_min(df, schema_dict["time_column"]["column_name"]),
            date_range_end=self._get_date_max(df, schema_dict["time_column"]["column_name"])
        )
        
        return schema
    
    def _detect_with_rules(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        dataset_name: str
    ) -> SemanticSchema:
        """
        Fallback: Rule-based schema detection
        Uses pattern matching and heuristics
        """
        
        # 1. Detect time column
        time_col = self.matcher.find_time_column(df)
        if not time_col:
            raise ValueError("No time/date column found in dataset")
        
        # 2. Detect metrics (numeric columns)
        metrics = []
        for col in df.columns:
            if col == time_col:
                continue
            
            if pd.api.types.is_numeric_dtype(df[col]):
                metric_info = self.matcher.classify_metric(col, df[col])
                metrics.append(MetricDefinition(
                    name=metric_info["name"],
                    source_column=col,
                    aggregation=metric_info["aggregation"],
                    metric_type=metric_info["type"],
                    unit=metric_info.get("unit"),
                    direction=metric_info["direction"]
                ))
        
        # 3. Detect dimensions (categorical columns)
        dimensions = []
        for col in df.columns:
            if col == time_col:
                continue
            
            if not pd.api.types.is_numeric_dtype(df[col]):
                dim_info = self.matcher.classify_dimension(col, df[col])
                dimensions.append(DimensionDefinition(
                    name=dim_info["name"],
                    source_column=col,
                    data_type=dim_info["data_type"]
                ))
        
        # 4. Build schema
        schema = SemanticSchema(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            time_column=TimeColumnDefinition(
                column_name=time_col,
                granularity="day"
            ),
            metrics=metrics,
            dimensions=dimensions,
            row_count=len(df),
            date_range_start=self._get_date_min(df, time_col),
            date_range_end=self._get_date_max(df, time_col)
        )
        
        return schema
    
    def _prepare_sample(self, df: pd.DataFrame, n: int = 5) -> str:
        """Prepare sample data for AI analysis"""
        sample = df.head(n).to_dict(orient='records')
        return json.dumps(sample, indent=2, default=str)
    
    def _get_date_min(self, df: pd.DataFrame, time_col: str) -> Optional[datetime]:
        """Get minimum date from time column"""
        try:
            return pd.to_datetime(df[time_col]).min().to_pydatetime()
        except Exception:
            return None
    
    def _get_date_max(self, df: pd.DataFrame, time_col: str) -> Optional[datetime]:
        """Get maximum date from time column"""
        try:
            return pd.to_datetime(df[time_col]).max().to_pydatetime()
        except Exception:
            return None