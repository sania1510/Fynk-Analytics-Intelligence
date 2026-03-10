"""
Column Matcher - Rule-based column type detection
Fallback when AI detection is unavailable
"""

import pandas as pd
import re
from typing import Dict, Any, Optional, List


class ColumnMatcher:
    """
    Uses pattern matching and heuristics to classify columns
    into time columns, metrics, and dimensions
    """
    
    # Patterns for detecting time columns
    TIME_PATTERNS = [
        r'date', r'time', r'timestamp', r'dt', r'day',
        r'month', r'year', r'created', r'updated',
        r'order_date', r'purchase_date', r'transaction_date'
    ]
    
    # Patterns for detecting metric types
    REVENUE_PATTERNS = [
        r'revenue', r'rev', r'sales', r'income', r'amount',
        r'total', r'price', r'value', r'gmv', r'earning'
    ]
    
    COST_PATTERNS = [
        r'cost', r'spend', r'expense', r'cogs', r'fee',
        r'charge', r'budget', r'investment'
    ]
    
    COUNT_PATTERNS = [
        r'count', r'quantity', r'qty', r'number', r'num',
        r'units', r'volume', r'orders', r'transactions'
    ]
    
    CONVERSION_PATTERNS = [
        r'conversion', r'cvr', r'rate', r'percentage',
        r'ratio', r'ctr', r'click'
    ]
    
    # Patterns for detecting dimensions
    PRODUCT_PATTERNS = [
        r'product', r'item', r'sku', r'category', r'brand'
    ]
    
    REGION_PATTERNS = [
        r'region', r'country', r'state', r'city', r'location',
        r'geo', r'territory', r'market'
    ]
    
    CUSTOMER_PATTERNS = [
        r'customer', r'user', r'client', r'account', r'segment'
    ]
    
    CHANNEL_PATTERNS = [
        r'channel', r'source', r'medium', r'campaign', r'platform'
    ]
    
    def find_time_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Find the time/date column in a DataFrame
        
        Strategy:
        1. Look for datetime dtype columns
        2. Look for columns with date-like names
        3. Try parsing columns with date-like values
        
        Returns:
            Column name or None
        """
        
        # 1. Check for datetime columns
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        
        # 2. Check for date-like column names
        for col in df.columns:
            col_lower = str(col).lower()
            for pattern in self.TIME_PATTERNS:
                if re.search(pattern, col_lower):
                    # Try parsing as datetime
                    try:
                        pd.to_datetime(df[col], errors='coerce')
                        return col
                    except Exception:
                        continue
        
        # 3. Try parsing object columns
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    parsed = pd.to_datetime(df[col], errors='coerce')
                    if parsed.notna().sum() / len(df) > 0.5:  # >50% parseable
                        return col
                except Exception:
                    continue
        
        return None
    
    def classify_metric(self, col_name: str, col_data: pd.Series) -> Dict[str, Any]:
        """
        Classify a numeric column as a specific metric type
        
        Args:
            col_name: Column name
            col_data: Column data (pandas Series)
        
        Returns:
            Dictionary with metric classification
        """
        col_lower = col_name.lower()
        
        # Default classification
        result = {
            "name": col_name,
            "aggregation": "sum",
            "type": "numeric",
            "unit": None,
            "direction": "positive"
        }
        
        # Check for revenue metrics
        if any(re.search(pattern, col_lower) for pattern in self.REVENUE_PATTERNS):
            result["name"] = "revenue"
            result["aggregation"] = "sum"
            result["type"] = "currency"
            result["unit"] = "$"
            result["direction"] = "positive"
        
        # Check for cost metrics
        elif any(re.search(pattern, col_lower) for pattern in self.COST_PATTERNS):
            result["name"] = "cost"
            result["aggregation"] = "sum"
            result["type"] = "currency"
            result["unit"] = "$"
            result["direction"] = "negative"
        
        # Check for count metrics
        elif any(re.search(pattern, col_lower) for pattern in self.COUNT_PATTERNS):
            result["name"] = "count"
            result["aggregation"] = "sum"
            result["type"] = "numeric"
            result["unit"] = "units"
            result["direction"] = "positive"
        
        # Check for conversion/rate metrics
        elif any(re.search(pattern, col_lower) for pattern in self.CONVERSION_PATTERNS):
            result["name"] = "conversion_rate"
            result["aggregation"] = "avg"
            result["type"] = "percentage"
            result["unit"] = "%"
            result["direction"] = "positive"
        
        # Analyze data values to refine classification
        if col_data.notna().sum() > 0:
            # Check if values look like percentages (0-100 or 0-1)
            max_val = col_data.max()
            if 0 <= max_val <= 1:
                result["type"] = "percentage"
                result["unit"] = "%"
                result["aggregation"] = "avg"
            elif 0 <= max_val <= 100 and result["type"] == "numeric":
                result["type"] = "percentage"
                result["unit"] = "%"
                result["aggregation"] = "avg"
        
        return result
    
    def classify_dimension(self, col_name: str, col_data: pd.Series) -> Dict[str, Any]:
        """
        Classify a categorical column as a specific dimension type
        
        Args:
            col_name: Column name
            col_data: Column data (pandas Series)
        
        Returns:
            Dictionary with dimension classification
        """
        col_lower = col_name.lower()
        
        # Default classification
        result = {
            "name": col_name,
            "data_type": "string"
        }
        
        # Check for product dimensions
        if any(re.search(pattern, col_lower) for pattern in self.PRODUCT_PATTERNS):
            result["name"] = "product"
        
        # Check for region dimensions
        elif any(re.search(pattern, col_lower) for pattern in self.REGION_PATTERNS):
            result["name"] = "region"
        
        # Check for customer dimensions
        elif any(re.search(pattern, col_lower) for pattern in self.CUSTOMER_PATTERNS):
            result["name"] = "customer"
        
        # Check for channel dimensions
        elif any(re.search(pattern, col_lower) for pattern in self.CHANNEL_PATTERNS):
            result["name"] = "channel"
        
        # Analyze data to refine classification
        if col_data.notna().sum() > 0:
            unique_count = col_data.nunique()
            total_count = len(col_data)
            
            # If very few unique values, it's likely a category
            if unique_count < 20 and unique_count / total_count < 0.1:
                result["data_type"] = "string"
            
            # Check if it's a boolean
            unique_values = set(col_data.dropna().unique())
            if unique_values <= {True, False, 'true', 'false', 'True', 'False', 1, 0, '1', '0', 'yes', 'no'}:
                result["data_type"] = "boolean"
        
        return result
    
    def analyze_column_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze all columns and suggest schema structure
        
        Returns:
            Dictionary with detected patterns
        """
        analysis = {
            "time_columns": [],
            "metric_columns": [],
            "dimension_columns": [],
            "unknown_columns": []
        }
        
        for col in df.columns:
            # Check if it's a time column
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                analysis["time_columns"].append(col)
            
            # Check if it's numeric (potential metric)
            elif pd.api.types.is_numeric_dtype(df[col]):
                metric_info = self.classify_metric(col, df[col])
                analysis["metric_columns"].append({
                    "column": col,
                    "classification": metric_info
                })
            
            # Check if it's categorical (potential dimension)
            elif df[col].dtype == 'object' or df[col].dtype == 'category':
                dim_info = self.classify_dimension(col, df[col])
                analysis["dimension_columns"].append({
                    "column": col,
                    "classification": dim_info
                })
            
            else:
                analysis["unknown_columns"].append(col)
        
        return analysis
    
    def suggest_schema_improvements(self, df: pd.DataFrame) -> List[str]:
        """
        Suggest improvements to the dataset structure
        
        Returns:
            List of suggestions
        """
        suggestions = []
        
        # Check for time column
        time_col = self.find_time_column(df)
        if not time_col:
            suggestions.append(
                "No time/date column detected. Add a column with dates for time-series analysis."
            )
        
        # Check for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            suggestions.append(
                "No numeric columns found. Add metrics (revenue, cost, quantity, etc.) for analysis."
            )
        
        # Check for categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) == 0:
            suggestions.append(
                "No categorical columns found. Add dimensions (product, region, customer, etc.) for breakdowns."
            )
        
        # Check for missing values
        for col in df.columns:
            missing_pct = df[col].isna().sum() / len(df) * 100
            if missing_pct > 30:
                suggestions.append(
                    f"Column '{col}' has {missing_pct:.1f}% missing values. Consider cleaning or removing."
                )
        
        # Check for low cardinality in numeric columns
        for col in numeric_cols:
            if df[col].nunique() < 10:
                suggestions.append(
                    f"Numeric column '{col}' has only {df[col].nunique()} unique values. Consider treating as categorical."
                )
        
        return suggestions
