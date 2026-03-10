
"""
Data Normalizer - Cleans and standardizes DataFrames
Ensures data quality before schema detection and analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import re


class DataNormalizer:
    """
    Cleans and normalizes DataFrames for analytics
    
    Operations:
    - Remove duplicates
    - Handle missing values
    - Standardize column names
    - Convert data types
    - Parse dates
    - Remove invalid rows
    """
    
    def clean_dataframe(
        self,
        df: pd.DataFrame,
        remove_duplicates: bool = True,
        standardize_columns: bool = True,
        handle_missing: bool = True,
        parse_dates: bool = True
    ) -> pd.DataFrame:
        """
        Main cleaning pipeline
        
        Args:
            df: Input DataFrame
            remove_duplicates: Remove duplicate rows
            standardize_columns: Clean column names
            handle_missing: Handle missing values
            parse_dates: Auto-detect and parse date columns
        
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # 1. Standardize column names
        if standardize_columns:
            df = self._standardize_column_names(df)
        
        # 2. Remove duplicates
        if remove_duplicates:
            df = self._remove_duplicates(df)
        
        # 3. Parse dates
        if parse_dates:
            df = self._parse_dates(df)
        
        # 4. Handle missing values
        if handle_missing:
            df = self._handle_missing_values(df)
        
        # 5. Convert numeric columns
        df = self._convert_numeric_columns(df)
        
        # 6. Remove empty rows
        df = df.dropna(how='all')
        
        # 7. Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean column names:
        - Remove leading/trailing spaces
        - Replace spaces with underscores
        - Convert to lowercase
        - Remove special characters
        """
        new_columns = {}
        
        for col in df.columns:
            # Convert to string
            col_str = str(col)
            
            # Remove leading/trailing spaces
            col_str = col_str.strip()
            
            # Replace spaces with underscores
            col_str = col_str.replace(' ', '_')
            
            # Convert to lowercase
            col_str = col_str.lower()
            
            # Remove special characters (keep only alphanumeric and underscores)
            col_str = re.sub(r'[^a-z0-9_]', '', col_str)
            
            # Ensure column name doesn't start with a number
            if col_str[0].isdigit():
                col_str = 'col_' + col_str
            
            new_columns[col] = col_str
        
        df = df.rename(columns=new_columns)
        
        # Handle duplicate column names
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [
                f"{dup}_{i}" if i != 0 else dup 
                for i in range(sum(cols == dup))
            ]
        df.columns = cols
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        initial_rows = len(df)
        df = df.drop_duplicates()
        final_rows = len(df)
        
        if initial_rows > final_rows:
            print(f"Removed {initial_rows - final_rows} duplicate rows")
        
        return df
    
    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Auto-detect and parse date columns
        Looks for columns with date-like names or values
        """
        date_keywords = [
            'date', 'time', 'timestamp', 'dt', 'day', 
            'month', 'year', 'created', 'updated', 'order'
        ]
        
        for col in df.columns:
            # Check if column name suggests it's a date
            is_date_column = any(keyword in col.lower() for keyword in date_keywords)
            
            if is_date_column and df[col].dtype == 'object':
                try:
                    # Try parsing as datetime
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    print(f"Parsed '{col}' as datetime")
                except Exception:
                    pass
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values based on column type
        
        Strategy:
        - Numeric: Fill with 0 (safer for aggregations)
        - Categorical: Fill with "Unknown"
        - Datetime: Leave as NaT (will be filtered out)
        """
        for col in df.columns:
            missing_count = df[col].isna().sum()
            
            if missing_count == 0:
                continue
            
            # Numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
                print(f"Filled {missing_count} missing values in '{col}' with 0")
            
            # Categorical columns
            elif df[col].dtype == 'object':
                df[col] = df[col].fillna("Unknown")
                print(f"Filled {missing_count} missing values in '{col}' with 'Unknown'")
            
            # Datetime columns - leave as NaT
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                print(f"Left {missing_count} missing datetime values in '{col}' as NaT")
        
        return df
    
    def _convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Try to convert string columns to numeric if they contain numbers
        Handles currency symbols, commas, percentages
        """
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains numeric values
                sample = df[col].dropna().head(100)
                
                if len(sample) == 0:
                    continue
                
                # Try to clean and convert
                try:
                    # Remove currency symbols, commas, spaces
                    cleaned = sample.astype(str).str.replace(r'[$,€£¥\s]', '', regex=True)
                    
                    # Remove percentage signs
                    cleaned = cleaned.str.replace('%', '', regex=False)
                    
                    # Try converting to numeric
                    numeric_sample = pd.to_numeric(cleaned, errors='coerce')
                    
                    # If > 80% of values are numeric, convert the whole column
                    if numeric_sample.notna().sum() / len(numeric_sample) > 0.8:
                        df[col] = pd.to_numeric(
                            df[col].astype(str).str.replace(r'[$,€£¥\s%]', '', regex=True),
                            errors='coerce'
                        )
                        print(f"Converted '{col}' to numeric")
                
                except Exception:
                    pass
        
        return df
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a data quality report
        
        Returns:
            Dictionary with quality metrics
        """
        report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "duplicate_rows": df.duplicated().sum(),
            "columns": {}
        }
        
        for col in df.columns:
            report["columns"][col] = {
                "dtype": str(df[col].dtype),
                "missing_count": int(df[col].isna().sum()),
                "missing_percentage": float(df[col].isna().sum() / len(df) * 100),
                "unique_values": int(df[col].nunique()),
                "sample_values": df[col].dropna().head(3).tolist()
            }
        
        return report
    
    def validate_for_analytics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate if DataFrame is ready for analytics
        
        Returns:
            Dictionary with validation results
        """
        issues = []
        warnings = []
        
        # Check for empty DataFrame
        if df.empty:
            issues.append("DataFrame is empty")
        
        # Check for columns
        if len(df.columns) < 2:
            issues.append("DataFrame must have at least 2 columns")
        
        # Check for date column
        has_date_column = any(
            pd.api.types.is_datetime64_any_dtype(df[col]) 
            for col in df.columns
        )
        if not has_date_column:
            warnings.append("No datetime column detected")
        
        # Check for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            issues.append("No numeric columns found for metrics")
        
        # Check for excessive missing values
        for col in df.columns:
            missing_pct = df[col].isna().sum() / len(df) * 100
            if missing_pct > 50:
                warnings.append(f"Column '{col}' has {missing_pct:.1f}% missing values")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "row_count": len(df),
            "column_count": len(df.columns),
            "numeric_columns": len(numeric_cols)
        }

