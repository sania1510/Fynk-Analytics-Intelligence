"""
Data Loader - Reads files and creates schemas
Supports: CSV, Excel, JSON
"""

import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json

from analytics.schema import SemanticSchema
from data.smart_detector import SmartSchemaDetector
from data.normalizer import DataNormalizer


class DataLoader:
    """
    Loads data files and auto-generates schemas
    """
    
    def __init__(self):
        self.data_store: Dict[str, pd.DataFrame] = {}
        self.schema_store: Dict[str, SemanticSchema] = {}
        self.detector = SmartSchemaDetector()
        self.normalizer = DataNormalizer()
    
    async def load_csv(
        self,
        file_path: str,
        dataset_id: str,
        dataset_name: Optional[str] = None,
        auto_detect: bool = True
    ) -> Dict[str, Any]:
        """
        Load CSV file and auto-detect schema
        
        Args:
            file_path: Path to CSV file
            dataset_id: Unique identifier for this dataset
            dataset_name: Human-readable name (optional)
            auto_detect: Whether to auto-detect schema using AI
        
        Returns:
            Dictionary with dataset info and schema
        """
        try:
            df = pd.read_csv(file_path)

            if df.empty:
                raise ValueError("CSV file is empty")
            
            
            df = self.normalizer.clean_dataframe(df)
            
            
            if auto_detect:
                schema = await self.detector.detect_schema(
                    df=df,
                    dataset_id=dataset_id,
                    dataset_name=dataset_name or dataset_id
                )
            else:
                
                schema = None
            
            self.data_store[dataset_id] = df
            if schema:
                self.schema_store[dataset_id] = schema
            
            # 6. Return summary
            return {
                "success": True,
                "dataset_id": dataset_id,
                "dataset_name": dataset_name or dataset_id,
                "rows": len(df),
                "columns": list(df.columns),
                "date_range": self._get_date_range(df, schema) if schema else None,
                "schema": schema.dict() if schema else None,
                "message": f"Loaded {len(df)} rows with {len(df.columns)} columns"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "dataset_id": dataset_id
            }
    
    async def load_excel(
        self,
        file_path: str,
        dataset_id: str,
        dataset_name: Optional[str] = None,
        sheet_name: Optional[str] = None,
        auto_detect: bool = True
    ) -> Dict[str, Any]:
        """
        Load Excel file and auto-detect schema
        
        Args:
            file_path: Path to Excel file
            dataset_id: Unique identifier
            dataset_name: Human-readable name
            sheet_name: Specific sheet to load (None = first sheet)
            auto_detect: Whether to auto-detect schema
        
        Returns:
            Dictionary with dataset info and schema
        """
        try:
            
            df = pd.read_excel(file_path, sheet_name=sheet_name or 0)
            
            if df.empty:
                raise ValueError("Excel file is empty")
            
            
            df = self.normalizer.clean_dataframe(df)
            
            if auto_detect:
                schema = await self.detector.detect_schema(
                    df=df,
                    dataset_id=dataset_id,
                    dataset_name=dataset_name or dataset_id
                )
            else:
                schema = None
            
            self.data_store[dataset_id] = df
            if schema:
                self.schema_store[dataset_id] = schema
            
            return {
                "success": True,
                "dataset_id": dataset_id,
                "dataset_name": dataset_name or dataset_id,
                "rows": len(df),
                "columns": list(df.columns),
                "sheet_name": sheet_name,
                "schema": schema.dict() if schema else None,
                "message": f"Loaded Excel file with {len(df)} rows"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "dataset_id": dataset_id
            }
    
    async def load_json(
        self,
        file_path: str,
        dataset_id: str,
        dataset_name: Optional[str] = None,
        auto_detect: bool = True
    ) -> Dict[str, Any]:
        """
        Load JSON file and auto-detect schema
        
        Supports:
        - Array of objects: [{"col1": "val1", "col2": "val2"}, ...]
        - Line-delimited JSON
        
        Args:
            file_path: Path to JSON file
            dataset_id: Unique identifier
            dataset_name: Human-readable name
            auto_detect: Whether to auto-detect schema
        
        Returns:
            Dictionary with dataset info and schema
        """
        try:
            
            try:
                df = pd.read_json(file_path)
            except ValueError:
                
                df = pd.read_json(file_path, lines=True)
            
            if df.empty:
                raise ValueError("JSON file is empty")
            
            
            df = self.normalizer.clean_dataframe(df)
            
            
            if auto_detect:
                schema = await self.detector.detect_schema(
                    df=df,
                    dataset_id=dataset_id,
                    dataset_name=dataset_name or dataset_id
                )
            else:
                schema = None
            
            # Store
            self.data_store[dataset_id] = df
            if schema:
                self.schema_store[dataset_id] = schema
            
            return {
                "success": True,
                "dataset_id": dataset_id,
                "dataset_name": dataset_name or dataset_id,
                "rows": len(df),
                "columns": list(df.columns),
                "schema": schema.dict() if schema else None,
                "message": f"Loaded JSON file with {len(df)} rows"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "dataset_id": dataset_id
            }
    
    def get_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Retrieve a loaded dataset"""
        return self.data_store.get(dataset_id)
    
    def get_schema(self, dataset_id: str) -> Optional[SemanticSchema]:
        """Retrieve a dataset's schema"""
        return self.schema_store.get(dataset_id)
    
    def list_datasets(self) -> Dict[str, Any]:
        """List all loaded datasets"""
        return {
            "datasets": [
                {
                    "dataset_id": dataset_id,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "has_schema": dataset_id in self.schema_store
                }
                for dataset_id, df in self.data_store.items()
            ],
            "count": len(self.data_store)
        }
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset and its schema"""
        deleted = False
        
        if dataset_id in self.data_store:
            del self.data_store[dataset_id]
            deleted = True
        
        if dataset_id in self.schema_store:
            del self.schema_store[dataset_id]
        
        return deleted
    
    def _get_date_range(
        self,
        df: pd.DataFrame,
        schema: SemanticSchema
    ) -> Optional[Dict[str, str]]:
        """Get date range from time column"""
        try:
            time_col = schema.time_column.column_name
            if time_col in df.columns:
                return {
                    "start": df[time_col].min().isoformat(),
                    "end": df[time_col].max().isoformat()
                }
        except Exception:
            pass
        return None
