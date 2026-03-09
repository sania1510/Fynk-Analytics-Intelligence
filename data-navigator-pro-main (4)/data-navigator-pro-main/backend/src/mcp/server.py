"""
Analytics MCP Server - Model Context Protocol Server
Exposes analytics tools for AI agents (Gemini, Claude, etc.)

Tools:
- get_schema(file_path) - Auto-detect schema from CSV/Excel
- get_metrics(file_path) - List all available metrics
- run_analysis() - Run time series, KPI, forecast, anomalies, etc.
- generate_dashboard() - Create HTML dashboard
- summarize_insights() - Generate AI insights

Run with: python analytics_mcp_server.py
Or: mcp install analytics_mcp_server.py
"""

import asyncio
import json
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime
from pathlib import Path
import pandas as pd

# FastMCP imports
from mcp.server.fastmcp import FastMCP

# Your existing modules
from src.data.loader import DataLoader
from src.analytics.analyzer import AnalyticsAnalyzer
from src.analytics.insights import InsightsGenerator
from src.data.normalizer import DataNormalizer
from src.data.smart_detector import SmartSchemaDetector

# Initialize FastMCP server
mcp = FastMCP("Analytics Server")

# Global instances (initialized on server start)
loader: DataLoader = None
analyzer: AnalyticsAnalyzer = None
insights_generator: InsightsGenerator = None

# Cache for loaded datasets (file_path -> dataset_id mapping)
dataset_cache: Dict[str, str] = {}


# ============================================
# INITIALIZATION
# ============================================

@mcp.tool()
async def initialize_server() -> dict:
    """
    Initialize the analytics server components.
    Call this first before using other tools.
    
    Returns:
        dict: Server status and available features
    """
    global loader, analyzer, insights_generator
    
    try:
        loader = DataLoader()
        analyzer = AnalyticsAnalyzer(enable_ml=True)
        insights_generator = InsightsGenerator()
        
        return {
            "success": True,
            "status": "initialized",
            "features": {
                "data_loading": True,
                "auto_schema_detection": True,
                "ml_forecasting": analyzer.ml_available,
                "ai_insights": insights_generator.ai_enabled,
                "deterministic_analytics": True
            },
            "supported_formats": ["csv", "xlsx", "xls"],
            "analysis_types": [
                "time_series",
                "dimension_breakdown", 
                "kpi",
                "forecast",
                "anomaly_detection",
                "seasonality"
            ],
            "message": "Analytics server ready!"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Server initialization failed"
        }


# ============================================
# TOOL 1: GET SCHEMA
# ============================================

@mcp.tool()
async def get_schema(file_path: str) -> dict:
    """
    Auto-detect and return the complete schema for a dataset.
    
    Uses AI (Gemini) or rule-based detection to identify:
    - Time/date column
    - Metrics (numeric columns to aggregate)
    - Dimensions (categorical columns for grouping)
    
    Args:
        file_path: Path to CSV or Excel file (e.g., "/data/sales.csv")
    
    Returns:
        dict: Complete schema with all metadata
        {
            "success": bool,
            "dataset_id": str,
            "file_path": str,
            "time_column": {
                "column_name": str,
                "granularity": str,
                "format": str
            },
            "metrics": [
                {
                    "name": str,
                    "source_column": str,
                    "aggregation": str,
                    "metric_type": str,
                    "unit": str,
                    "direction": str,
                    "description": str
                }
            ],
            "dimensions": [
                {
                    "name": str,
                    "source_column": str,
                    "data_type": str,
                    "cardinality": int
                }
            ],
            "dataset_info": {
                "row_count": int,
                "column_count": int,
                "date_range": {
                    "start": str,
                    "end": str
                },
                "file_size_mb": float,
                "memory_usage_mb": float
            },
            "detection_method": str,
            "schema_confidence": float
        }
    """
    global loader, dataset_cache
    
    try:
        # Ensure server is initialized
        if loader is None:
            init_result = await initialize_server()
            if not init_result["success"]:
                return init_result
        
        # Validate file exists
        if not Path(file_path).exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "suggestion": "Check file path and try again"
            }
        
        # Generate unique dataset ID from filename
        dataset_id = Path(file_path).stem + "_" + str(hash(file_path))[-6:]
        
        # Load dataset with auto-detection
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.csv':
            result = await loader.load_csv(
                file_path=file_path,
                dataset_id=dataset_id,
                dataset_name=Path(file_path).name,
                auto_detect=True
            )
        elif file_extension in ['.xlsx', '.xls']:
            result = await loader.load_excel(
                file_path=file_path,
                dataset_id=dataset_id,
                dataset_name=Path(file_path).name,
                auto_detect=True
            )
        else:
            return {
                "success": False,
                "error": f"Unsupported file format: {file_extension}",
                "supported_formats": [".csv", ".xlsx", ".xls"]
            }
        
        if not result['success']:
            return {
                "success": False,
                "error": result.get('error', 'Unknown error during loading')
            }
        
        # Cache the dataset
        dataset_cache[file_path] = dataset_id
        
        # Get schema
        schema = loader.get_schema(dataset_id)
        df = loader.get_dataset(dataset_id)
        
        if schema is None:
            return {
                "success": False,
                "error": "Schema detection failed",
                "suggestion": "File may be empty or improperly formatted"
            }
        
        # Calculate dataset statistics
        file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        # Build detailed response
        return {
            "success": True,
            "dataset_id": dataset_id,
            "file_path": file_path,
            "file_name": Path(file_path).name,
            
            # Time column details
            "time_column": {
                "column_name": schema.time_column.column_name,
                "granularity": schema.time_column.granularity,
                "format": schema.time_column.format,
                "description": f"Primary time/date column for time-series analysis"
            },
            
            # Metrics details
            "metrics": [
                {
                    "name": m.name,
                    "source_column": m.source_column,
                    "aggregation": m.aggregation,
                    "metric_type": m.metric_type,
                    "unit": m.unit,
                    "direction": m.direction,
                    "description": f"{m.metric_type.title()} metric aggregated by {m.aggregation}"
                }
                for m in schema.metrics
            ],
            
            # Dimensions details
            "dimensions": [
                {
                    "name": d.name,
                    "source_column": d.source_column,
                    "data_type": d.data_type,
                    "cardinality": int(df[d.source_column].nunique()),
                    "sample_values": df[d.source_column].value_counts().head(5).to_dict(),
                    "description": f"Categorical dimension for grouping/filtering"
                }
                for d in schema.dimensions
            ],
            
            # Dataset information
            "dataset_info": {
                "row_count": len(df),
                "column_count": len(df.columns),
                "date_range": {
                    "start": schema.date_range_start.isoformat() if schema.date_range_start else None,
                    "end": schema.date_range_end.isoformat() if schema.date_range_end else None,
                    "total_days": (schema.date_range_end - schema.date_range_start).days if schema.date_range_start and schema.date_range_end else None
                },
                "file_size_mb": round(file_size_mb, 2),
                "memory_usage_mb": round(memory_usage_mb, 2),
                "columns": list(df.columns)
            },
            
            # Detection metadata
            "detection_method": "gemini_ai" if insights_generator.ai_enabled else "rule_based",
            "schema_confidence": 0.95 if insights_generator.ai_enabled else 0.80,
            "auto_detected": True,
            
            "message": f"Schema detected successfully for {Path(file_path).name}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "suggestion": "Check file format and content"
        }


# ============================================
# TOOL 2: GET METRICS
# ============================================

@mcp.tool()
async def get_metrics(file_path: str) -> dict:
    """
    Get detailed information about all available metrics in a dataset.
    
    Args:
        file_path: Path to dataset file
    
    Returns:
        dict: Detailed metrics information
        {
            "success": bool,
            "dataset_id": str,
            "metrics": [
                {
                    "name": str,
                    "source_column": str,
                    "aggregation": str,
                    "metric_type": str,
                    "unit": str,
                    "direction": str,
                    "statistics": {
                        "min": float,
                        "max": float,
                        "mean": float,
                        "median": float,
                        "std": float,
                        "total": float
                    },
                    "data_quality": {
                        "missing_count": int,
                        "missing_percent": float,
                        "zero_count": int,
                        "negative_count": int
                    },
                    "recommended_analyses": List[str]
                }
            ],
            "metric_count": int
        }
    """
    try:
        # Get schema first (this will load the dataset if not already loaded)
        schema_result = await get_schema(file_path)
        
        if not schema_result["success"]:
            return schema_result
        
        dataset_id = schema_result["dataset_id"]
        df = loader.get_dataset(dataset_id)
        schema = loader.get_schema(dataset_id)
        
        detailed_metrics = []
        
        for metric in schema.metrics:
            col = metric.source_column
            values = df[col].dropna()
            
            # Calculate statistics
            stats = {
                "min": float(values.min()) if len(values) > 0 else None,
                "max": float(values.max()) if len(values) > 0 else None,
                "mean": float(values.mean()) if len(values) > 0 else None,
                "median": float(values.median()) if len(values) > 0 else None,
                "std": float(values.std()) if len(values) > 0 else None,
                "total": float(values.sum()) if metric.aggregation == "sum" and len(values) > 0 else None,
                "count": int(len(values))
            }
            
            # Data quality metrics
            data_quality = {
                "missing_count": int(df[col].isna().sum()),
                "missing_percent": float((df[col].isna().sum() / len(df)) * 100),
                "zero_count": int((df[col] == 0).sum()),
                "negative_count": int((df[col] < 0).sum()) if pd.api.types.is_numeric_dtype(df[col]) else 0,
                "unique_count": int(df[col].nunique())
            }
            
            # Recommended analyses based on metric type
            recommended = ["time_series", "kpi"]
            
            if metric.metric_type == "currency":
                recommended.extend(["forecast", "anomaly_detection"])
            
            if data_quality["zero_count"] > len(df) * 0.3:
                recommended.append("investigate_zeros")
            
            detailed_metrics.append({
                "name": metric.name,
                "source_column": metric.source_column,
                "aggregation": metric.aggregation,
                "metric_type": metric.metric_type,
                "unit": metric.unit,
                "direction": metric.direction,
                "description": f"{metric.metric_type.title()} metric that should be {metric.direction}",
                "statistics": stats,
                "data_quality": data_quality,
                "recommended_analyses": recommended,
                "interpretation": {
                    "good_direction": "higher" if metric.direction == "positive" else "lower",
                    "aggregation_meaning": f"Values are {metric.aggregation}ed across time periods"
                }
            })
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "file_path": file_path,
            "metrics": detailed_metrics,
            "metric_count": len(detailed_metrics),
            "summary": {
                "currency_metrics": len([m for m in detailed_metrics if m["metric_type"] == "currency"]),
                "numeric_metrics": len([m for m in detailed_metrics if m["metric_type"] == "numeric"]),
                "percentage_metrics": len([m for m in detailed_metrics if m["metric_type"] == "percentage"])
            },
            "message": f"Found {len(detailed_metrics)} metrics in dataset"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


# ============================================
# TOOL 3: RUN ANALYSIS
# ============================================

@mcp.tool()
async def run_analysis(
    file_path: str,
    analysis_type: Literal[
        "time_series",
        "dimension_breakdown", 
        "kpi",
        "forecast",
        "anomaly_detection",
        "seasonality"
    ],
    metrics: List[str],
    dimension: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    granularity: Optional[str] = "day",
    top_n: Optional[int] = 10,
    forecast_days: Optional[int] = 30,
    sensitivity: Optional[float] = 2.0,
    comparison_period: Optional[str] = "previous_period"
) -> dict:
    """
    Run comprehensive analytics on the dataset.
    
    Args:
        file_path: Path to dataset file
        analysis_type: Type of analysis to run
        metrics: List of metric names to analyze
        dimension: Dimension for breakdown analysis (optional)
        start_date: Start date filter (YYYY-MM-DD) (optional)
        end_date: End date filter (YYYY-MM-DD) (optional)
        granularity: Time granularity - "day", "week", "month", "quarter" (default: "day")
        top_n: Number of top items for breakdown (default: 10)
        forecast_days: Days to forecast ahead (default: 30)
        sensitivity: Anomaly detection sensitivity (default: 2.0)
        comparison_period: KPI comparison - "previous_period" or "previous_year" (default: "previous_period")
    
    Returns:
        dict: Complete analysis results with data, summary, and insights
    """
    try:
        # Get schema and load dataset
        schema_result = await get_schema(file_path)
        if not schema_result["success"]:
            return schema_result
        
        dataset_id = schema_result["dataset_id"]
        df = loader.get_dataset(dataset_id)
        schema = loader.get_schema(dataset_id)
        
        # Parse dates if provided
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None
        
        # Route to appropriate analysis method
        if analysis_type == "time_series":
            result = await analyzer.time_series_analysis(
                df=df,
                schema=schema,
                metric_names=metrics,
                start_date=start_dt,
                end_date=end_dt,
                granularity=granularity
            )
            
        elif analysis_type == "dimension_breakdown":
            if not dimension:
                return {
                    "success": False,
                    "error": "dimension parameter required for breakdown analysis",
                    "available_dimensions": [d.name for d in schema.dimensions]
                }
            
            result = await analyzer.dimension_breakdown(
                df=df,
                schema=schema,
                metric_names=metrics,
                dimension_name=dimension,
                top_n=top_n,
                start_date=start_dt,
                end_date=end_dt
            )
            
        elif analysis_type == "kpi":
            result = await analyzer.kpi_analysis(
                df=df,
                schema=schema,
                metric_names=metrics,
                comparison_period=comparison_period
            )
            
        elif analysis_type == "forecast":
            if len(metrics) != 1:
                return {
                    "success": False,
                    "error": "Forecast requires exactly one metric",
                    "provided_metrics": metrics
                }
            
            result = await analyzer.forecast_metric(
                df=df,
                schema=schema,
                metric_name=metrics[0],
                periods_ahead=forecast_days,
                confidence_interval=0.95,
                include_historical=True
            )
            
        elif analysis_type == "anomaly_detection":
            if len(metrics) != 1:
                return {
                    "success": False,
                    "error": "Anomaly detection requires exactly one metric",
                    "provided_metrics": metrics
                }
            
            result = await analyzer.detect_anomalies(
                df=df,
                schema=schema,
                metric_name=metrics[0],
                sensitivity=sensitivity,
                min_anomaly_score=0.7
            )
            
        elif analysis_type == "seasonality":
            if len(metrics) != 1:
                return {
                    "success": False,
                    "error": "Seasonality analysis requires exactly one metric",
                    "provided_metrics": metrics
                }
            
            result = await analyzer.analyze_seasonality(
                df=df,
                schema=schema,
                metric_name=metrics[0]
            )
        
        else:
            return {
                "success": False,
                "error": f"Unknown analysis type: {analysis_type}",
                "supported_types": [
                    "time_series",
                    "dimension_breakdown",
                    "kpi",
                    "forecast",
                    "anomaly_detection",
                    "seasonality"
                ]
            }
        
        # Add metadata to result
        result["success"] = True
        result["dataset_id"] = dataset_id
        result["file_path"] = file_path
        result["parameters"] = {
            "analysis_type": analysis_type,
            "metrics": metrics,
            "dimension": dimension,
            "start_date": start_date,
            "end_date": end_date,
            "granularity": granularity,
            "top_n": top_n if analysis_type == "dimension_breakdown" else None,
            "forecast_days": forecast_days if analysis_type == "forecast" else None,
            "sensitivity": sensitivity if analysis_type == "anomaly_detection" else None
        }
        result["timestamp"] = datetime.utcnow().isoformat()
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "analysis_type": analysis_type
        }


# ============================================
# TOOL 4: GENERATE DASHBOARD
# ============================================

@mcp.tool()
async def generate_dashboard(
    file_path: str,
    metrics: List[str],
    analysis_types: List[str],
    output_path: Optional[str] = None
) -> dict:
    """
    Generate a complete HTML dashboard with multiple analyses and visualizations.
    
    Args:
        file_path: Path to dataset file
        metrics: List of metrics to include in dashboard
        analysis_types: List of analysis types to run (e.g., ["time_series", "kpi", "forecast"])
        output_path: Where to save HTML dashboard (optional, defaults to /tmp/dashboard_<timestamp>.html)
    
    Returns:
        dict: Dashboard generation result with file path and preview data
        {
            "success": bool,
            "dashboard_path": str,
            "dashboard_url": str,
            "analyses_included": List[str],
            "file_size_kb": float,
            "preview_available": bool
        }
    """
    try:
        # Get schema
        schema_result = await get_schema(file_path)
        if not schema_result["success"]:
            return schema_result
        
        # Run all requested analyses
        analysis_results = {}
        
        for analysis_type in analysis_types:
            try:
                result = await run_analysis(
                    file_path=file_path,
                    analysis_type=analysis_type,
                    metrics=metrics[:2] if analysis_type in ["time_series", "kpi"] else metrics[:1],
                    granularity="week"
                )
                
                if result.get("success"):
                    analysis_results[analysis_type] = result
                    
            except Exception as e:
                print(f"Warning: {analysis_type} analysis failed: {e}")
                continue
        
        # Generate HTML dashboard
        html_content = _generate_html_dashboard(
            schema_result=schema_result,
            analysis_results=analysis_results,
            metrics=metrics
        )
        
        # Determine output path
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"/tmp/dashboard_{timestamp}.html"
        
        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        file_size = Path(output_path).stat().st_size / 1024  # KB
        
        return {
            "success": True,
            "dashboard_path": output_path,
            "dashboard_url": f"file://{Path(output_path).absolute()}",
            "analyses_included": list(analysis_results.keys()),
            "metrics_analyzed": metrics,
            "file_size_kb": round(file_size, 2),
            "preview_available": True,
            "message": f"Dashboard generated successfully at {output_path}",
            "open_instructions": f"Open in browser: file://{Path(output_path).absolute()}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


def _generate_html_dashboard(schema_result: dict, analysis_results: dict, metrics: List[str]) -> str:
    """Helper function to generate HTML dashboard"""
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Analytics Dashboard - {schema_result['file_name']}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f7fa;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1e3a8a;
            border-bottom: 3px solid #3b82f6;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #374151;
            margin-top: 30px;
        }}
        .info-box {{
            background: #eff6ff;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #3b82f6;
        }}
        .metric-card {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px;
            min-width: 200px;
        }}
        .chart-container {{
            margin: 20px 0;
        }}
        pre {{
            background: #f3f4f6;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Analytics Dashboard</h1>
        
        <div class="info-box">
            <strong>Dataset:</strong> {schema_result['file_name']}<br>
            <strong>Rows:</strong> {schema_result['dataset_info']['row_count']:,}<br>
            <strong>Date Range:</strong> {schema_result['dataset_info']['date_range']['start']} to {schema_result['dataset_info']['date_range']['end']}<br>
            <strong>Metrics:</strong> {', '.join(metrics)}
        </div>
        
        <h2>📈 Analyses</h2>
"""
    
    # Add each analysis result
    for analysis_type, result in analysis_results.items():
        html += f"""
        <div class="chart-container">
            <h3>{analysis_type.replace('_', ' ').title()}</h3>
            <pre>{json.dumps(result.get('summary', {}), indent=2)}</pre>
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    
    return html


# ============================================
# TOOL 5: SUMMARIZE INSIGHTS
# ============================================

@mcp.tool()
async def summarize_insights(
    file_path: str,
    analysis_type: str,
    metrics: List[str],
    insight_type: Literal["executive", "detailed"] = "detailed",
    dimension: Optional[str] = None
) -> dict:
    """
    Generate AI-powered natural language insights from analysis results.
    
    Args:
        file_path: Path to dataset file
        analysis_type: Type of analysis to run and summarize
        metrics: Metrics to analyze
        insight_type: "executive" for brief summary or "detailed" for comprehensive analysis
        dimension: Dimension for breakdown analysis (optional)
    
    Returns:
        dict: AI-generated insights
        {
            "success": bool,
            "insights": List[str],
            "analysis_summary": dict,
            "insight_type": str,
            "ai_powered": bool,
            "confidence": float
        }
    """
    try:
        # First run the analysis
        analysis_result = await run_analysis(
            file_path=file_path,
            analysis_type=analysis_type,
            metrics=metrics,
            dimension=dimension
        )
        
        if not analysis_result.get("success"):
            return analysis_result
        
        # Generate insights using AI
        insights_result = await insights_generator.generate_insights(
            analysis_result=analysis_result,
            insight_type=insight_type
        )
        
        # Enhance with additional context
        insights_result["success"] = True
        insights_result["file_path"] = file_path
        insights_result["analysis_type"] = analysis_type
        insights_result["metrics_analyzed"] = metrics
        insights_result["insight_count"] = len(insights_result.get("insights", []))
        
        return insights_result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


# ============================================
# BONUS TOOLS
# ============================================

@mcp.tool()
async def list_loaded_datasets() -> dict:
    """
    List all currently loaded datasets in the server.
    
    Returns:
        dict: List of loaded datasets with metadata
    """
    try:
        if loader is None:
            return {
                "success": False,
                "error": "Server not initialized. Call initialize_server() first."
            }
        
        datasets_info = loader.list_datasets()
        
        # Enhance with file paths
        detailed_datasets = []
        for file_path, dataset_id in dataset_cache.items():
            schema = loader.get_schema(dataset_id)
            df = loader.get_dataset(dataset_id)
            
            if schema and df is not None:
                detailed_datasets.append({
                    "file_path": file_path,
                    "dataset_id": dataset_id,
                    "file_name": Path(file_path).name,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "metrics": [m.name for m in schema.metrics],
                    "dimensions": [d.name for d in schema.dimensions],
                    "date_range": {
                        "start": schema.date_range_start.isoformat() if schema.date_range_start else None,
                        "end": schema.date_range_end.isoformat() if schema.date_range_end else None
                    }
                })
        
        return {
            "success": True,
            "datasets": detailed_datasets,
            "total_count": len(detailed_datasets)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()