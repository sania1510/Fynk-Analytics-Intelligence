# backend/src/api/routes.py
"""
API Routes - FastAPI Endpoints
Complete implementation with ALL analytics endpoints
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Literal
import os
import tempfile
import uuid
from datetime import datetime

# Import backend services
from data.loader import DataLoader
from analytics.schema import SemanticSchema
from analytics.analyzer import AnalyticsAnalyzer
from analytics.insights import InsightsGenerator

# Initialize router
router = APIRouter()

# Initialize services (singletons)
data_loader = DataLoader()
analyzer = AnalyticsAnalyzer(enable_ml=True)
insights_generator = InsightsGenerator()


# ============================================
# REQUEST MODELS
# ============================================

class AnalysisRequest(BaseModel):
    dataset_id: str
    metric_names: Optional[List[str]] = None
    dimension_name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    granularity: Optional[str] = "day"
    top_n: Optional[int] = 10
    comparison_period: Optional[str] = "previous_period"

class ForecastRequest(BaseModel):
    dataset_id: str
    metric_name: Optional[str] = None
    periods_ahead: int = 30
    confidence_interval: float = 0.95
    include_historical: bool = True

class AnomalyRequest(BaseModel):
    dataset_id: str
    metric_name: Optional[str] = None
    sensitivity: float = 2.0
    min_anomaly_score: float = 0.7

class InsightsRequest(BaseModel):
    analysis_result: Dict[str, Any]
    insight_type: Literal["detailed", "executive"] = "detailed"


# ============================================
# UTILITY
# ============================================

def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except Exception:
        return None


# ============================================
# FILE UPLOAD
# ============================================

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    dataset_id: Optional[str] = Form(None),
    dataset_name: Optional[str] = Form(None)
):
    try:
        if not dataset_id:
            dataset_id = file.filename.split('.')[0].replace(' ', '_').replace('-', '_')
        if not dataset_name:
            dataset_name = file.filename

        file_extension = file.filename.split('.')[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        if file_extension == 'csv':
            result = await data_loader.load_csv(tmp_path, dataset_id, dataset_name, auto_detect=True)
        elif file_extension in ['xlsx', 'xls']:
            result = await data_loader.load_excel(tmp_path, dataset_id, dataset_name, auto_detect=True)
        elif file_extension == 'json':
            result = await data_loader.load_json(tmp_path, dataset_id, dataset_name, auto_detect=True)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")

        os.unlink(tmp_path)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Upload failed"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


# ============================================
# DATASET MANAGEMENT
# ============================================

@router.get("/datasets")
async def list_datasets():
    try:
        return data_loader.list_datasets()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schema/{dataset_id}")
async def get_schema(dataset_id: str):
    try:
        schema = data_loader.get_schema(dataset_id)
        if not schema:
            raise HTTPException(status_code=404, detail=f"Schema not found for: {dataset_id}")
        return schema.dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    try:
        deleted = data_loader.delete_dataset(dataset_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
        return {"success": True, "message": f"Dataset '{dataset_id}' deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/preview/{dataset_id}")
async def preview_dataset(dataset_id: str, limit: int = 10):
    try:
        df = data_loader.get_dataset(dataset_id)
        if df is None:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
        return {
            "dataset_id": dataset_id,
            "rows": df.head(limit).to_dict(orient='records'),
            "total_rows": len(df),
            "columns": list(df.columns)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# ANALYTICS ENDPOINTS
# ============================================

@router.post("/analyze/kpi")
async def analyze_kpi(request: AnalysisRequest):
    try:
        df = data_loader.get_dataset(request.dataset_id)
        schema = data_loader.get_schema(request.dataset_id)
        if df is None:
            raise HTTPException(status_code=404, detail="Dataset not found")

        result = await analyzer.kpi_analysis(
            df=df,
            schema=schema,
            metric_names=request.metric_names,
            comparison_period=request.comparison_period or "previous_period",
            dataset_id=request.dataset_id
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/time-series")
async def analyze_time_series(request: AnalysisRequest):
    try:
        df = data_loader.get_dataset(request.dataset_id)
        schema = data_loader.get_schema(request.dataset_id)
        if df is None:
            raise HTTPException(status_code=404, detail="Dataset not found")

        result = await analyzer.time_series_analysis(
            df=df,
            schema=schema,
            metric_names=request.metric_names,
            start_date=parse_date(request.start_date),
            end_date=parse_date(request.end_date),
            granularity=request.granularity or "day",
            dataset_id=request.dataset_id
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/breakdown")
async def analyze_breakdown(request: AnalysisRequest):
    try:
        df = data_loader.get_dataset(request.dataset_id)
        schema = data_loader.get_schema(request.dataset_id)
        if df is None:
            raise HTTPException(status_code=404, detail="Dataset not found")

        result = await analyzer.dimension_breakdown(
            df=df,
            schema=schema,
            metric_names=request.metric_names,
            dimension_name=request.dimension_name,
            top_n=request.top_n or 10,
            start_date=parse_date(request.start_date),
            end_date=parse_date(request.end_date),
            dataset_id=request.dataset_id
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/forecast")
async def analyze_forecast(request: ForecastRequest):
    try:
        df = data_loader.get_dataset(request.dataset_id)
        schema = data_loader.get_schema(request.dataset_id)
        if df is None:
            raise HTTPException(status_code=404, detail="Dataset not found")

        result = await analyzer.forecast_metric(
            df=df,
            schema=schema,
            metric_name=request.metric_name,
            periods_ahead=request.periods_ahead,
            confidence_interval=request.confidence_interval,
            include_historical=request.include_historical,
            dataset_id=request.dataset_id
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/anomalies")
async def analyze_anomalies(request: AnomalyRequest):
    try:
        df = data_loader.get_dataset(request.dataset_id)
        schema = data_loader.get_schema(request.dataset_id)
        if df is None:
            raise HTTPException(status_code=404, detail="Dataset not found")

        result = await analyzer.detect_anomalies(
            df=df,
            schema=schema,
            metric_name=request.metric_name,
            sensitivity=request.sensitivity,
            min_anomaly_score=request.min_anomaly_score,
            dataset_id=request.dataset_id
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/seasonality")
async def analyze_seasonality(request: AnalysisRequest):
    try:
        df = data_loader.get_dataset(request.dataset_id)
        schema = data_loader.get_schema(request.dataset_id)
        if df is None:
            raise HTTPException(status_code=404, detail="Dataset not found")

        metric_name = request.metric_names[0] if request.metric_names else None
        result = await analyzer.analyze_seasonality(
            df=df,
            schema=schema,
            metric_name=metric_name,
            dataset_id=request.dataset_id
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# INSIGHTS ENDPOINT
# ============================================

@router.post("/insights")
async def generate_insights(request: InsightsRequest):
    try:
        result = await insights_generator.generate_insights(
            analysis_result=request.analysis_result,
            insight_type=request.insight_type
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# DASHBOARD CONVENIENCE ENDPOINTS
# ============================================

@router.get("/dashboard/{dataset_id}")
async def get_dashboard(dataset_id: str):
    try:
        df = data_loader.get_dataset(dataset_id)
        schema = data_loader.get_schema(dataset_id)
        if df is None:
            raise HTTPException(status_code=404, detail="Dataset not found")

        kpi_result = await analyzer.kpi_analysis(
            df=df, schema=schema, dataset_id=dataset_id
        )
        return {"success": True, "dataset_id": dataset_id, "kpi": kpi_result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/kpis")
async def get_dashboard_kpis():
    """Legacy KPI endpoint — uses first loaded dataset"""
    try:
        datasets = data_loader.list_datasets()
        if not datasets["datasets"]:
            return {"total_revenue": 0, "net_profit": 0, "total_expenses": 0,
                    "active_clients": 0, "message": "No data loaded"}

        dataset_id = datasets["datasets"][0]["dataset_id"]
        df = data_loader.get_dataset(dataset_id)
        schema = data_loader.get_schema(dataset_id)

        kpi_result = await analyzer.kpi_analysis(
            df=df, schema=schema, dataset_id=dataset_id
        )
        return kpi_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# INSIGHTS MANAGEMENT (legacy routes)
# ============================================

@router.post("/insights/generate/{dataset_id}")
async def generate_insights_for_dataset(dataset_id: str, insight_type: str = "detailed"):
    try:
        df = data_loader.get_dataset(dataset_id)
        schema = data_loader.get_schema(dataset_id)
        if df is None:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

        kpi_result = await analyzer.kpi_analysis(
            df=df, schema=schema, dataset_id=dataset_id
        )
        insights = await insights_generator.generate_insights(
            analysis_result=kpi_result, insight_type=insight_type
        )
        return {"success": True, "dataset_id": dataset_id, "insights": insights}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insights/{dataset_id}")
async def get_insights(dataset_id: str):
    raise HTTPException(status_code=404, detail="Use POST /insights with analysis_result body")


@router.get("/info")
async def system_info():
    import sys
    import pandas as pd
    return {
        "python_version": sys.version,
        "pandas_version": pd.__version__,
        "gemini_configured": bool(os.getenv("GOOGLE_API_KEY")),
        "ml_available": getattr(analyzer, 'ml_available', False),
        "supported_formats": ["csv", "xlsx", "xls", "json"]
    }
