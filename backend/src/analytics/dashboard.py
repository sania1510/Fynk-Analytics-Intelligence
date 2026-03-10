"""
Analytics Dashboard Backend - FastAPI REST API
Exposes all analyzer and insights functionality as HTTP endpoints

Features:
- File upload (CSV/Excel)
- Auto schema detection
- Time series analysis
- Dimension breakdown
- KPI analysis
- Forecasting
- Anomaly detection
- Seasonality analysis
- AI insights generation

Run with: uvicorn backend:app --reload --port 8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
import pandas as pd
import os
import uuid
import asyncio

# Import your modules
from src.data.loader import DataLoader
from src.analytics.analyzer import AnalyticsAnalyzer
from src.analytics.insights import InsightsGenerator

# Initialize FastAPI app
app = FastAPI(
    title="Analytics Dashboard API",
    description="AI-powered analytics and insights API",
    version="1.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
loader = DataLoader()
analyzer = AnalyticsAnalyzer(enable_ml=True)
insights_generator = InsightsGenerator()

# In-memory storage for uploaded files (use Redis/DB in production)
uploaded_files_storage = {}

# ============================================
# REQUEST/RESPONSE MODELS
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
# UTILITY FUNCTIONS
# ============================================

def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse date string to datetime"""
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except:
        return None

# ============================================
# HEALTH CHECK
# ============================================

@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "online",
        "message": "Analytics Dashboard API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/api/upload",
            "datasets": "/api/datasets",
            "schema": "/api/schema/{dataset_id}",
            "time_series": "/api/analyze/time-series",
            "breakdown": "/api/analyze/breakdown",
            "kpi": "/api/analyze/kpi",
            "forecast": "/api/analyze/forecast",
            "anomalies": "/api/analyze/anomalies",
            "seasonality": "/api/analyze/seasonality",
            "insights": "/api/insights"
        }
    }

@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "analyzer": "ready",
            "insights_generator": "ready" if insights_generator.ai_enabled else "disabled",
            "ml_forecasting": "ready" if analyzer.ml_available else "disabled"
        }
    }

# ============================================
# FILE UPLOAD & DATASET MANAGEMENT
# ============================================

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload CSV or Excel file for analysis
    
    Returns schema and dataset info
    """
    try:
        # Generate unique dataset ID
        dataset_id = str(uuid.uuid4())
        
        # Save file temporarily
        file_extension = os.path.splitext(file.filename)[1].lower()
        temp_file_path = f"temp_{dataset_id}{file_extension}"
        
        # Write uploaded file
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Load based on file type
        if file_extension == '.csv':
            result = await loader.load_csv(
                temp_file_path,
                dataset_id,
                file.filename,
                auto_detect=True
            )
        elif file_extension in ['.xlsx', '.xls']:
            result = await loader.load_excel(
                temp_file_path,
                dataset_id,
                file.filename,
                auto_detect=True
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result.get('error', 'Upload failed'))
        
        # Store file info
        uploaded_files_storage[dataset_id] = {
            "filename": file.filename,
            "uploaded_at": datetime.utcnow().isoformat(),
            "rows": result['rows'],
            "columns": result['columns']
        }
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "message": f"Successfully uploaded {file.filename}",
            "data": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/datasets")
async def list_datasets():
    """List all uploaded datasets"""
    datasets_info = loader.list_datasets()
    
    # Enhance with upload metadata
    for dataset in datasets_info['datasets']:
        dataset_id = dataset['dataset_id']
        if dataset_id in uploaded_files_storage:
            dataset.update(uploaded_files_storage[dataset_id])
    
    return datasets_info

@app.get("/api/schema/{dataset_id}")
async def get_schema(dataset_id: str):
    """Get schema for a specific dataset"""
    schema = loader.get_schema(dataset_id)
    
    if not schema:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return schema.dict()

@app.delete("/api/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset"""
    success = loader.delete_dataset(dataset_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Remove from storage
    if dataset_id in uploaded_files_storage:
        del uploaded_files_storage[dataset_id]
    
    return {"success": True, "message": "Dataset deleted"}

# ============================================
# ANALYTICS ENDPOINTS
# ============================================

@app.post("/api/analyze/time-series")
async def analyze_time_series(request: AnalysisRequest):
    """
    Time series analysis endpoint
    
    Returns historical trends over time
    """
    try:
        df = loader.get_dataset(request.dataset_id)
        schema = loader.get_schema(request.dataset_id)
        
        if df is None or schema is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        result = await analyzer.time_series_analysis(
            df=df,
            schema=schema,
            metric_names=request.metric_names,
            start_date=parse_date(request.start_date),
            end_date=parse_date(request.end_date),
            granularity=request.granularity,
            dataset_id=request.dataset_id
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/breakdown")
async def analyze_breakdown(request: AnalysisRequest):
    """
    Dimension breakdown analysis
    
    Returns metrics broken down by dimension
    """
    try:
        df = loader.get_dataset(request.dataset_id)
        schema = loader.get_schema(request.dataset_id)
        
        if df is None or schema is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        result = await analyzer.dimension_breakdown(
            df=df,
            schema=schema,
            metric_names=request.metric_names,
            dimension_name=request.dimension_name,
            top_n=request.top_n,
            start_date=parse_date(request.start_date),
            end_date=parse_date(request.end_date),
            dataset_id=request.dataset_id
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/kpi")
async def analyze_kpi(request: AnalysisRequest):
    """
    KPI analysis with period comparison
    
    Returns current vs previous period metrics
    """
    try:
        df = loader.get_dataset(request.dataset_id)
        schema = loader.get_schema(request.dataset_id)
        
        if df is None or schema is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        result = await analyzer.kpi_analysis(
            df=df,
            schema=schema,
            metric_names=request.metric_names,
            comparison_period=request.comparison_period,
            dataset_id=request.dataset_id
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/forecast")
async def analyze_forecast(request: ForecastRequest):
    """
    ML-powered forecasting
    
    Returns future predictions using Prophet
    """
    try:
        df = loader.get_dataset(request.dataset_id)
        schema = loader.get_schema(request.dataset_id)
        
        if df is None or schema is None:
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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/anomalies")
async def analyze_anomalies(request: AnomalyRequest):
    """
    Anomaly detection
    
    Returns unusual data points
    """
    try:
        df = loader.get_dataset(request.dataset_id)
        schema = loader.get_schema(request.dataset_id)
        
        if df is None or schema is None:
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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/seasonality")
async def analyze_seasonality(request: AnalysisRequest):
    """
    Seasonality analysis
    
    Returns day-of-week and monthly patterns
    """
    try:
        df = loader.get_dataset(request.dataset_id)
        schema = loader.get_schema(request.dataset_id)
        
        if df is None or schema is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Use first metric if none specified
        metric_name = request.metric_names[0] if request.metric_names else None
        
        result = await analyzer.analyze_seasonality(
            df=df,
            schema=schema,
            metric_name=metric_name,
            dataset_id=request.dataset_id
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# INSIGHTS ENDPOINTS
# ============================================

@app.post("/api/insights")
async def generate_insights(request: InsightsRequest):
    """
    Generate AI-powered insights from analysis results
    
    Takes any analyzer result and generates natural language insights
    """
    try:
        result = await insights_generator.generate_insights(
            analysis_result=request.analysis_result,
            insight_type=request.insight_type
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# COMBINED ENDPOINTS (Convenience)
# ============================================

@app.post("/api/analyze/full-report")
async def generate_full_report(request: AnalysisRequest):
    """
    Generate complete analysis report
    
    Returns time series, breakdown, KPIs, and insights in one call
    """
    try:
        df = loader.get_dataset(request.dataset_id)
        schema = loader.get_schema(request.dataset_id)
        
        if df is None or schema is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Run all analyses
        time_series = await analyzer.time_series_analysis(
            df=df,
            schema=schema,
            metric_names=request.metric_names,
            granularity=request.granularity,
            dataset_id=request.dataset_id
        )
        
        kpi = await analyzer.kpi_analysis(
            df=df,
            schema=schema,
            metric_names=request.metric_names,
            dataset_id=request.dataset_id
        )
        
        breakdown = None
        if request.dimension_name:
            breakdown = await analyzer.dimension_breakdown(
                df=df,
                schema=schema,
                metric_names=request.metric_names,
                dimension_name=request.dimension_name,
                top_n=request.top_n,
                dataset_id=request.dataset_id
            )
        
        # Generate insights for KPIs
        insights = await insights_generator.generate_insights(
            analysis_result=kpi,
            insight_type="executive"
        )
        
        return {
            "dataset_id": request.dataset_id,
            "schema": schema.dict(),
            "time_series": time_series,
            "kpi": kpi,
            "breakdown": breakdown,
            "insights": insights
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# ERROR HANDLERS
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc),
            "status_code": 500
        }
    )

# ============================================
# STARTUP/SHUTDOWN EVENTS
# ============================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("🚀 Analytics Dashboard API starting...")
    print(f"📊 Analyzer: {'Ready' if analyzer else 'Error'}")
    print(f"🤖 Insights Generator: {'Ready' if insights_generator.ai_enabled else 'Disabled (No API key)'}")
    print(f"🔮 ML Forecasting: {'Ready' if analyzer.ml_available else 'Disabled (Prophet not installed)'}")
    print("✅ API is ready at http://localhost:8000")
    print("📖 Docs available at http://localhost:8000/docs")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("👋 Shutting down Analytics Dashboard API...")

# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
