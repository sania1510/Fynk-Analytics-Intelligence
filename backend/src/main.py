from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv
import os

# Import routers
from src.api.routes import router as api_router

# Load environment variables
load_dotenv()

# Initialize FastAPI app-----------------------------
app = FastAPI(
    title="AI Analytics Orchestrator",
    description="Enterprise-grade analytics platform powered by AI",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# ============================================
# CORS Configuration - Allow Frontend Access
# ============================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",  # Vite dev server
        "http://localhost:5173",  # Alternative Vite port
        "http://localhost:3000",  # Create React App
        "http://127.0.0.1:8080",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)

# ============================================
# Include API Routes
# ============================================
app.include_router(api_router, prefix="/api")

# ============================================
# Health Check Endpoint
# ============================================
@app.get("/")
async def root():
    """Root endpoint - Health check"""
    return {
        "status": "online",
        "service": "AI Analytics Orchestrator",
        "version": "1.0.0",
        "docs": "/api/docs"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    
    # Check if Gemini API key is configured
    gemini_configured = bool(os.getenv("GOOGLE_API_KEY"))
    
    return {
        "status": "healthy",
        "services": {
            "api": "online",
            "gemini_ai": "configured" if gemini_configured else "not_configured"
        },
        "endpoints": {
            "upload": "/api/upload",
            "datasets": "/api/datasets",
            "schema": "/api/schema/{dataset_id}",
            "analyze": "/api/analyze",
            "dashboard": "/api/dashboard/{dataset_id}"
        }
    }

# ============================================
# Error Handlers
# ============================================
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"The endpoint {request.url.path} does not exist",
            "docs": "/api/docs"
        }
    )

@app.exception_handler(500)
async def server_error_handler(request, exc):
    """Custom 500 handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again.",
            "support": "Check logs for details"
        }
    )

# ============================================
# Startup & Shutdown Events
# ============================================
@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("=" * 60)
    print(" AI Analytics Orchestrator Starting...")
    print("=" * 60)
    print(f" API Docs: http://localhost:8000/api/docs")
    print(f" Health Check: http://localhost:8000/health")
    print(f" Gemini AI: {' Configured' if os.getenv('GOOGLE_API_KEY') else ' Not Configured'}")
    print("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    print("=" * 60)
    print(" AI Analytics Orchestrator Shutting Down...")
    print("=" * 60)

# ============================================
# Run Server (for development)
# ============================================
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on file changes
        log_level="info"
    )
