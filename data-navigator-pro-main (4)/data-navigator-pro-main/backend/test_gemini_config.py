#!/usr/bin/env python
"""Test Gemini API Configuration"""
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("GEMINI API CONFIGURATION TEST")
print("=" * 60)

# Check environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")
gemini_model = os.getenv("GEMINI_MODEL")

print(f"\n✓ GOOGLE_API_KEY: {'✅ CONFIGURED' if google_api_key else '❌ NOT CONFIGURED'}")
print(f"  Value: {google_api_key[:20]}...{google_api_key[-10:] if google_api_key else 'N/A'}")

print(f"\n✓ GEMINI_MODEL: {'✅ CONFIGURED' if gemini_model else '❌ NOT CONFIGURED'}")
print(f"  Value: {gemini_model}")

# Test SmartSchemaDetector
print("\n" + "=" * 60)
print("TESTING SMART SCHEMA DETECTOR")
print("=" * 60)

try:
    from src.data.smart_detector import SmartSchemaDetector
    detector = SmartSchemaDetector()
    print("\n✓ SmartSchemaDetector initialized successfully")
    print(f"  Has API key in detector: {bool(detector.model)}")
except Exception as e:
    print(f"\n❌ Error initializing SmartSchemaDetector: {e}")

# Test health endpoint
print("\n" + "=" * 60)
print("TESTING HEALTH ENDPOINT")
print("=" * 60)

try:
    from src.main import app
    from fastapi.testclient import TestClient
    
    client = TestClient(app)
    response = client.get("/health")
    data = response.json()
    
    print(f"\n✓ Health Endpoint Status: {response.status_code}")
    print(f"  Gemini AI Status: {data['services']['gemini_ai']}")
    
    if data['services']['gemini_ai'] == 'configured':
        print("\n🎉 SUCCESS: Gemini API is now CONFIGURED!")
    else:
        print("\n⚠️  WARNING: Gemini API is still showing as not_configured")
        
except Exception as e:
    print(f"\n❌ Error testing health endpoint: {e}")

print("\n" + "=" * 60)
